/**
 * @file swarmkv_benchmark.cpp
 * @brief Baseline vs SwarmKV harness for 4K-context analytical pipelines (GTX 1080 class).
 */

#include "swarmkv/context_budget.h"
#include "swarmkv/memory_pool.h"
#include "swarmkv/nodes/analytical_node.h"
#include "swarmkv/nodes/prefill_node.h"
#include "swarmkv/orchestrator.h"
#include "swarmkv/pipeline_state.h"
#include "swarmkv/model_params.h"
#include "swarmkv/defaults.h"
#include "swarmkv/prefix_constants.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr const char * kBranchPromptA = "Summarize security posture.";
static constexpr const char * kBranchPromptB = "Note license obligations.";

// Heterogeneous V2 watermarks: AgentA starts at 512 tokens, AgentB at 3000 (document continues to ~3500).
static constexpr int32_t kSwarmkvAgentAWatermark = 512;
static constexpr int32_t kSwarmkvAgentBWatermark = 3000;

static std::vector<llama_token> tokenize_all(llama_model * model, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 8u);
    int32_t n = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            false);
        if (n < 0) {
            throw std::runtime_error("tokenize failed");
        }
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

static std::string build_document(llama_model * model, int target_tokens) {
    const std::string chunk = "The quick brown fox jumps over the lazy dog. ";
    std::string doc;
    while (static_cast<int>(tokenize_all(model, doc).size()) < target_tokens) {
        doc += chunk;
        if (doc.size() > 8000000) {
            throw std::runtime_error("build_document: target too large for safety cap.");
        }
    }
    return doc;
}

static void decode_tokens_chunked(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const int32_t n_batch = llama_n_batch(ctx);
    int32_t offset = 0;
    while (offset < static_cast<int32_t>(tokens.size())) {
        const int32_t chunk = std::min(n_batch, static_cast<int32_t>(tokens.size()) - offset);
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        batch.n_tokens = chunk;
        for (int32_t i = 0; i < chunk; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(offset + i)];
            batch.pos[i] = offset + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }
        batch.logits[chunk - 1] = 1;
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed in baseline prefill.");
        }
        llama_batch_free(batch);
        offset += chunk;
    }
}

struct BaselineAgentTimings {
    int64_t doc_prefill_ms = 0;
    int64_t branch_decode_ms = 0;
    int64_t total_ms = 0;
};

static BaselineAgentTimings run_baseline_agent(
    llama_model * model,
    const std::vector<llama_token> & doc_tokens,
    llama_context_params p,
    const std::string & branch_prompt,
    const char * label) {
    BaselineAgentTimings out{};
    const auto total_start = std::chrono::steady_clock::now();

    llama_context * lctx = llama_init_from_model(model, p);
    if (!lctx) {
        throw std::runtime_error("baseline: llama_init_from_model failed.");
    }
    llama_perf_context_reset(lctx);
    std::cerr << "--- " << label << " (llama_perf_context_print follows) ---\n";

    const auto prefill_start = std::chrono::steady_clock::now();
    decode_tokens_chunked(lctx, doc_tokens);
    const auto prefill_end = std::chrono::steady_clock::now();
    out.doc_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();

    std::vector<llama_token> tail_tok = tokenize_all(model, branch_prompt);
    const auto branch_start = std::chrono::steady_clock::now();
    llama_batch tb = llama_batch_init(static_cast<int32_t>(tail_tok.size()), 0, 1);
    tb.n_tokens = static_cast<int32_t>(tail_tok.size());
    const int32_t base_pos = static_cast<int32_t>(doc_tokens.size());
    for (int32_t i = 0; i < tb.n_tokens; ++i) {
        tb.token[i] = tail_tok[static_cast<size_t>(i)];
        tb.pos[i] = base_pos + i;
        tb.n_seq_id[i] = 1;
        tb.seq_id[i][0] = 0;
        tb.logits[i] = 0;
    }
    tb.logits[tb.n_tokens - 1] = 1;
    if (llama_decode(lctx, tb) != 0) {
        llama_batch_free(tb);
        llama_free(lctx);
        throw std::runtime_error("baseline: branch decode failed.");
    }
    llama_batch_free(tb);
    const auto branch_end = std::chrono::steady_clock::now();
    out.branch_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(branch_end - branch_start).count();

    llama_perf_context_print(lctx);
    llama_free(lctx);

    const auto total_end = std::chrono::steady_clock::now();
    out.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    return out;
}

static void emit_metric(const char * key, int64_t value) {
    std::cout << key << ": " << value << '\n' << std::flush;
}

static void emit_metric(const char * key, double value) {
    std::cout << key << ": " << std::fixed << std::setprecision(2) << value << '\n' << std::flush;
}

static void emit_metric(const char * key, const std::string & value) {
    std::cout << key << ": " << value << '\n' << std::flush;
}

/** V2 heterogeneous harness: prefiller thread + main-thread branches at 512 and 3000 watermarks. */
static void run_swarm_v2_heterogeneous(
    llama_model * model,
    llama_context_params p,
    const std::string & base_doc_path,
    PipelineState * state,
    MemoryPool * pool,
    int64_t * out_e2e_ms,
    int64_t * out_prefill_ms,
    int64_t * out_agent_a_ms,
    int64_t * out_agent_b_ms,
    int64_t * out_branch1_ttft_proxy_ms,
    int64_t * out_branch2_ttft_proxy_ms) {
    state->materialized_branch_buffer = pool->allocate_prefix_cache(static_cast<uint32_t>(p.n_ctx));
    state->register_snapshot_milestone(kSwarmkvAgentAWatermark);
    state->register_snapshot_milestone(kSwarmkvAgentBWatermark);
    state->register_milestone_consumer(kSwarmkvAgentAWatermark);
    state->register_milestone_consumer(kSwarmkvAgentBWatermark);

    std::exception_ptr prefill_ex;
    const auto e2e0 = std::chrono::steady_clock::now();
    std::thread prefill_thread([&]() {
        try {
            PrefillNode prefill(model, base_doc_path);
            OrchestratorContext pctx = {model, p, pool, "Prefiller"};
            prefill.execute(state, &pctx);
        } catch (...) {
            prefill_ex = std::current_exception();
        }
    });

    state->wait_for_watermark(kSwarmkvAgentAWatermark);
    const auto ttft_a0 = std::chrono::steady_clock::now();
    AnalyticalNode agent_a(model, kBranchPromptA, kSwarmkvAgentAWatermark);
    OrchestratorContext actx = {model, p, pool, "AgentA"};
    agent_a.execute(state, &actx);
    const auto ttft_a1 = std::chrono::steady_clock::now();

    state->wait_for_watermark(kSwarmkvAgentBWatermark);
    const auto ttft_b0 = std::chrono::steady_clock::now();
    AnalyticalNode agent_b(model, kBranchPromptB, kSwarmkvAgentBWatermark);
    OrchestratorContext bctx = {model, p, pool, "AgentB"};
    agent_b.execute(state, &bctx);
    const auto ttft_b1 = std::chrono::steady_clock::now();

    prefill_thread.join();
    if (prefill_ex) {
        std::rethrow_exception(prefill_ex);
    }

    const auto e2e1 = std::chrono::steady_clock::now();
    *out_e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e2e1 - e2e0).count();
    *out_branch1_ttft_proxy_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(ttft_a1 - e2e0).count();
    *out_branch2_ttft_proxy_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(ttft_b1 - e2e0).count();
    *out_prefill_ms = state->timings_ms.count("Prefiller") ? state->timings_ms.at("Prefiller") : 0;
    *out_agent_a_ms = state->timings_ms.count("AgentA") ? state->timings_ms.at("AgentA") : 0;
    *out_agent_b_ms = state->timings_ms.count("AgentB") ? state->timings_ms.at("AgentB") : 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "swarmkv_benchmark")
                  << " <model.gguf> [approx_prefill_tokens] [base_doc_path]\n";
        return 1;
    }

    int target_tokens = 3500;
    if (argc >= 3) {
        target_tokens = std::atoi(argv[2]);
        if (target_tokens < 8) {
            target_tokens = 8;
        }
    }

    std::string base_doc_path = "examples/base_doc.txt";
    if (argc >= 4) {
        base_doc_path = argv[3];
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], swarmkv_default_model_params());
    if (!model) {
        std::cerr << "Failed to load model.\n";
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> doc_tokens;
    try {
        const std::string doc = build_document(model, target_tokens);
        doc_tokens = tokenize_all(model, doc);
        {
            std::ofstream out(base_doc_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("failed to write base_doc at " + base_doc_path);
            }
            out.write(doc.data(), static_cast<std::streamsize>(doc.size()));
        }
    } catch (const std::exception & e) {
        std::cerr << "build_document: " << e.what() << '\n';
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params p = llama_context_default_params();
    p.n_ctx = kSwarmkvDefaultPipelineCtx;

    const int32_t doc_n = static_cast<int32_t>(doc_tokens.size());
    std::vector<llama_token> branch_a_tok = tokenize_all(model, kBranchPromptA);
    std::vector<llama_token> branch_b_tok = tokenize_all(model, kBranchPromptB);
    const int32_t max_branch = std::max(static_cast<int32_t>(branch_a_tok.size()), static_cast<int32_t>(branch_b_tok.size()));
    const int32_t headroom = 128;

    if (doc_n < kSwarmkvAgentBWatermark) {
        std::cerr << "Document has " << doc_n << " tokens; need at least " << kSwarmkvAgentBWatermark
                  << " for heterogeneous AgentB watermark.\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    try {
        swarmkv_validate_context_budget(
            model,
            doc_n,
            {kBranchPromptA, kBranchPromptB},
            p.n_ctx,
            headroom);
        swarmkv_validate_context_budget(
            model,
            kSwarmkvAgentAWatermark,
            {kBranchPromptA},
            p.n_ctx,
            headroom);
        swarmkv_validate_context_budget(
            model,
            kSwarmkvAgentBWatermark,
            {kBranchPromptB},
            p.n_ctx,
            headroom);
    } catch (const std::exception & e) {
        std::cerr << "Context budget check failed: " << e.what() << '\n';
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    emit_metric("Context_Label", "4K Context (n_ctx=4096)");
    emit_metric("Document_Tokens", static_cast<int64_t>(doc_n));
    emit_metric("Target_Prefill_Tokens", static_cast<int64_t>(target_tokens));
    emit_metric("Base_Doc_Path", base_doc_path);
    emit_metric("BranchA_Tokens", static_cast<int64_t>(branch_a_tok.size()));
    emit_metric("BranchB_Tokens", static_cast<int64_t>(branch_b_tok.size()));
    emit_metric("SwarmKV_AgentA_Required_Prefix_Tokens", static_cast<int64_t>(kSwarmkvAgentAWatermark));
    emit_metric("SwarmKV_AgentB_Required_Prefix_Tokens", static_cast<int64_t>(kSwarmkvAgentBWatermark));

    std::cerr << "swarmkv_benchmark: n_ctx=" << p.n_ctx << " document_tokens=" << doc_n << '\n';

    BaselineAgentTimings agent1{};
    BaselineAgentTimings agent2{};
    int64_t baseline_e2e_ms = 0;
    const bool skip_baseline = std::getenv("SWARMKV_SKIP_BASELINE") != nullptr;
    if (!skip_baseline) {
        std::cerr << "--- BASELINE: two sequential full-prefill agents (4K context) ---\n";
        const auto baseline_start = std::chrono::steady_clock::now();
        agent1 = run_baseline_agent(model, doc_tokens, p, kBranchPromptA, "BASELINE agent 1");
        agent2 = run_baseline_agent(model, doc_tokens, p, kBranchPromptB, "BASELINE agent 2");
        const auto baseline_end = std::chrono::steady_clock::now();
        baseline_e2e_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(baseline_end - baseline_start).count();
    } else {
        std::cerr << "--- BASELINE: skipped (SWARMKV_SKIP_BASELINE set) ---\n";
    }

    emit_metric("Baseline_Agent1_Prefill_Ms", agent1.doc_prefill_ms);
    emit_metric("Baseline_Agent1_Branch_Ms", agent1.branch_decode_ms);
    emit_metric("Baseline_Agent1_Total_Ms", agent1.total_ms);
    emit_metric("Baseline_Agent2_Prefill_Ms", agent2.doc_prefill_ms);
    emit_metric("Baseline_Agent2_Branch_Ms", agent2.branch_decode_ms);
    emit_metric("Baseline_Agent2_Total_Ms", agent2.total_ms);
    emit_metric("Baseline_End_To_End_Ms", baseline_e2e_ms);
    emit_metric("Baseline_Branch2_TTFT_Proxy_Ms", agent2.doc_prefill_ms);

    std::cerr << "--- SWARMKV V2: heterogeneous watermarks (512 + 3000) with overlapped prefill ---\n";
    PipelineState state{};
    int64_t swarm_e2e_ms = 0;
    int64_t swarm_prefill_ms = 0;
    int64_t swarm_a_ms = 0;
    int64_t swarm_b_ms = 0;
    int64_t swarm_branch1_ttft_proxy_ms = 0;
    int64_t swarm_branch2_ttft_proxy_ms = 0;
    try {
        MemoryPool pool(model);
        run_swarm_v2_heterogeneous(
            model,
            p,
            base_doc_path,
            &state,
            &pool,
            &swarm_e2e_ms,
            &swarm_prefill_ms,
            &swarm_a_ms,
            &swarm_b_ms,
            &swarm_branch1_ttft_proxy_ms,
            &swarm_branch2_ttft_proxy_ms);
        pool.free_all();
    } catch (const std::exception & e) {
        std::cerr << "SwarmKV path failed: " << e.what() << '\n';
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int64_t swarm_branch_avg_ms = (swarm_a_ms + swarm_b_ms) / 2;

    emit_metric("SwarmKV_Prefill_Ms", swarm_prefill_ms);
    emit_metric("SwarmKV_AgentA_Ms", swarm_a_ms);
    emit_metric("SwarmKV_AgentB_Ms", swarm_b_ms);
    emit_metric("SwarmKV_Branch_Avg_Ms", swarm_branch_avg_ms);
    emit_metric("SwarmKV_Branch1_TTFT_Proxy_Ms", swarm_branch1_ttft_proxy_ms);
    emit_metric("SwarmKV_Branch2_TTFT_Proxy_Ms", swarm_branch2_ttft_proxy_ms);
    emit_metric("SwarmKV_End_To_End_Ms", swarm_e2e_ms);

    const double e2e_improvement_pct =
        baseline_e2e_ms > 0
            ? (100.0 * static_cast<double>(baseline_e2e_ms - swarm_e2e_ms) / static_cast<double>(baseline_e2e_ms))
            : 0.0;
    const double ttft_improvement_pct =
        agent2.doc_prefill_ms > 0
            ? (100.0 * static_cast<double>(agent2.doc_prefill_ms - swarm_branch2_ttft_proxy_ms) /
               static_cast<double>(agent2.doc_prefill_ms))
            : 0.0;
    const double ttft_a_improvement_pct =
        agent1.doc_prefill_ms > 0
            ? (100.0 * static_cast<double>(agent1.doc_prefill_ms - swarm_branch1_ttft_proxy_ms) /
               static_cast<double>(agent1.doc_prefill_ms))
            : 0.0;
    const double redundant_prefill_eliminated_ms =
        static_cast<double>(agent1.doc_prefill_ms + agent2.doc_prefill_ms - swarm_prefill_ms);

    emit_metric("End_To_End_Improvement_Pct", e2e_improvement_pct);
    emit_metric("Branch1_TTFT_Improvement_Pct", ttft_a_improvement_pct);
    emit_metric("Branch2_TTFT_Improvement_Pct", ttft_improvement_pct);
    emit_metric("Redundant_Prefill_Eliminated_Ms", static_cast<int64_t>(redundant_prefill_eliminated_ms));

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
