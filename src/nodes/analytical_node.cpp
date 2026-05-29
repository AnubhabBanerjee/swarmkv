/**
 * @file analytical_node.cpp
 * @brief Branch DAG node: fork prefix KV at a watermark and decode a branch prompt.
 */

// analytical_node.h declares AnalyticalNode and required_prefix_tokens metadata.
#include "swarmkv/nodes/analytical_node.h"

// context_budget.h validates doc+branch token counts against n_ctx before decode.
#include "swarmkv/context_budget.h"

// llama_guard.h serializes all llama_* calls across worker threads to avoid deadlocks.
#include "swarmkv/llama_guard.h"

// kv_handoff.h implements copy-on-fork memcpy between pool buffers.
#include "swarmkv/kv_handoff.h"

// memory_pool.h allocates per-branch ggml buffers sized via llama_state_get_size.
#include "swarmkv/memory_pool.h"

// prefix_constants.h defines kSwarmkvWaitForPrefillComplete sentinel semantics.
#include "swarmkv/prefix_constants.h"

// llama.h provides batching, decode, and llama_state_seq_set_data restoration APIs.
#include <llama.h>

// chrono captures per-node timings_ms for benchmark reporting.
#include <chrono>

// mutex guards PipelineState::node_outputs and timings_ms maps.
#include <mutex>

// sstream formats a small diagnostic string stored in node_outputs.
#include <sstream>

// stdexcept throws when snapshots, budgets, or decode steps fail.
#include <stdexcept>

// string stores branch prompts and orchestrator node name keys.
#include <string>

// vector holds tokenized branch prompt ids.
#include <vector>

// cstring provides std::memcpy when copying immutable milestone snapshots into branch buffers.
#include <cstring>

// AnalyticalNode stores the branch prompt and the watermark requirement for orchestrator gating.
AnalyticalNode::AnalyticalNode(llama_model * /*model*/, const std::string & prompt, int32_t required_prefix_tokens)
    : branch_prompt(prompt),
      required_prefix_tokens_(required_prefix_tokens) {}

// required_prefix_tokens exposes W to the orchestrator before worker threads are spawned.
int32_t AnalyticalNode::required_prefix_tokens() const {
    // Return the stored watermark requirement unchanged for orchestrator milestone registration.
    return required_prefix_tokens_;
}

// tokenize_text converts UTF-8 branch prompts into llama_token ids for batch construction.
static std::vector<llama_token> tokenize_text(llama_model * model, const std::string & text) {
    // Fetch the vocabulary object tied to the loaded model weights.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // Pre-size a vector generously because special tokens can expand the true count.
    std::vector<llama_token> tokens(text.size() + 8u);
    // First tokenize attempt using the caller-provided buffer capacity.
    int32_t n_tokens = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false);
    // Negative return means insufficient capacity; resize and retry once.
    if (n_tokens < 0) {
        // Resize to the exact demand reported by llama_tokenize on the first pass.
        tokens.resize(static_cast<size_t>(-n_tokens));
        // Retry tokenization after resizing so we do not fail on long prompts.
        n_tokens = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            false);
        // If the second attempt still fails, treat this as a hard configuration error.
        if (n_tokens < 0) {
            // Throw so the orchestrator promise path records the failure for dependents.
            throw std::runtime_error("AnalyticalNode: tokenize failed after resize.");
        }
    }
    // Shrink the vector to the exact number of valid token ids returned by llama_tokenize.
    tokens.resize(static_cast<size_t>(n_tokens));
    // Return the finalized token id vector to the caller for batch construction.
    return tokens;
}

// execute forks immutable prefix KV and decodes the branch prompt starting at prefix_seq_len.
void AnalyticalNode::execute(PipelineState * state, OrchestratorContext * ctx) {
    LlamaGuard guard;
    // Start the wall-clock timer used for timings_ms reporting in benchmarks.
    const auto t0 = std::chrono::steady_clock::now();
    // MemoryPool must exist so each branch can allocate an independent fork buffer.
    if (!ctx->memory_pool) {
        // Throw before allocating contexts if orchestrator wiring forgot the pool pointer.
        throw std::runtime_error("AnalyticalNode: null MemoryPool in OrchestratorContext.");
    }
    // fork_prefix_len becomes the RoPE offset used for branch token positions.
    int32_t fork_prefix_len = 0;
    // fork_kv_bytes counts how many serialized bytes must be copied/restored for this fork.
    size_t fork_kv_bytes = 0;
    // fork_data points at immutable snapshot bytes (milestone map or canonical buffer).
    const uint8_t * fork_data = nullptr;
    // Choose snapshot source depending on whether this branch uses watermark forking or full prefill.
    if (required_prefix_tokens_ >= 0) {
        // Watermark branch: read the frozen snapshot captured exactly at W tokens.
        if (!state->try_get_prefix_snapshot(
                required_prefix_tokens_,
                &fork_data,
                &fork_kv_bytes,
                &fork_prefix_len)) {
            // Throw if the prefiller never registered or published the milestone snapshot.
            throw std::runtime_error(
                "AnalyticalNode: missing prefix snapshot at watermark " +
                std::to_string(required_prefix_tokens_) + ".");
        }
    } else {
        // Full-prefill branch: use the final canonical export published by PrefillNode.
        fork_prefix_len = state->prefix_seq_len;
        // Copy the byte count from the latest canonical llama_state_seq_get_data export.
        fork_kv_bytes = state->prefix_kv_serial_bytes;
        // Point at canonical buffer bytes only when an export actually occurred.
        if (fork_kv_bytes > 0) {
            // Validate canonical buffer presence before dereferencing host memory.
            if (!state->materialized_branch_buffer) {
                // Throw if bytes are claimed but the staging buffer pointer is null.
                throw std::runtime_error("AnalyticalNode: exported prefix bytes but no canonical buffer.");
            }
            // Resolve the host base pointer for reading canonical serialized KV bytes.
            fork_data = static_cast<const uint8_t *>(
                ggml_backend_buffer_get_base(state->materialized_branch_buffer));
        }
    }
    // branch_buf holds the per-branch copy that will be passed to llama_state_seq_set_data.
    ggml_backend_buffer_t branch_buf = nullptr;
    // Allocate and memcpy fork bytes when there is KV state to restore on this branch context.
    if (fork_kv_bytes > 0) {
        // Validate fork_data is non-null whenever byte count is positive.
        if (!fork_data) {
            // Throw to avoid calling memcpy with a null source pointer.
            throw std::runtime_error("AnalyticalNode: fork_kv_bytes > 0 but fork_data is null.");
        }
        // Allocate a branch buffer sized for n_ctx so later decode has headroom in the same blob policy.
        branch_buf = ctx->memory_pool->allocate_branch_cache(static_cast<uint32_t>(ctx->ctx_params.n_ctx));
        // If canonical buffer exists, memcpy from canonical; otherwise memcpy from milestone vector via fork_data.
        if (state->materialized_branch_buffer && required_prefix_tokens_ < 0) {
            // Full-prefill path copies from canonical staging into the branch allocation.
            KVHandoff::materialize_branch_cache(
                state->materialized_branch_buffer,
                branch_buf,
                fork_kv_bytes);
        } else {
            // Watermark path copies immutable milestone bytes into the branch allocation directly.
            void * dst = ggml_backend_buffer_get_base(branch_buf);
            // Ensure destination capacity can hold the fork snapshot byte length.
            if (ggml_backend_buffer_get_size(branch_buf) < fork_kv_bytes) {
                // Throw if MemoryPool sizing is smaller than the milestone export footprint.
                throw std::runtime_error("AnalyticalNode: branch buffer smaller than fork_kv_bytes.");
            }
            // Copy immutable snapshot bytes into the branch buffer host mapping.
            std::memcpy(dst, fork_data, fork_kv_bytes);
        }
    }
    // Create an isolated llama_context for this branch decode under orchestrator ctx params.
    llama_context * lctx = llama_init_from_model(ctx->model, ctx->ctx_params);
    // Fail fast if llama cannot construct a context (OOM or invalid params).
    if (!lctx) {
        // Throw so the orchestrator records the failure and does not hang dependents.
        throw std::runtime_error("AnalyticalNode: llama_init_from_model returned null.");
    }
    // Restore forked KV into the branch context when serialized bytes are available.
    if (fork_kv_bytes > 0) {
        // Resolve host pointer for llama_state_seq_set_data on the branch buffer.
        void * base = ggml_backend_buffer_get_base(branch_buf);
        // Restore only the prefix sequence lane so branch decode stays isolated on seq 0.
        const size_t n = llama_state_seq_set_data(
            lctx,
            static_cast<const uint8_t *>(base),
            fork_kv_bytes,
            kSwarmkvPrefixSeqId);
        // Verify llama consumed exactly the number of bytes we copied into the branch buffer.
        if (n != fork_kv_bytes) {
            // Free the context before throwing to avoid leaking VRAM on failure paths.
            llama_free(lctx);
            // Throw with a clear message so operators can debug size mismatches quickly.
            throw std::runtime_error("AnalyticalNode: llama_state_seq_set_data size mismatch.");
        }
    }
    // bind_target preserves the V1 control-flow call site for future external KV bind APIs.
    ggml_backend_buffer_t bind_target = branch_buf ? branch_buf : state->materialized_branch_buffer;
    // Invoke bind (currently a documented no-op) so all nodes share identical setup steps.
    KVHandoff::bind_contiguous_cache(lctx, bind_target);
    // Tokenize the branch prompt into llama_token ids for batch construction.
    std::vector<llama_token> tokens = tokenize_text(ctx->model, branch_prompt);
    // Reject empty prompts because decode would not advance branch state meaningfully.
    if (tokens.empty()) {
        // Free context before throwing to keep shutdown paths leak-free.
        llama_free(lctx);
        // Throw so misconfigured prompts are caught during examples and benchmarks.
        throw std::runtime_error("AnalyticalNode: empty prompt after tokenization.");
    }
    // Validate that fork_prefix_len plus branch tokens fit within n_ctx headroom policy.
    try {
        // swarmkv_validate_context_budget throws when doc+branch would exceed n_ctx.
        swarmkv_validate_context_budget(
            ctx->model,
            fork_prefix_len,
            {branch_prompt},
            ctx->ctx_params.n_ctx);
    } catch (const std::exception & e) {
        // Free context before rethrowing so failures do not leak llama contexts.
        llama_free(lctx);
        // Wrap the budget error with AnalyticalNode context for easier log attribution.
        throw std::runtime_error(std::string("AnalyticalNode: ") + e.what());
    }
    // Allocate a llama_batch sized exactly for the branch prompt token count.
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    // Set the number of tokens in this decode step to the branch prompt length.
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    // Fill per-token fields: ids, RoPE positions, and sequence lane assignment.
    for (int i = 0; i < batch.n_tokens; ++i) {
        // Copy the i-th branch token id into the batch slot.
        batch.token[i] = tokens[static_cast<size_t>(i)];
        // Place branch tokens immediately after the forked prefix positions for correct RoPE.
        batch.pos[i] = static_cast<llama_pos>(fork_prefix_len) + static_cast<llama_pos>(i);
        // Each token participates in exactly one sequence id list entry.
        batch.n_seq_id[i] = 1;
        // Bind all branch tokens to the shared prefix sequence lane constant.
        batch.seq_id[i][0] = kSwarmkvPrefixSeqId;
        // Disable logits for all tokens except the last one in this branch step.
        batch.logits[i] = 0;
    }
    // Enable logits on the final branch token so downstream consumers can sample.
    batch.logits[batch.n_tokens - 1] = 1;
    // Run a single branch decode step that extends KV beyond the forked prefix.
    if (llama_decode(lctx, batch) != 0) {
        // Free batch and context before throwing on decode failure.
        llama_batch_free(batch);
        llama_free(lctx);
        // Throw so orchestrator marks this node failed and does not hang dependents forever.
        throw std::runtime_error("AnalyticalNode: llama_decode failed.");
    }
    // Publish a small diagnostic string into PipelineState under this node's orchestrator name.
    {
        // Lock node_outputs because multiple branches may finish around the same time.
        std::lock_guard<std::mutex> lock(state->output_lock);
        // Build a human-readable status line for examples and quick printf debugging.
        std::ostringstream oss;
        // Include branch token count, fork prefix length, and whether watermark forking was used.
        oss << "branch_ok n_tok=" << tokens.size() << " prefix=" << fork_prefix_len
            << " watermark=" << required_prefix_tokens_;
        // Resolve the output map key from orchestrator context (falls back to generic label).
        const char * key = ctx->node_name ? ctx->node_name : "analytical";
        // Store the formatted output string for the main thread to print after join.
        state->node_outputs[std::string(key)] = oss.str();
    }
    // Free the llama_batch object now that decode has completed successfully.
    llama_batch_free(batch);
    // Free the branch llama_context to release VRAM before returning to the orchestrator join barrier.
    llama_free(lctx);
    // Tell the prefiller it may resume extending the document past this milestone watermark.
    if (required_prefix_tokens_ > 0) {
        state->signal_milestone_consumed(required_prefix_tokens_);
    }
    // Stop the wall-clock timer and compute elapsed milliseconds for timings_ms reporting.
    const auto t1 = std::chrono::steady_clock::now();
    // Convert steady_clock delta into integer milliseconds for JSON-friendly metrics.
    const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    // Record per-node timing under the orchestrator node name for benchmark scraping.
    {
        // Lock timings_ms because multiple nodes may write concurrently at pipeline tail.
        std::lock_guard<std::mutex> lock(state->timing_lock);
        // Resolve timing map key from orchestrator context node name pointer.
        const char * key = ctx->node_name ? ctx->node_name : "analytical";
        // Store elapsed milliseconds for this branch execution.
        state->timings_ms[std::string(key)] = ms;
    }
}
