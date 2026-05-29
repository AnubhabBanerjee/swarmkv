/**
 * @file prefill_node.cpp
 * @brief Root DAG node: chunked document prefill with readiness watermark publication (V2).
 */

// prefill_node.h declares PrefillNode and is_prefill_provider metadata for orchestrator gating.
#include "swarmkv/nodes/prefill_node.h"

// context_budget.h is available for future strict document sizing checks at prefill time.
#include "swarmkv/context_budget.h"

// llama_guard.h serializes llama API calls across orchestrator worker threads.
#include "swarmkv/llama_guard.h"

// kv_handoff.h preserves the bind_contiguous_cache call site for future external KV bind APIs.
#include "swarmkv/kv_handoff.h"

// prefix_constants.h defines chunk size and prefix sequence id constants for export.
#include "swarmkv/prefix_constants.h"

// llama.h provides tokenization, batching, decode, and sequence-state export APIs.
#include <llama.h>

// vector stores tokenized document ids for chunked decode loops.
#include <vector>

// stdexcept throws on decode failures, missing buffers, and invalid configuration.
#include <stdexcept>

// algorithm provides std::min for chunk sizing against llama_n_batch limits.
#include <algorithm>

// chrono measures PrefillNode wall time for timings_ms reporting.
#include <chrono>

// fstream reads the shared document bytes from disk when the path is readable.
#include <fstream>

// PrefillNode remembers which filesystem path contains the shared document body.
PrefillNode::PrefillNode(llama_model * /*model*/, const std::string & path)
    : source_document_path(path) {}

// tokenize_text converts UTF-8 document text into llama_token ids for chunked prefill.
static std::vector<llama_token> tokenize_text(llama_model * model, const std::string & text) {
    // Resolve vocabulary from the loaded model for tokenization.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // Pre-allocate token buffer with a small slack for special tokens.
    std::vector<llama_token> tokens(text.size() + 8u);
    // First tokenize attempt using current vector capacity.
    int32_t n_tokens = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false);
    // Negative return indicates insufficient capacity; resize and retry once.
    if (n_tokens < 0) {
        // Resize vector to exact required capacity reported by llama_tokenize.
        tokens.resize(static_cast<size_t>(-n_tokens));
        // Retry tokenization after resizing to avoid failing on long documents.
        n_tokens = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            false);
        // If still negative, treat as a hard failure for the root prefill node.
        if (n_tokens < 0) {
            // Throw so orchestrator records failure and dependent branches do not hang.
            throw std::runtime_error("PrefillNode: tokenize failed after resize.");
        }
    }
    // Shrink vector to the true number of token ids returned by llama_tokenize.
    tokens.resize(static_cast<size_t>(n_tokens));
    // Return finalized document token ids to the caller.
    return tokens;
}

// load_document_text reads the configured path or returns a small built-in fallback string.
static std::string load_document_text(const std::string & path) {
    // Attempt to open the configured document path as an input file stream.
    std::ifstream in(path);
    // If the file opens successfully, read the entire contents into a string.
    if (in.good()) {
        // Use stream iterators to load all bytes without manual size accounting.
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    // Fallback body keeps demos runnable when examples/base_doc.txt is missing.
    return "SwarmKV demo document body for path: " + path;
}

// decode_token_range decodes tokens[offset, offset+count) on the prefix sequence lane.
static void decode_token_range(
    llama_context * lctx,
    const std::vector<llama_token> & tokens,
    int32_t offset,
    int32_t count) {
    // Respect llama_n_batch so each llama_decode call satisfies n_tokens <= n_batch.
    const int32_t n_batch = llama_n_batch(lctx);
    // Track how many tokens remain to decode within this [offset, offset+count) slice.
    int32_t remaining = count;
    // Current absolute offset inside the full document token array.
    int32_t cur = offset;
    // Loop until the slice is fully prefilled into KV.
    while (remaining > 0) {
        // Choose a chunk no larger than n_batch and no larger than remaining slice length.
        const int32_t chunk = std::min(n_batch, remaining);
        // Allocate a batch object sized for this chunk on sequence lane 0.
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        // Set number of tokens in this decode call.
        batch.n_tokens = chunk;
        // Fill per-token fields for positions, sequence ids, and logits flags.
        for (int32_t i = 0; i < chunk; ++i) {
            // Copy token id from the document vector at absolute index cur+i.
            batch.token[i] = tokens[static_cast<size_t>(cur + i)];
            // Absolute RoPE position equals index in the full document token stream.
            batch.pos[i] = cur + i;
            // Each token belongs to exactly one sequence id list.
            batch.n_seq_id[i] = 1;
            // Bind all document tokens to the shared prefix sequence lane constant.
            batch.seq_id[i][0] = kSwarmkvPrefixSeqId;
            // Disable logits during prefill except we keep zeros for all tokens here.
            batch.logits[i] = 0;
        }
        // Run llama_decode for this chunk; non-zero indicates a hard engine failure.
        if (llama_decode(lctx, batch) != 0) {
            // Free batch before throwing so llama batch allocations are not leaked.
            llama_batch_free(batch);
            // Throw to abort PrefillNode and signal orchestrator failure to dependents.
            throw std::runtime_error("PrefillNode: llama_decode failed during chunked prefill.");
        }
        // Free batch memory now that this chunk decode completed successfully.
        llama_batch_free(batch);
        // Advance absolute offset by the number of tokens decoded in this chunk.
        cur += chunk;
        // Decrease remaining tokens left in this slice.
        remaining -= chunk;
    }
}

// restore_prefix_snapshot loads immutable milestone KV back into a fresh prefill context.
static void restore_prefix_snapshot(llama_context * lctx, PipelineState * state, int32_t watermark_w) {
    const uint8_t * data = nullptr;
    size_t nbytes = 0;
    int32_t seq_len = 0;
    if (!state->try_get_prefix_snapshot(watermark_w, &data, &nbytes, &seq_len)) {
        throw std::runtime_error(
            "PrefillNode: missing snapshot to resume prefill at watermark " + std::to_string(watermark_w));
    }
    if (nbytes == 0) {
        return;
    }
    const size_t nr = llama_state_seq_set_data(
        lctx,
        data,
        nbytes,
        kSwarmkvPrefixSeqId);
    if (nr != nbytes) {
        throw std::runtime_error("PrefillNode: llama_state_seq_set_data resume size mismatch.");
    }
}

// execute performs V2 chunked prefill, commits watermarks, and marks prefill_complete at the end.
void PrefillNode::execute(PipelineState * state, OrchestratorContext * ctx) {
    // Start wall-clock timer for timings_ms reporting (benchmark reads Prefiller key).
    const auto t0 = std::chrono::steady_clock::now();
    // Create an isolated llama context for root prefill using orchestrator hyperparameters.
    llama_context * lctx = llama_init_from_model(ctx->model, ctx->ctx_params);
    // Fail fast if context creation fails (OOM or invalid ctx params).
    if (!lctx) {
        // Throw so orchestrator records the failure and wakes any waiting dependents.
        throw std::runtime_error("PrefillNode: llama_init_from_model returned null.");
    }
    // Preserve bind call shape for future external KV bind support (currently a no-op).
    KVHandoff::bind_contiguous_cache(lctx, state->materialized_branch_buffer);
    // Load shared document bytes from disk (or fallback) for tokenization.
    const std::string file_content = load_document_text(source_document_path);
    // Tokenize the entire document once; chunked decode consumes slices of this vector.
    std::vector<llama_token> tokens = tokenize_text(ctx->model, file_content);
    // Reject documents that do not fit within configured n_ctx capacity.
    if (static_cast<int32_t>(tokens.size()) >= static_cast<int32_t>(ctx->ctx_params.n_ctx)) {
        // Free context before throwing so we do not leak VRAM on validation failure.
        llama_free(lctx);
        // Throw with counts so operators can adjust n_ctx or shorten the document file.
        throw std::runtime_error(
            "PrefillNode: document token count (" + std::to_string(tokens.size()) +
            ") exceeds n_ctx=" + std::to_string(ctx->ctx_params.n_ctx));
    }
    // Track how many document tokens have been prefilled so far for watermark publication.
    int32_t prefilled = 0;
    // Total number of document tokens to prefill before marking prefill_complete.
    const int32_t total = static_cast<int32_t>(tokens.size());
    // V2 loop: decode fixed-size readiness chunks and commit watermark after each chunk.
    while (prefilled < total) {
        // Compute how many tokens remain until the full document has been prefilled.
        const int32_t remaining = total - prefilled;
        // Choose chunk size as min(128, remaining) so the final chunk may be smaller than 128.
        const int32_t chunk = std::min(kSwarmkvPrefillChunkTokens, remaining);
        // Hold LlamaGuard only per chunk so branches can fork while prefill continues (V2 overlap).
        {
            LlamaGuard guard;
            // Decode this slice starting at absolute offset prefilled for chunk tokens.
            decode_token_range(lctx, tokens, prefilled, chunk);
            // Advance prefilled count by the number of tokens decoded in this slice.
            prefilled += chunk;
            // Publish readiness watermark and export prefix-seq KV for branch subscribers.
            state->commit_watermark(lctx, prefilled);
        }
        // On 8 GiB GPUs: free prefiller VRAM under guard, then let branches fork before resuming prefill.
        if (state->is_snapshot_milestone(prefilled)) {
            {
                LlamaGuard guard;
                llama_free(lctx);
                lctx = nullptr;
            }
            state->wait_for_milestone_consumers(prefilled);
            {
                LlamaGuard guard;
                lctx = llama_init_from_model(ctx->model, ctx->ctx_params);
                if (!lctx) {
                    throw std::runtime_error("PrefillNode: llama_init_from_model failed after milestone resume.");
                }
                KVHandoff::bind_contiguous_cache(lctx, state->materialized_branch_buffer);
                restore_prefix_snapshot(lctx, state, prefilled);
            }
        }
    }
    // Mark prefill_complete so branches using kSwarmkvWaitForPrefillComplete can proceed.
    state->mark_prefill_complete();
    // Free the root prefill context now that export and watermark publication are done.
    llama_free(lctx);
    // Stop wall-clock timer and compute elapsed milliseconds for timings_ms map.
    const auto t1 = std::chrono::steady_clock::now();
    // Convert steady_clock delta to integer milliseconds for metrics friendliness.
    const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    // Store PrefillNode timing under orchestrator node name for benchmark scraping.
    {
        // Lock timings_ms because multiple nodes may write concurrently near pipeline end.
        std::lock_guard<std::mutex> lock(state->timing_lock);
        // Resolve map key from orchestrator context node name pointer.
        const char * key = ctx->node_name ? ctx->node_name : "Prefiller";
        // Record elapsed milliseconds for the root prefill node execution.
        state->timings_ms[std::string(key)] = ms;
    }
}
