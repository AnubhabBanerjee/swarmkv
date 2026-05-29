/**
 * @file pipeline_state.cpp
 * @brief V2 readiness watermark: commit, wait, and immutable per-milestone KV snapshots.
 */

// prefix_constants.h defines kSwarmkvPrefixSeqId used during llama_state_seq_get_data export.
#include "swarmkv/prefix_constants.h"

// pipeline_state.h declares PipelineState fields and synchronization entrypoints.
#include "swarmkv/pipeline_state.h"

// llama.h provides llama_state_seq_get_size/data used inside commit_watermark.
#include <llama.h>

// stdexcept surfaces allocation and export failures to the orchestrator try/catch path.
#include <stdexcept>

// cstring provides std::memcpy when freezing milestone snapshots into std::vector storage.
#include <cstring>

// register_snapshot_milestone records a watermark value that must receive a frozen KV snapshot.
// The orchestrator calls this once per analytical branch before threads start running.
void PipelineState::register_snapshot_milestone(int32_t token_count) {
    // Reject non-positive milestones because watermark zero means no document tokens yet.
    if (token_count <= 0) {
        // Throw so misconfigured graphs fail fast during pipeline setup instead of at runtime.
        throw std::invalid_argument("PipelineState::register_snapshot_milestone: token_count must be > 0.");
    }
    // Acquire watermark_mutex so milestone registration does not race commit_watermark.
    std::lock_guard<std::mutex> lock(watermark_mutex);
    // Insert the milestone into the set; duplicates are harmless for std::unordered_set.
    snapshot_milestones.insert(token_count);
}

// commit_watermark publishes tokens_ready, exports prefix-seq KV, and optionally freezes a snapshot.
void PipelineState::commit_watermark(llama_context * ctx, int32_t tokens_ready) {
    // A null llama context cannot export KV; treat this as a hard programming error.
    if (!ctx) {
        throw std::invalid_argument("PipelineState::commit_watermark: null llama_context.");
    }
    if (tokens_ready < 0) {
        throw std::invalid_argument("PipelineState::commit_watermark: tokens_ready must be >= 0.");
    }
    bool export_kv = false;
    {
        std::lock_guard<std::mutex> lock(watermark_mutex);
        export_kv = snapshot_milestones.count(tokens_ready) != 0;
        if (!export_kv) {
            watermark = tokens_ready;
            watermark_cv.notify_all();
            return;
        }
    }
    size_t nw = 0;
    std::vector<uint8_t> export_buf;
    if (materialized_branch_buffer) {
        const size_t need = llama_state_seq_get_size(ctx, kSwarmkvPrefixSeqId);
        const size_t cap = ggml_backend_buffer_get_size(materialized_branch_buffer);
        if (cap < need) {
            throw std::runtime_error(
                "PipelineState::commit_watermark: canonical buffer smaller than llama_state_seq_get_size.");
        }
        void * base = ggml_backend_buffer_get_base(materialized_branch_buffer);
        nw = llama_state_seq_get_data(
            ctx,
            static_cast<uint8_t *>(base),
            cap,
            kSwarmkvPrefixSeqId);
        export_buf.resize(nw);
        if (nw > 0) {
            std::memcpy(export_buf.data(), base, nw);
        }
    }
    {
        std::lock_guard<std::mutex> lock(watermark_mutex);
        watermark = tokens_ready;
        prefix_seq_len = tokens_ready;
        prefix_kv_serial_bytes = nw;
        if (export_kv && nw > 0) {
            PrefixKvSnapshot & snap = prefix_snapshots[tokens_ready];
            snap.data = std::move(export_buf);
            snap.seq_len = tokens_ready;
        } else if (export_kv) {
            PrefixKvSnapshot & snap = prefix_snapshots[tokens_ready];
            snap.data.clear();
            snap.seq_len = tokens_ready;
        }
        watermark_cv.notify_all();
    }
}

// wait_for_watermark blocks the calling thread until watermark >= required_tokens.
void PipelineState::wait_for_watermark(int32_t required_tokens) const {
    // Negative requirements are invalid for watermark gating; use wait_for_prefill_complete instead.
    if (required_tokens < 0) {
        // Throw so orchestrator wiring mistakes are caught before decode starts.
        throw std::invalid_argument("PipelineState::wait_for_watermark: required_tokens must be >= 0.");
    }
    // Acquire a unique_lock because condition_variable::wait must unlock/relock the mutex.
    std::unique_lock<std::mutex> lock(watermark_mutex);
    // Sleep until the prefiller publishes a watermark that satisfies this branch requirement.
    watermark_cv.wait(lock, [this, required_tokens]() { return watermark >= required_tokens; });
}

// mark_prefill_complete signals that the document prefill thread has finished all chunks.
void PipelineState::mark_prefill_complete() {
    // Acquire watermark_mutex so completion visibility is ordered with snapshot publication.
    std::lock_guard<std::mutex> lock(watermark_mutex);
    // Set the completion flag read by wait_for_prefill_complete and orchestrator logic.
    prefill_complete = true;
    // Wake any threads waiting for full-document completion rather than a partial watermark.
    watermark_cv.notify_all();
}

// wait_for_prefill_complete blocks until PrefillNode calls mark_prefill_complete.
void PipelineState::wait_for_prefill_complete() const {
    // Acquire unique_lock to use condition_variable with the watermark mutex.
    std::unique_lock<std::mutex> lock(watermark_mutex);
    // Sleep until prefill_complete becomes true at the end of the root prefill node.
    watermark_cv.wait(lock, [this]() { return prefill_complete; });
}

// try_get_prefix_snapshot returns immutable bytes captured at watermark W if present.
bool PipelineState::try_get_prefix_snapshot(
    int32_t watermark_w,
    const uint8_t ** out_data,
    size_t * out_bytes,
    int32_t * out_seq_len) const {
    // Validate output pointers so callers cannot pass null and crash on dereference.
    if (!out_data || !out_bytes || !out_seq_len) {
        // Return false to indicate the snapshot could not be returned.
        return false;
    }
    // Acquire watermark_mutex so map lookup is not concurrent with commit_watermark writes.
    std::lock_guard<std::mutex> lock(watermark_mutex);
    // Find the frozen snapshot entry for the requested milestone watermark.
    const auto it = prefix_snapshots.find(watermark_w);
    // If no snapshot exists, the branch woke too late or the milestone was not registered.
    if (it == prefix_snapshots.end()) {
        // Return false so AnalyticalNode can throw a descriptive error upstream.
        return false;
    }
    // Point the caller at the immutable vector storage owned by PipelineState.
    *out_data = it->second.data.empty() ? nullptr : it->second.data.data();
    // Report how many bytes are valid inside the frozen snapshot vector.
    *out_bytes = it->second.data.size();
    // Report the RoPE prefix length that must be used when decoding branch tokens.
    *out_seq_len = it->second.seq_len;
    // Return true to indicate the snapshot metadata was populated successfully.
    return true;
}

void PipelineState::register_milestone_consumer(int32_t token_count) {
    if (token_count <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(watermark_mutex);
    milestone_consumers_remaining[token_count]++;
}

void PipelineState::signal_milestone_consumed(int32_t token_count) {
    if (token_count <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(watermark_mutex);
    auto it = milestone_consumers_remaining.find(token_count);
    if (it == milestone_consumers_remaining.end()) {
        return;
    }
    it->second--;
    if (it->second <= 0) {
        prefill_resume_cv.notify_all();
    }
}

bool PipelineState::is_snapshot_milestone(int32_t token_count) const {
    std::lock_guard<std::mutex> lock(watermark_mutex);
    return snapshot_milestones.count(token_count) != 0;
}

void PipelineState::wait_for_milestone_consumers(int32_t token_count) {
    if (token_count <= 0) {
        return;
    }
    std::unique_lock<std::mutex> lock(watermark_mutex);
    prefill_resume_cv.wait(lock, [this, token_count]() {
        const auto it = milestone_consumers_remaining.find(token_count);
        if (it == milestone_consumers_remaining.end()) {
            return true;
        }
        return it->second <= 0;
    });
}
