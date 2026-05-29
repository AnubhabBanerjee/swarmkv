#pragma once

#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <ggml-backend.h>

struct llama_context;

// PrefixKvSnapshot stores an immutable KV export taken at a specific watermark milestone.
// Branches that declare required_prefix_tokens=W must fork from this blob, not a later one.
struct PrefixKvSnapshot {
    // data holds the serialized llama_state_seq_get_data bytes for the prefix sequence.
    std::vector<uint8_t> data;
    // seq_len records how many document tokens were prefilled when the snapshot was captured.
    int32_t seq_len = 0;
};

// PipelineState is the shared blackboard between PrefillNode and AnalyticalNode workers.
// V2 adds a readiness watermark so branches can start before the full document prefill ends.
struct PipelineState {

    // materialized_branch_buffer is the MemoryPool-owned staging area for the latest KV export.
    // PrefillNode writes the most recent llama_state_seq_get_data snapshot into this buffer.
    ggml_backend_buffer_t materialized_branch_buffer = nullptr;

    // prefix_kv_serial_bytes counts valid bytes in the latest canonical export (0 if none).
    size_t prefix_kv_serial_bytes = 0;

    // prefix_seq_len is the RoPE offset for the latest canonical export (tokens prefilled so far).
    int32_t prefix_seq_len = 0;

    // watermark is the highest contiguous prefix token count published by PrefillNode.
    // Branches block on wait_for_watermark(W) until watermark >= W.
    int32_t watermark = 0;

    // prefill_complete becomes true after PrefillNode finishes the entire document.
    bool prefill_complete = false;

    // node_outputs stores per-branch textual results keyed by orchestrator node name.
    std::unordered_map<std::string, std::string> node_outputs;

    // output_lock guards concurrent writes into node_outputs from parallel branch threads.
    std::mutex output_lock;

    // timings_ms records wall-clock segments per node name for benchmarks and diagnostics.
    std::unordered_map<std::string, int64_t> timings_ms;

    // timing_lock protects timings_ms from races when nodes report concurrently.
    std::mutex timing_lock;

    // register_snapshot_milestone marks watermark values that need immutable KV snapshots.
    // The orchestrator registers each AnalyticalNode::required_prefix_tokens() before execute.
    void register_snapshot_milestone(int32_t token_count);

    // commit_watermark exports prefix-seq KV after a prefill chunk and notifies waiting branches.
    void commit_watermark(struct llama_context * ctx, int32_t tokens_ready);

    // wait_for_watermark blocks until watermark >= required_tokens (used by orchestrator).
    void wait_for_watermark(int32_t required_tokens) const;

    // mark_prefill_complete sets the completion flag and wakes any full-document waiters.
    void mark_prefill_complete();

    // wait_for_prefill_complete blocks until the prefiller thread has finished the document.
    void wait_for_prefill_complete() const;

    // try_get_prefix_snapshot returns immutable bytes captured exactly at watermark W.
    // Returns false if no milestone snapshot exists for W (caller should treat as error).
    bool try_get_prefix_snapshot(
        int32_t watermark_w,
        const uint8_t ** out_data,
        size_t * out_bytes,
        int32_t * out_seq_len) const;

    // register_milestone_consumer records one branch that will fork at watermark W (orchestrator).
    void register_milestone_consumer(int32_t token_count);

    // signal_milestone_consumed is called after a branch finishes forking at watermark W.
    void signal_milestone_consumed(int32_t token_count);

    // wait_for_milestone_consumers blocks the prefiller until all branches at W have forked (VRAM gate).
    void wait_for_milestone_consumers(int32_t token_count);

    // is_snapshot_milestone returns true when W was registered for immutable KV export.
    bool is_snapshot_milestone(int32_t token_count) const;

private:
    // watermark_mutex serializes snapshot export and coordinates with condition_variable waits.
    mutable std::mutex watermark_mutex;

    // watermark_cv wakes branch threads when commit_watermark raises the readiness watermark.
    mutable std::condition_variable watermark_cv;

    // snapshot_milestones lists watermark values that require a frozen PrefixKvSnapshot entry.
    std::unordered_set<int32_t> snapshot_milestones;

    // prefix_snapshots maps watermark W -> immutable KV bytes valid for branching at position W.
    std::unordered_map<int32_t, PrefixKvSnapshot> prefix_snapshots;

    // milestone_consumers_remaining counts branches not yet done forking at each watermark W.
    std::unordered_map<int32_t, int32_t> milestone_consumers_remaining;

    // prefill_resume_cv wakes the prefiller after the last branch at a milestone releases VRAM.
    std::condition_variable prefill_resume_cv;
};
