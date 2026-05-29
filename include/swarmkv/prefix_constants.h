#pragma once

#include <cstdint>
#include <llama.h>

// kSwarmkvPrefillChunkTokens controls how many document tokens PrefillNode decodes per slice.
// Smaller chunks publish the readiness watermark more often for speculative branches.
// The value 128 is a practical default that balances llama_n_batch limits and notify frequency.
inline constexpr int32_t kSwarmkvPrefillChunkTokens = 128;

// kSwarmkvPrefixSeqId is the sole llama sequence lane used for the shared document prefix.
// llama_state_seq_get_data must snapshot only this id so branch forks never mix other seq slots.
inline constexpr llama_seq_id kSwarmkvPrefixSeqId = 0;

// kSwarmkvWaitForPrefillComplete is the sentinel for AnalyticalNode::required_prefix_tokens().
// Negative values mean the branch waits for the prefiller thread to finish entirely (V1 behavior).
inline constexpr int32_t kSwarmkvWaitForPrefillComplete = -1;
