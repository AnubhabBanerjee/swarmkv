#pragma once

#include "swarmkv/execution_node.h"
#include "swarmkv/prefix_constants.h"
#include <llama.h>
#include <string>

// AnalyticalNode decodes a branch prompt after forking prefix KV at a chosen watermark.
class AnalyticalNode : public ExecutionNode {
public:
    // Construct with prompt text and optional required_prefix_tokens watermark (V2).
    // Use kSwarmkvWaitForPrefillComplete to wait for full document prefill (legacy behavior).
    AnalyticalNode(
        llama_model * model,
        const std::string & prompt,
        int32_t required_prefix_tokens = kSwarmkvWaitForPrefillComplete);

    // execute materializes KV, restores seq state, and decodes the branch prompt under LlamaGuard.
    void execute(PipelineState * state, OrchestratorContext * ctx) override;

    // required_prefix_tokens exposes the watermark W this branch needs before it may start.
    int32_t required_prefix_tokens() const override;

private:
    // branch_prompt is the UTF-8 branch instruction tokenized at execute time.
    std::string branch_prompt;

    // required_prefix_tokens_ stores W for watermark gating (-1 means wait for full prefill).
    int32_t required_prefix_tokens_;
};
