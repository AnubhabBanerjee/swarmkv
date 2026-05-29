#pragma once

#include "swarmkv/pipeline_state.h"
#include "swarmkv/orchestrator_context.h"
#include "swarmkv/prefix_constants.h"

// ExecutionNode is the virtual base class implemented by PrefillNode and AnalyticalNode.
// The orchestrator calls execute() on worker threads after dependency or watermark gating.
class ExecutionNode {
public:
    // Virtual destructor ensures derived node destructors run when stored as unique_ptr.
    virtual ~ExecutionNode() = default;

    // execute runs the node-specific llama work using shared PipelineState and per-node context.
    virtual void execute(PipelineState * state, OrchestratorContext * ctx) = 0;

    // required_prefix_tokens returns the readiness watermark for speculative branching (V2).
    // kSwarmkvWaitForPrefillComplete means wait for the prefiller thread to finish entirely.
    virtual int32_t required_prefix_tokens() const { return kSwarmkvWaitForPrefillComplete; }

    // is_prefill_provider returns true only for PrefillNode so the orchestrator can gate on watermarks.
    virtual bool is_prefill_provider() const { return false; }
};
