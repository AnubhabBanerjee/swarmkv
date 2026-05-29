#pragma once

#include "swarmkv/execution_node.h"
#include <llama.h>
#include <string>

// PrefillNode is the DAG root that ingests the shared document and publishes KV watermarks.
class PrefillNode : public ExecutionNode {
public:
    // Construct with a filesystem path to the shared document body.
    PrefillNode(llama_model * model, const std::string & source_path);

    // execute performs chunked prefill, commits watermarks, and marks prefill_complete at the end.
    void execute(PipelineState * state, OrchestratorContext * ctx) override;

    // is_prefill_provider identifies this node as the watermark publisher for the orchestrator.
    bool is_prefill_provider() const override { return true; }

private:
    // source_document_path points at examples/base_doc.txt or another shared document file.
    std::string source_document_path;
};
