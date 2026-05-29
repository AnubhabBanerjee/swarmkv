/**
 * @file codebase_audit_pipeline.cpp
 * @brief Example DAG: one prefill root feeding two analytical branches.
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

#include <iostream>
#include <llama.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/model.gguf>\n";
        return 1;
    }

    llama_backend_init();

    llama_model * model = llama_model_load_from_file(argv[1], swarmkv_default_model_params());
    if (!model) {
        std::cerr << "Failed to load model.\n";
        llama_backend_free();
        return 1;
    }

    static constexpr const char * kBaseDocPath = "examples/base_doc.txt";
    static constexpr const char * kBranchPromptSecurity = "Summarize security posture.";
    static constexpr const char * kBranchPromptLicense = "Note license obligations.";

    try {
        const std::string document = swarmkv_read_document_file(kBaseDocPath);
        const int32_t doc_tokens = static_cast<int32_t>(swarmkv_tokenize_for_budget(model, document).size());
        swarmkv_validate_context_budget(
            model,
            doc_tokens,
            {kBranchPromptSecurity, kBranchPromptLicense},
            kSwarmkvDefaultPipelineCtx);
    } catch (const std::exception & e) {
        std::cerr << "Context budget check failed: " << e.what() << '\n';
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    MemoryPool pool(model);
    Orchestrator orchestrator(&pool);

    orchestrator.add_node("Prefiller", std::make_unique<PrefillNode>(model, kBaseDocPath));
    // SecurityAuditor starts at watermark 512 (speculative branch) while prefill continues.
    orchestrator.add_node(
        "SecurityAuditor",
        std::make_unique<AnalyticalNode>(model, kBranchPromptSecurity, 512));
    // LicenseChecker waits for full document prefill (legacy completion semantics).
    orchestrator.add_node(
        "LicenseChecker",
        std::make_unique<AnalyticalNode>(model, kBranchPromptLicense, kSwarmkvWaitForPrefillComplete));
    orchestrator.add_dependency("Prefiller", "SecurityAuditor");
    orchestrator.add_dependency("Prefiller", "LicenseChecker");
    orchestrator.validate_acyclic();

    PipelineState state{};
    state.materialized_branch_buffer = pool.allocate_prefix_cache(kSwarmkvDefaultPipelineCtx);

    orchestrator.execute_pipeline(&state);

    llama_model_free(model);
    llama_backend_free();

    std::cout << "Security Audit: " << state.node_outputs["SecurityAuditor"] << "\n";
    std::cout << "License Check: " << state.node_outputs["LicenseChecker"] << "\n";
    return 0;
}
