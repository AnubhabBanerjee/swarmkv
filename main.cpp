/**
 * @file main.cpp
 * @brief Minimal SwarmKV demo executable (`swarmkv_demo` CMake target).
 *
 * This translation unit exists to prove the static library links against llama.cpp
 * and that the orchestrator can schedule a tiny DAG without involving Python or HTTP.
 * It intentionally keeps the graph small so classroom and CI machines can iterate quickly.
 */

// MemoryPool owns ggml host buffers that are sized using llama_state_get_size snapshots.
#include "swarmkv/memory_pool.h"

// AnalyticalNode implements ExecutionNode for short branch prompts after optional prefill.
#include "swarmkv/nodes/analytical_node.h"

// Orchestrator owns ExecutionNode instances and executes the DAG with futures/promises.
#include "swarmkv/orchestrator.h"

// PipelineState is the shared blackboard passed into every node execute() invocation.
#include "swarmkv/pipeline_state.h"
#include "swarmkv/model_params.h"
#include "swarmkv/defaults.h"

// iostream is used for argv usage text and the final success line on stdout.
#include <iostream>

// llama.h declares llama_backend_init, model load/free, and the llama_model opaque type.
#include <llama.h>

/**
 * @brief Program entry: wire llama, pool, orchestrator, shared state, then run the DAG.
 *
 * argv[1] must point to a GGUF on disk because we do not embed weights in the repo.
 * The function returns a non-zero exit status early when arguments are incomplete.
 */
int main(int argc, char ** argv) {
    // Guard clause: without a model path we cannot construct llama_model safely.
    // Printing usage to stderr helps first-time users who double-click the binary.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/model.gguf>\n";
        return 1;
    }

    // Initialize global ggml/llama runtime state once per process lifetime.
    // Missing this call can crash later during backend tensor allocation paths.
    llama_backend_init();

    // Load weights from disk; failures return nullptr and we abort before touching the pool.
    llama_model * model = llama_model_load_from_file(argv[1], swarmkv_default_model_params());
    if (!model) {
        std::cerr << "Failed to load model from " << argv[1] << "\n";
        llama_backend_free();
        return 1;
    }

    // Construct the arena allocator that tracks every ggml_backend_buffer we hand out.
    MemoryPool pool(model);

    // Construct the orchestrator with a non-owning pointer to the pool for teardown ordering.
    Orchestrator orchestrator(&pool);

    // Register two analytical nodes with different string prompts to simulate branches.
    orchestrator.add_node("Extractor", std::make_unique<AnalyticalNode>(model, "Summarize: hello."));
    orchestrator.add_node("Auditor", std::make_unique<AnalyticalNode>(model, "List keywords in: hello."));

    // Enforce ordering: Auditor waits until Extractor fulfills its completion promise.
    orchestrator.add_dependency("Extractor", "Auditor");

    // Statically reject cycles before spawning any std::async work.
    orchestrator.validate_acyclic();

    // Zero-initialize PipelineState so prefix_seq_len starts at zero before any prefill node runs.
    PipelineState state{};
    state.materialized_branch_buffer = pool.allocate_prefix_cache(kSwarmkvDefaultPipelineCtx);

    // Run the full DAG; this blocks until every worker thread finishes and pool free_all runs.
    orchestrator.execute_pipeline(&state);

    // Free model weights before tearing down the global llama backend to keep shutdown ordered.
    llama_model_free(model);

    // Release process-wide llama resources after all contexts and models are gone.
    llama_backend_free();

    // Emit a friendly line so shell scripts can grep for success in automation harnesses.
    std::cout << "Pipeline finished successfully.\n";
    return 0;
}
