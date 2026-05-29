#pragma once

#include <llama.h>

class MemoryPool;

// OrchestratorContext wraps the engine-level resources required by worker nodes.
// It is passed to nodes during execution to provide model access and configuration.
struct OrchestratorContext {
    // The loaded model pointer. Nodes need this to construct their temporary contexts.
    llama_model* model;

    // Context parameters (n_ctx, n_batch) shared across the pipeline.
    // This ensures consistency in token capacity across different analytical branches.
    llama_context_params ctx_params;

    // Arena used to allocate per-branch KV clones (copy-on-fork isolation).
    MemoryPool* memory_pool;

    // The name of the node currently executing.
    // Useful for logging and error reporting within the worker threads.
    const char* node_name;
};