#pragma once

#include <ggml-backend.h>
#include <llama.h>

// KVHandoff provides static utilities to bridge memory between buffers.
// It enforces the copy-on-fork architecture by isolating memory transfer logic.
class KVHandoff {
public:
    // Performs a backend-aware copy of the source prefix buffer to the target branch buffer.
    // This is the implementation of the copy-on-fork mechanic. If nbytes is 0, copies the full source allocation.
    static void materialize_branch_cache(ggml_backend_buffer_t source, ggml_backend_buffer_t target, size_t nbytes = 0);

    // Binds a pre-allocated contiguous buffer to a specific llama_context.
    // This allows the engine to treat our shared memory as its native KV cache.
    static void bind_contiguous_cache(llama_context* ctx, ggml_backend_buffer_t cache);
};