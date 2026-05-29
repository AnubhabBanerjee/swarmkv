/**
 * @file memory_pool.cpp
 * @brief MemoryPool arena: tracks ggml_backend_buffer allocations for KV staging.
 */

// Class interface for MemoryPool and llama/ggml includes for engine queries.
#include "swarmkv/memory_pool.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <iostream>
#include <llama.h>

/**
 * @brief Construct the pool and retain the model pointer used for sizing queries.
 *
 * The model must outlive the pool in normal SwarmKV usage because allocation
 * routines create temporary llama_context objects to ask the engine how large
 * serialized state would be for a given n_ctx budget.
 */
MemoryPool::MemoryPool(llama_model * model) : model_ref(model) {
    // Reject null models because every allocation path dereferences model_ref.
    // Failing here avoids mysterious nullptr crashes inside llama_init_from_model.
    if (!model_ref) {
        throw std::invalid_argument("MemoryPool: null llama_model pointer.");
    }
}

/**
 * @brief Destructor: always attempt to return tracked buffers to the allocator.
 *
 * We swallow exceptions because throwing from destructors is undefined behavior
 * in many C++ patterns; orchestrator should still call free_all explicitly.
 */
MemoryPool::~MemoryPool() {
    try {
        free_all();
    } catch (...) {
        std::cerr << "MemoryPool: exception during destructor free_all (ignored).\n";
    }
}

/**
 * @brief Return the owning model pointer for OrchestratorContext wiring.
 */
llama_model * MemoryPool::get_model() const {
    return model_ref;
}

/**
 * @brief Query the engine for a conservative byte size for a context of n_ctx tokens.
 *
 * We spin up a temporary context with the requested n_ctx and read
 * llama_state_get_size, which includes KV/logits/session storage as reported by
 * the pinned llama.cpp revision. This avoids hand-derived KV tensor math that
 * drifts across quantization formats.
 */
size_t MemoryPool::get_required_kv_size(uint32_t n_ctx) {
    // Start from library defaults so fields we do not care about remain sane.
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;

    // Construct a disposable context solely to query serialized state footprint.
    llama_context * ctx = llama_init_from_model(model_ref, params);
    if (!ctx) {
        throw std::runtime_error("MemoryPool::get_required_kv_size: llama_init_from_model failed.");
    }

    // Ask llama.cpp how many bytes a full state blob would occupy for this ctx.
    const size_t sz = llama_state_get_size(ctx);
    llama_free(ctx);

    // If the engine reports zero, fall back to a small non-zero allocation so tests
    // still exercise the registry without pretending we know exact tensor layouts.
    if (sz == 0) {
        return size_t{1} << 20;
    }
    return sz;
}

/**
 * @brief Allocate and register a buffer sized for a prefix-length n_ctx.
 */
ggml_backend_buffer_t MemoryPool::allocate_prefix_cache(uint32_t n_ctx) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    const size_t kv_size = get_required_kv_size(n_ctx);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(cpu_dev);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, kv_size);
    if (!buffer) {
        throw std::runtime_error("MemoryPool::allocate_prefix_cache: allocation failed.");
    }
    active_buffers.push_back(buffer);
    return buffer;
}

/**
 * @brief Allocate a branch buffer for n_ctx tokens (prefix clone + headroom policy).
 *
 * V1 uses the same sizing helper as the prefix path; callers pass a larger n_ctx
 * when they want additional decode slots reserved in the same ggml buffer.
 */
ggml_backend_buffer_t MemoryPool::allocate_branch_cache(uint32_t n_ctx) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    const size_t kv_size = get_required_kv_size(n_ctx);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(cpu_dev);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, kv_size);
    if (!buffer) {
        throw std::runtime_error("MemoryPool::allocate_branch_cache: allocation failed.");
    }
    active_buffers.push_back(buffer);
    return buffer;
}

/**
 * @brief Free every tracked buffer; safe to call repeatedly after clearing vector.
 */
void MemoryPool::free_all() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    for (ggml_backend_buffer_t buffer : active_buffers) {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
    }
    active_buffers.clear();
}
