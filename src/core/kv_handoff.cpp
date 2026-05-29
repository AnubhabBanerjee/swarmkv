/**
 * @file kv_handoff.cpp
 * @brief Copy-on-fork materialization and KV binding hooks for SwarmKV.
 *
 * This translation unit implements the bridge between MemoryPool-owned
 * ggml_backend_buffer handles and llama.cpp execution contexts. The public
 * surface is intentionally small so upstream API churn stays localized here.
 */

// Primary declaration of KVHandoff static methods and buffer-related types.
// We pull ggml backend definitions for buffer sizing and host pointer access.
#include "swarmkv/kv_handoff.h"

// llama.h is included for llama_context forward type used in bind signatures.
// Even when binding is a no-op, the signature must remain stable for nodes.
#include <llama.h>

// Standard headers for error reporting when callers pass invalid handles.
#include <cstring>
#include <stdexcept>

/**
 * @brief Copy the entire source prefix buffer into the start of the target buffer.
 *
 * SwarmKV uses this for copy-on-fork: the canonical prefix lives in `source`,
 * and each branch receives an independent `target` allocation large enough to
 * hold the prefix bytes plus future decode growth. The copy is byte-oriented.
 *
 * LIMITATION (documented): ggml_backend_buffer_get_base returns a pointer that
 * is only safe to memcpy on host-accessible backends (typically CPU). When
 * llama.cpp is built with CUDA, buffers may be device-resident; memcpy is then
 * undefined unless your deployment maps them host-visible. The demo pipeline
 * assumes CPU or mapped buffers; GPU-only deployments must swap this for a
 * backend-aware copy using ggml's scheduling APIs.
 */
void KVHandoff::materialize_branch_cache(ggml_backend_buffer_t source, ggml_backend_buffer_t target, size_t nbytes) {
    // Reject null handles early so we never pass invalid pointers to memcpy.
    // This keeps failures loud and close to the caller instead of deep inside memcpy.
    if (!source || !target) {
        throw std::invalid_argument("KVHandoff::materialize_branch_cache: null buffer handle.");
    }

    // Query the byte length of the source allocation from ggml.
    // The source is expected to be sized exactly for the prefilled prefix KV bytes.
    void * src_ptr = ggml_backend_buffer_get_base(source);
    void * dst_ptr = ggml_backend_buffer_get_base(target);
    const size_t src_cap = ggml_backend_buffer_get_size(source);
    const size_t dst_cap = ggml_backend_buffer_get_size(target);
    const size_t ncopy = nbytes == 0 ? src_cap : nbytes;

    if (ncopy > src_cap || ncopy > dst_cap) {
        throw std::runtime_error("KVHandoff::materialize_branch_cache: copy size exceeds buffer capacity.");
    }

    // Perform the byte-level clone that avoids recomputing attention for the prefix.
    // This is the mechanical heart of copy-on-fork at the storage layer.
    std::memcpy(dst_ptr, src_ptr, ncopy);
}

/**
 * @brief Intended hook: attach an external contiguous buffer as the context KV store.
 *
 * CURRENT UPSTREAM REALITY: ggml-org/llama.cpp exposes llama_memory_t and graph
 * decode paths, but there is no stable exported C symbol named llama_kv_cache_bind
 * in the public headers vendored with this repository. SwarmKV therefore treats
 * binding as a documented no-op while still requiring the call site to exist so
 * the DAG nodes and tests share one consistent control flow. Decode uses the
 * context's internally managed KV; MemoryPool buffers remain staging for the
 * materialize path and future engine hooks when a stable bind API lands.
 */
void KVHandoff::bind_contiguous_cache(llama_context * ctx, ggml_backend_buffer_t cache) {
    // Validate arguments so misuse fails fast during bring-up and CI smoke runs.
    // A null context cannot decode; a null cache handle is a configuration bug.
    if (!ctx || !cache) {
        throw std::invalid_argument("KVHandoff::bind_contiguous_cache: null context or buffer.");
    }

    // Explicitly mark both parameters as intentionally unused in this revision.
    // This prevents -Wunused-parameter warnings under strict warning flags.
    (void) ctx;
    (void) cache;

    // No stable bind call is issued here; see file-level comment above.
    // When upstream adds a supported attachment API, implement it only in this function.
}
