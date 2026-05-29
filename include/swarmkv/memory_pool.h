#pragma once

#include <llama.h>
#include <ggml-backend.h>
#include <vector>
#include <mutex>

/**
 * @brief Manages physical VRAM buffers for the SwarmKV pipeline.
 * * The MemoryPool acts as an Arena Allocator, ensuring that KV-cache tensors
 * are allocated in isolated memory blocks to prevent aliasing or cross-talk 
 * between concurrent analytical branches.
 */
class MemoryPool {
public:
    /**
     * @brief Initializes the pool with a reference to the loaded model.
     * @param model Pointer to the loaded llama_model for engine-native queries.
     */
    explicit MemoryPool(llama_model* model);

    /**
     * @brief Destructor ensures complete teardown of all VRAM allocations.
     * This is the authorized point for freeing all tracked backend buffers.
     */
    ~MemoryPool();

    /**
     * @brief Allocates a dedicated buffer sized for a precomputed KV prefix.
     * @param n_ctx The context length for the prefix.
     * @return ggml_backend_buffer_t Handle to the allocated prefix buffer.
     */
    ggml_backend_buffer_t allocate_prefix_cache(uint32_t n_ctx);

    /**
     * @brief Allocates a dedicated buffer sized for a branch's inference delta.
     * @param n_ctx The context length for the specific branch.
     * @return ggml_backend_buffer_t Handle to the allocated branch buffer.
     */
    ggml_backend_buffer_t allocate_branch_cache(uint32_t n_ctx);

    /**
     * @brief Retrieves the pointer to the underlying llama_model.
     * @return llama_model* The pointer to the loaded model.
     */
    llama_model* get_model() const;

    /**
     * @brief Wipes all registered buffers from VRAM.
     * Must be called by the orchestrator upon DAG completion.
     */
    void free_all();

private:
    /**
     * @brief Queries the engine for exact byte footprints for KV tensors.
     */
    size_t get_required_kv_size(uint32_t n_ctx);

    /// Pointer to the loaded model, used for engine-level allocation queries.
    llama_model* model_ref;

    /// Registry of all active buffers to prevent memory leaks.
    std::vector<ggml_backend_buffer_t> active_buffers;

    /// Mutex to protect buffer allocation/deallocation from concurrent thread access.
    std::mutex pool_mutex;
};