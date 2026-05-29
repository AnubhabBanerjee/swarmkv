/**
 * @file test_rope_invariant.cpp
 * @brief Smoke test: materialize_branch_cache copies bytes between pool allocations.
 */

#include "swarmkv/kv_handoff.h"
#include "swarmkv/memory_pool.h"
#include "swarmkv/model_params.h"

#include <cstring>
#include <iostream>
#include <llama.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "SKIP: pass model.gguf as argv[1] for RopeInvariantTest\n";
        return 77;
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], swarmkv_default_model_params());
    if (!model) {
        llama_backend_free();
        return 1;
    }

    MemoryPool pool(model);
    ggml_backend_buffer_t a = pool.allocate_prefix_cache(512);
    ggml_backend_buffer_t b = pool.allocate_branch_cache(512);
    void * pa = ggml_backend_buffer_get_base(a);
    std::memset(pa, 0xAB, ggml_backend_buffer_get_size(a));
    KVHandoff::materialize_branch_cache(a, b);
    void * pb = ggml_backend_buffer_get_base(b);
    if (std::memcmp(pa, pb, ggml_backend_buffer_get_size(a)) != 0) {
        pool.free_all();
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    pool.free_all();
    llama_model_free(model);
    llama_backend_free();
    std::cout << "SUCCESS: materialize_branch_cache byte identity verified.\n";
    return 0;
}
