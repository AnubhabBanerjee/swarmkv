/**
 * @file test_memory_lifecycle.cpp
 * @brief Smoke test: MemoryPool host buffer survives a throwaway llama context lifetime.
 */

#include "swarmkv/memory_pool.h"
#include "swarmkv/model_params.h"

#include <cstring>
#include <iostream>
#include <llama.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "SKIP: pass model.gguf as argv[1] for MemoryLifecycleTest\n";
        return 77;
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], swarmkv_default_model_params());
    if (!model) {
        std::cerr << "Failed to load model\n";
        llama_backend_free();
        return 1;
    }

    MemoryPool pool(model);
    ggml_backend_buffer_t buf = pool.allocate_prefix_cache(512);
    void * p = ggml_backend_buffer_get_base(buf);
    std::memset(p, 0xDE, ggml_backend_buffer_get_size(buf));

    llama_context_params pparams = llama_context_default_params();
    pparams.n_ctx = 512;
    llama_context * ctx = llama_init_from_model(model, pparams);
    if (!ctx) {
        pool.free_all();
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    llama_free(ctx);

    if (static_cast<unsigned char *>(p)[0] != 0xDE) {
        std::cerr << "Buffer pattern corrupted\n";
        pool.free_all();
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    pool.free_all();
    llama_model_free(model);
    llama_backend_free();
    std::cout << "SUCCESS: host buffer pattern intact after llama context teardown.\n";
    return 0;
}
