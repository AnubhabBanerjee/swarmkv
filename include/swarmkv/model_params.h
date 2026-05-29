#pragma once

#include <llama.h>

/**
 * @brief Model load parameters tuned for SwarmKV demos, tests, and benchmarks.
 *
 * Offloads all transformer layers to the GPU when llama.cpp is built with a GPU
 * backend (e.g. GGML_CUDA=ON). If the binary is CPU-only, extra GPU layer settings
 * are ignored by llama.cpp at runtime.
 *
 * n_gpu_layers: use -1 for "all layers" (see llama.h). A large positive (e.g. 99)
 * is equivalent for 7B-class models.
 */
inline llama_model_params swarmkv_default_model_params() {
    llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = -1;
    return p;
}
