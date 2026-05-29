#pragma once

#include <mutex>

namespace swarmkv {

// llama.cpp CUDA paths are not safe for concurrent decode from multiple threads
// on one GPU without external serialization. All node execute() bodies must
// hold this mutex around llama_init / llama_decode / llama_free / state I/O.
inline std::mutex & llama_api_mutex() {
    static std::mutex m;
    return m;
}

} // namespace swarmkv

struct LlamaGuard {
    std::lock_guard<std::mutex> lock;
    LlamaGuard() : lock(swarmkv::llama_api_mutex()) {}
};
