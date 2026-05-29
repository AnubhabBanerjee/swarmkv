cmake_minimum_required(VERSION 3.20)
project(SwarmKV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Dependency Integration
# We assume llama.cpp is present as a submodule in third_party/
add_subdirectory(third_party/llama.cpp)

# 2. SwarmKV Library Definition
# We create the core library target that encapsulates our runtime logic.
add_library(swarmkv STATIC 
    src/core/orchestrator.cpp
    src/core/pipeline_state.cpp
    src/core/kv_handoff.cpp
    src/memory/memory_pool.cpp
    src/nodes/prefill_node.cpp
    src/nodes/analytical_node.cpp
)

# Export headers so targets can find them
target_include_directories(swarmkv PUBLIC include)

# Link against llama.cpp (which provides the ggml/llama interface)
target_link_libraries(swarmkv PUBLIC llama)

# 3. Main Executable (The entry point)
add_executable(swarmkv_demo main.cpp)
target_link_libraries(swarmkv_demo PRIVATE swarmkv llama)

# 4. Example Pipeline (Codebase Audit)
add_executable(codebase_audit examples/codebase_audit_pipeline.cpp)
target_link_libraries(codebase_audit PRIVATE swarmkv llama)

# 5. Testing Suite
# Enable CTest integration
enable_testing()

# Define test executables
add_executable(test_rope tests/test_rope_invariant.cpp)
target_link_libraries(test_rope PRIVATE swarmkv llama)
add_test(NAME RopeInvariantTest COMMAND $<TARGET_FILE:test_rope>)
set_tests_properties(RopeInvariantTest PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_memory tests/test_memory_lifecycle.cpp)
target_link_libraries(test_memory PRIVATE swarmkv llama)
add_test(NAME MemoryLifecycleTest COMMAND $<TARGET_FILE:test_memory>)
set_tests_properties(MemoryLifecycleTest PROPERTIES SKIP_RETURN_CODE 77)

# V2 gate: partial KV export/resume (see tests/spike_partial_kv.cpp)
add_executable(spike_partial_kv tests/spike_partial_kv.cpp)
target_link_libraries(spike_partial_kv PRIVATE llama)
target_include_directories(spike_partial_kv PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME SpikePartialKV COMMAND $<TARGET_FILE:spike_partial_kv>)
set_tests_properties(SpikePartialKV PROPERTIES SKIP_RETURN_CODE 77)

add_executable(swarmkv_benchmark benchmarks/swarmkv_benchmark.cpp)
target_link_libraries(swarmkv_benchmark PRIVATE swarmkv llama)

# Optional: Set CUDA support if llama.cpp was configured for it
if(GGML_CUDA)
    target_compile_definitions(swarmkv PUBLIC GGML_USE_CUDA)
endif()