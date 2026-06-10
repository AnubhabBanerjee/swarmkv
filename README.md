**Copyright (c) 2026 Anubhab Banerjee (AnubhabBanerjee/swarmkv)**  
**All rights reserved. No part of this repository may be used, redistributed, or modified in any form or by any means without the prior written permission of the author.**

---
<img width="1536" height="1024" alt="swarmkv-overview" src="https://github.com/user-attachments/assets/ab4c08f0-6fa5-42f8-95cd-08a704a3093a" />

---

# 🚀 SwarmKV: Shared-Prefix Runtime for Long-Context Analytical Pipelines

> Single-process **C++20** orchestration over **llama.cpp** with DAG-ordered stages, **copy-on-fork** host KV buffers, and **RoPE-aware** branch offsets—without pretending an unsupported upstream KV “bind” API exists. **SwarmKV** is a systems-style inference **runtime skeleton** for *sequential* analytical graphs: **prefill once**, then run branch decodes whose **batch positions** continue after the shared prefix. It is **not** a Python agent framework and **not** a multi-tenant serving stack; it is deliberately scoped so you can **build, run, and reason about** pipeline-level KV reuse on one machine.

**This repository is Part 1** of the *Production-Grade Agentic Inference* series (*Towards Data Science*). Please see below for other parts in this series.

## 🔗 Series links

- **Part 1 — this repository** (Full deep-dive: `https://towardsdatascience.com/kv-cache-reuse-for-multi-agent-llm-inference-i-built-a-c-orchestrator-so-my-gpu-would-stop-reading-the-same-document-twice/`)
- **Part 2 — Kube-TimeSlice-Profiler:** (Repository link: `https://github.com/AnubhabBanerjee/Kube-Timeslice-Profiler`)
- **Part 3 — CUDA-TopK-Retrieval** (Repository link: `https://github.com/AnubhabBanerjee/CUDA-TopK-Retrieval`)
- **Part 4 — ILCP (future):** persist agent state across hand-offs (not implemented yet)

## 🧠 System Architecture

SwarmKV runs as a **single-process** runtime with a **join barrier** before VRAM/host buffer teardown:

1. **`MemoryPool`** — Owns **`ggml_backend_buffer_t`** allocations on the **CPU host** backend. Sizes come from **`llama_state_get_size`** on a disposable `llama_context`, with a small fallback if the engine reports `0`.
2. **`Orchestrator`** — Registers **`ExecutionNode`** instances, keeps **forward + reverse** adjacency for dependencies, runs **DFS cycle detection**, spawns **`std::async(..., std::launch::async)`** workers, waits on **all futures**, then calls **`MemoryPool::free_all()`**.
3. **`PrefillNode` / `AnalyticalNode`** — Construct **`llama_batch`** manually, call **`llama_decode`** in **chunks** bounded by **`llama_n_batch`** (required for multi-k token documents). **`PrefillNode`** loads text from a filesystem path (e.g. **`examples/base_doc.txt`**), sets **`PipelineState::prefix_seq_len`**, and exports KV via **`llama_state_get_data`**. Each **`AnalyticalNode`** **`memcpy`**’s the prefix snapshot into a per-branch buffer, **`llama_state_set_data`**, then decodes with **RoPE positions** starting at **`prefix_seq_len`**.
4. **`KVHandoff`** — **`materialize_branch_cache`** performs a **byte `memcpy`** between host buffer bases (the copy-on-fork hook). **`bind_contiguous_cache`** is currently a **documented no-op**: the pinned **`llama.h`** surface used here does **not** expose a stable **`llama_kv_cache_bind`**-style attachment API, so decode uses the **context’s internal KV** while the call site stays stable for future engine hooks.
5. **`context_budget.h`** — Fail-fast validation that **`doc_tokens + branch_tokens + headroom ≤ n_ctx`** before **`llama_decode`** (used by **`codebase_audit`**, nodes, and the benchmark harness).
6. **`llama_guard.h`** — Global mutex serializing all **`llama_*`** calls from worker threads (llama.cpp is not safe for concurrent GPU decode on one device without external locking).
7. **`swarmkv_benchmark`** — Baseline vs SwarmKV wall-clock harness with structured **`Key: Value`** metrics on stdout; see **`scripts/benchmark_campaign.py.example`** for a full 4K campaign (plots + **`final_result.docx`**).

The orchestration graph is a **DAG** (dependencies only); there is **no** feedback edge from decode back into the scheduler in V1.

## 🛠️ Build & Engine Layout

Designed for a **local developer loop** first; GPU offload inside `llama.cpp` remains optional and orthogonal to SwarmKV’s **host-buffer** copy path:

* **Host (default path)** — `MemoryPool` allocates via **`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)`** so **`memcpy`** between buffers is well-defined for the smoke build.
* **GPU (optional)** — You can configure **`llama.cpp`** with CUDA/Metal backends for **faster `llama_decode`**, but you must treat **`materialize_branch_cache`** as **device-aware** in that deployment (replace `memcpy` with a backend copy) or keep weights/KV on host-mapped memory.
* **Third-party tree** — **`third_party/llama.cpp`** is expected as a **local checkout** (clone or submodule). This README does not vendor upstream sources beyond what you place in that directory.
* **~8 GiB VRAM class GPUs** — Default pipeline context is **`kSwarmkvDefaultPipelineCtx` = 4096** (`include/swarmkv/defaults.h`) so **Qwen2.5‑7B Q4_K_M** (~4.4 GiB weights) plus **two concurrent** analytical `llama_context`s after prefill stay under typical **8 GiB** budgets. Raise the constant on 12 GiB+ cards if you need longer shared prefixes.

## ✅ Prerequisites

* **CMake 3.20+** and a **C++20** toolchain (GCC 13+ tested).
* **Git** — to fetch **`third_party/llama.cpp`** consistently across machines.
* A local **GGUF** model path passed on the command line to **`swarmkv_demo`**, **`codebase_audit`**, **`swarmkv_benchmark`**, and the smoke tests when you want them to **run** (not skip).
* **Optional:** `nvcc` / GPU stack only if you configure **`llama.cpp`** for CUDA; SwarmKV itself is CMake **`LANGUAGES CXX`** only.

## ⚙️ Installation

```bash
git clone <github-repo-url> swarm-kv
cd swarm-kv

# Populate llama.cpp (submodule OR manual clone into third_party/llama.cpp)
git submodule update --init --recursive

# CPU-only (default). For interview-grade TTFT/VRAM numbers, use NVIDIA CUDA:
# cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake -S . -B build -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j
```

All **`llama_model_load_from_file`** call sites use **`swarmkv_default_model_params()`** (`include/swarmkv/model_params.h`), which sets **`n_gpu_layers = -1`** (offload **all** layers when the `llama` target is built with CUDA). Reconfigure with **`-DGGML_CUDA=ON`** or layers stay on CPU.

## 📂 Upstream Engine Setup (BYO `llama.cpp`)

This repository does **not** assume a particular **`llama.cpp`** commit is redistributed inside SwarmKV beyond your own checkout policy.

1. Ensure **`third_party/llama.cpp`** exists and CMake **`add_subdirectory(third_party/llama.cpp)`** succeeds.
2. Record the **exact commit hash** you validated in **`projectdocs/project_planning.md`** (and your paper/demo appendix) so reviewers can reproduce allocator and **`llama_batch`** semantics.

## 🚀 Execution

Run the **minimal demo** (two analytical nodes with a simple dependency edge):

```bash
./build/swarmkv_demo /path/to/model.gguf
```

Run the **codebase audit example** (prefill root feeding two branches):

```bash
./build/codebase_audit /path/to/model.gguf
```

Run **CTest** smoke targets (they **skip with exit code `77`** when no model argv is wired through `ctest`; invoke binaries directly for full runs):

```bash
cd build
ctest --output-on-failure
./test_rope /path/to/model.gguf
./test_memory /path/to/model.gguf
```

Run the **benchmark harness** directly (metrics on stdout; **`llama_perf_context_print`** on stderr):

```bash
export LD_LIBRARY_PATH=build/bin:$LD_LIBRARY_PATH
export GGML_LOG_LEVEL=ERROR
./build/swarmkv_benchmark /path/to/model.gguf 3500 examples/base_doc.txt
```

Run the **full benchmark campaign** (3 trials, CSV/JSON, plots, narrative **`final_result.docx`**):

```bash
cp scripts/benchmark_campaign.py.example scripts/benchmark_campaign.py
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install matplotlib python-docx
scripts/.venv/bin/python scripts/benchmark_campaign.py
```

Artifacts land under **`examples/example-run-results/`**. See **`examples/example-run-results/REPRODUCE.md`** and **`scripts/README.md`**.

## 🎬 Example Run

```text
$ ./build/swarmkv_demo ./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
Pipeline finished successfully.

$ cd build && ctest --output-on-failure
100% tests passed, 0 tests failed out of 2
The following tests did not run:
  1 - RopeInvariantTest (Skipped)
  2 - MemoryLifecycleTest (Skipped)

$ ./build/test_rope ../models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
SUCCESS: materialize_branch_cache byte identity verified.
```

## 📁 Project Layout

```text
swarm-kv/
├── CMakeLists.txt                 # swarmkv static lib + demos + benchmark + CTest
├── README.md
├── main.cpp                       # swarmkv_demo entry point
├── benchmarks/
│   └── swarmkv_benchmark.cpp      # baseline vs DAG timing harness
├── examples/
│   ├── base_doc.txt                 # shared document written by benchmark (≈3.5K tokens)
│   ├── codebase_audit_pipeline.cpp
│   └── example-run-results/         # benchmark CSV/JSON/plots/final_result.docx (see REPRODUCE.md)
├── include/swarmkv/
│   ├── context_budget.h             # n_ctx fail-fast validation
│   ├── execution_node.h
│   ├── kv_handoff.h
│   ├── llama_guard.h                # serialize llama API across worker threads
│   ├── memory_pool.h
│   ├── orchestrator.h
│   ├── orchestrator_context.h
│   ├── pipeline_state.h
│   ├── defaults.h
│   ├── model_params.h
│   └── nodes/
│       ├── analytical_node.h
│       └── prefill_node.h
├── src/
│   ├── core/
│   │   ├── kv_handoff.cpp         # memcpy materialize + documented bind no-op
|   |   ├── pipeline_state.cpp        
│   │   └── orchestrator.cpp       # DAG + async workers + join + free_all
│   ├── memory/
│   │   └── memory_pool.cpp        # host buffers + llama_state_get_size sizing
│   └── nodes/
│       ├── analytical_node.cpp
│       └── prefill_node.cpp
├── tests/
│   ├── test_rope_invariant.cpp
│   └── test_memory_lifecycle.cpp
└── third_party/llama.cpp/         # local llama.cpp checkout (not redistributed here)
```

## 📊 Measured results (4K context, GTX 1080, Qwen2.5-7B Q4_K_M)

Captured with **`scripts/benchmark_campaign.py`** (~3501-token shared document, two analytical branches). Best of 3 trials:

| Metric | Baseline | SwarmKV | Improvement |
|--------|----------|---------|-------------|
| End-to-end (ms) | 10,275 | 5,272 | **48.7%** (~1.95×) |
| Branch-2 TTFT proxy (ms) | 4,339 | 83 | **98.1%** (~52×) |

Full narrative, methodology, and charts: **`examples/example-run-results/final_result.docx`**. Raw numbers: **`best_run.json`**, **`all_trials.csv`**.

## 🛣️ Roadmap (Coming in the Next Release)

Active work-in-progress items already on the bench:

* **Speculative branching (streaming prefill).** Readiness watermark so branches that only need a prefix slice can start decode before the full document prefill finishes (see **`projectdocs/project_planning.md`**).
* **Real KV bind hook.** Replacing the **`bind_contiguous_cache` no-op** with a **pin-verified** upstream API when a stable external-KV attachment path exists—plus a **golden-string** regression vs a single vanilla context.
* **Device-aware materialize.** Backend-scheduled copy when KV buffers live on CUDA/Metal without host mapping.
* **KV snapshot to disk.** Serializing prefilled prefix state for **fast cold start** with explicit **(model hash, quant, n_ctx, commit)** identity in the file header.
* **Bounded thread pool.** Migrating from **one `std::async` per node** to a **fixed worker pool** for large DAGs while preserving the **`LlamaGuard`** policy.
* **PrefillNode timing fix.** Per-node **`timings_ms`** for **`Prefiller`** currently reports `0` in the benchmark harness (instrumentation gap; effective prefill cost is derivable from end-to-end metrics).

## 🙏 Acknowledgments

Built on **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** and the **ggml** backend layer it ships with. Upstream tensor, batch, and memory APIs evolve quickly—**pin a commit** when you freeze a demo, paper, or interview artifact. GGUF model weights and upstream documentation remain the property of their respective licensors and are **not** redistributed by this project.
