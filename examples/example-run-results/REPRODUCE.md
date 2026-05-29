# Reproducing SwarmKV 4K benchmark results

This folder holds **generated artifacts** from the benchmark campaign. The **driver script** lives under **`scripts/`** (not here).

## Hardware / software

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce GTX 1080, 8 GiB, CUDA arch 6.1 |
| Model | `example-models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| Context | `n_ctx = 4096` (`kSwarmkvDefaultPipelineCtx`) |
| Document | ~3501 tokens → `examples/base_doc.txt` (target prefill: 3500) |
| Build | `cmake -DGGML_CUDA=ON`, Release |

## Build (CUDA)

```bash
cd swarm-kv
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j"$(nproc)"
```

## Run full campaign (recommended)

Driver: **`scripts/benchmark_campaign.py`** (copy from `.example` first; gitignored when copied).

```bash
cp scripts/benchmark_campaign.py.example scripts/benchmark_campaign.py
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install matplotlib python-docx
nohup scripts/.venv/bin/python scripts/benchmark_campaign.py \
  > examples/example-run-results/campaign.nohup.log 2>&1 &
echo $! > examples/example-run-results/campaign.pid
```

Wait for **`campaign.done`**, then inspect:

| File | Description |
|------|-------------|
| `best_run.json` | Winning trial metrics |
| `all_trials.csv` / `all_trials.json` | All 3 trials |
| `plots/*.png` | TTFT, E2E, prefill breakdown, trial consistency |
| `final_result.docx` | Full narrative report for first-time readers |
| `runs/` | Per-trial stdout/stderr (gitignored) |

## Single benchmark invocation (manual)

```bash
export LD_LIBRARY_PATH=build/bin:$LD_LIBRARY_PATH
export GGML_LOG_LEVEL=ERROR
./build/swarmkv_benchmark \
  example-models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  3500 \
  examples/base_doc.txt
```

Stdout lines like `Baseline_End_To_End_Ms: …` and `SwarmKV_End_To_End_Ms: …` are parsed by the campaign script.

## Regenerate report only (no re-benchmark)

If `best_run.json` and `all_trials.json` already exist:

```bash
scripts/.venv/bin/python -c "
import json, importlib.util
from pathlib import Path
p = Path('scripts/benchmark_campaign.py')
spec = importlib.util.spec_from_file_location('bc', p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r = m.REPO / 'examples/example-run-results'
best = json.loads((r/'best_run.json').read_text())
all_runs = json.loads((r/'all_trials.json').read_text())
m.generate_docx(best, all_runs, m.generate_plots(best, all_runs))
"
```

## Captured results (best trial #3)

| Metric | Value |
|--------|--------|
| End-to-end improvement | **48.69%** (10,275 ms → 5,272 ms) |
| Branch-2 TTFT improvement | **98.09%** (4,339 ms → 83 ms) |
| Redundant prefill eliminated | **8,685 ms** |
| Git commit at capture | `02533b4bb4413ce3e73074aa5088e6e8f7cc2e1a` |

See **`final_result.docx`** for methodology, architecture, limitations, and embedded charts.
