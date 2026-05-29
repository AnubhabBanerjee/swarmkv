# Example run results

**Outputs** from the SwarmKV 4K-context benchmark campaign (baseline vs shared-prefill DAG).

## Start here

| Artifact | Audience |
|----------|----------|
| **`final_result.docx`** | First-time readers — full story (problem, architecture, experiment, results, limitations) |
| **`REPRODUCE.md`** | Engineers reproducing the numbers |
| **`best_run.json`** | Machines / scripts consuming metrics |
| **`plots/`** | Slides and papers |

## How these files were produced

1. Build with CUDA: see root **`README.md`**.
2. Run **`scripts/benchmark_campaign.py`** (from **`scripts/benchmark_campaign.py.example`**).
3. Artifacts are written **here**; the Python driver and venv live under **`scripts/`**.

## What is gitignored

By default, regenerated logs, `runs/`, `campaign.pid`, and optionally JSON/CSV outputs are listed in **`.gitignore`**. Commit **`final_result.docx`** and plots if you want frozen demo artifacts in the repository.
