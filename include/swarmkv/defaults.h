#pragma once

#include <cstdint>

/**
 * Default llama context length for SwarmKV pipelines (orchestrator + pool sizing).
 *
 * Tuned for ~8 GiB VRAM (e.g. GTX 1080 / RTX 3060 8GB): Qwen2.5-7B Q4_K_M weights plus
 * two concurrent AnalyticalNode llama_context instances after PrefillNode completes.
 * n_ctx=8192 with the same model risks OOM when two branches decode in parallel.
 *
 * For 12+ GiB cards or sequential-only graphs, you can raise this locally.
 */
inline constexpr uint32_t kSwarmkvDefaultPipelineCtx = 4096;
