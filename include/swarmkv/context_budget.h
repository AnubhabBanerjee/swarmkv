#pragma once

#include "swarmkv/defaults.h"

#include <algorithm>
#include <fstream>
#include <llama.h>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Tokenize UTF-8 text for context-budget checks (matches node tokenize flags).
 */
inline std::vector<llama_token> swarmkv_tokenize_for_budget(llama_model * model, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 8u);
    int32_t n = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            false);
        if (n < 0) {
            throw std::runtime_error("swarmkv_tokenize_for_budget: tokenize failed.");
        }
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

inline std::string swarmkv_read_document_file(const std::string & path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("swarmkv_read_document_file: cannot read " + path);
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/**
 * @brief Fail-fast if shared prefix + largest branch prompt + headroom exceeds n_ctx.
 *
 * @param prefix_tokens Token count of the shared document prefill.
 * @param branch_prompts Branch prompts that will decode after the prefix (RoPE continues at prefix length).
 * @param generation_headroom Reserved slots for continuation / logits (default 128).
 */
inline void swarmkv_validate_context_budget(
    llama_model * model,
    int32_t prefix_tokens,
    const std::vector<std::string> & branch_prompts,
    uint32_t n_ctx = kSwarmkvDefaultPipelineCtx,
    int32_t generation_headroom = 128) {
    if (prefix_tokens < 0) {
        throw std::invalid_argument("swarmkv_validate_context_budget: negative prefix_tokens.");
    }

    int32_t max_branch = 0;
    for (const auto & prompt : branch_prompts) {
        const int32_t n = static_cast<int32_t>(swarmkv_tokenize_for_budget(model, prompt).size());
        max_branch = std::max(max_branch, n);
    }

    const int32_t limit = static_cast<int32_t>(n_ctx);
    const int32_t required = prefix_tokens + max_branch + generation_headroom;
    if (required > limit) {
        throw std::runtime_error(
            "Context budget exceeded: prefix_tokens=" + std::to_string(prefix_tokens) +
            " max_branch_tokens=" + std::to_string(max_branch) +
            " headroom=" + std::to_string(generation_headroom) +
            " required=" + std::to_string(required) + " n_ctx=" + std::to_string(limit));
    }
}
