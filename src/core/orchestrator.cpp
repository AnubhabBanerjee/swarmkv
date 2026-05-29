// orchestrator.h declares Orchestrator and the DAG execution API surface.
#include "swarmkv/orchestrator.h"

// orchestrator_context.h defines OrchestratorContext passed into each node execute() call.
#include "swarmkv/orchestrator_context.h"

// defaults.h provides kSwarmkvDefaultPipelineCtx used when constructing llama_context_params.
#include "swarmkv/defaults.h"

// prefix_constants.h provides kSwarmkvWaitForPrefillComplete sentinel for branch gating logic.
#include "swarmkv/prefix_constants.h"

// future enables std::async worker tasks and shared_future dependency barriers.
#include <future>

// thread is included for completeness with std::async usage patterns in this translation unit.
#include <thread>

// validate_acyclic performs DFS cycle detection before any worker threads are spawned.
void Orchestrator::validate_acyclic() {
    // state maps each node name to DFS color: 0 unvisited, 1 visiting, 2 visited.
    std::unordered_map<std::string, int> state;
    // dfs lambda walks adjacency lists and throws when a back-edge indicates a cycle.
    auto dfs = [&](auto self, const std::string & u) -> void {
        // Mark node u as currently on the recursion stack (visiting).
        state[u] = 1;
        // Explore all outgoing dependency edges from u to downstream nodes v.
        for (const auto & v : adj[u]) {
            // If v is visiting, we found a cycle u -> v and must abort pipeline setup.
            if (state[v] == 1) {
                // Throw with edge names so graph misconfiguration is easy to diagnose.
                throw std::runtime_error("Dependency cycle detected: " + u + " -> " + v);
            }
            // Recurse only when v has not been fully processed yet.
            if (state[v] == 0) {
                // Continue DFS from child node v.
                self(self, v);
            }
        }
        // Mark u as fully processed (visited) after all descendants are checked.
        state[u] = 2;
    };
    // Start DFS from every node to cover disconnected subgraphs in the dependency map.
    for (const auto & [node, neighbors] : adj) {
        // neighbors is unused here but structured binding keeps iteration consistent with adj type.
        (void) neighbors;
        // Launch DFS only for nodes not yet visited.
        if (state[node] == 0) {
            // Run DFS starting at node to validate reachability without cycles.
            dfs(dfs, node);
        }
    }
}

// add_dependency records a directed edge from_node must complete (or watermark) before to_node starts.
void Orchestrator::add_dependency(const std::string & from_node, const std::string & to_node) {
    // Append to_node to from_node's outgoing adjacency list.
    adj[from_node].push_back(to_node);
    // Append from_node to to_node's reverse adjacency list for upstream lookups.
    rev_adj[to_node].push_back(from_node);
    // Ensure to_node exists in adj even if it has no outgoing edges yet.
    if (adj.find(to_node) == adj.end()) {
        // Create an empty outgoing list placeholder for to_node.
        adj[to_node] = {};
    }
    // Ensure from_node exists in rev_adj even if it has no upstream dependencies.
    if (rev_adj.find(from_node) == rev_adj.end()) {
        // Create an empty reverse list placeholder for from_node.
        rev_adj[from_node] = {};
    }
}

// get_dependencies_for_node returns upstream node names that must be satisfied before execution.
std::vector<std::string> Orchestrator::get_dependencies_for_node(const std::string & node_name) const {
    // Lookup reverse adjacency list entry for the requested node name.
    auto it = rev_adj.find(node_name);
    // If node has no rev_adj entry, it has no upstream dependencies (e.g., root prefill).
    if (it == rev_adj.end()) {
        // Return empty vector meaning no wait barriers besides watermark gating logic.
        return {};
    }
    // Return the stored upstream dependency list for orchestrator worker setup.
    return it->second;
}

// add_node registers an ExecutionNode instance under a unique orchestrator name.
void Orchestrator::add_node(const std::string & name, std::unique_ptr<ExecutionNode> node) {
    // Reject duplicate names because futures/promises are keyed by node name strings.
    if (nodes.find(name) != nodes.end()) {
        // Throw before execution starts so misconfigured graphs fail immediately.
        throw std::runtime_error("Node with name '" + name + "' already registered.");
    }
    // Transfer ownership of the node into the orchestrator registry map.
    nodes[name] = std::move(node);
}

// execute_pipeline spawns worker threads, applies watermark gating, and joins all tasks.
void Orchestrator::execute_pipeline(PipelineState * state) {
    // V2: register immutable snapshot milestones for any branch that declares required_prefix_tokens>0.
    for (const auto & [name, node] : nodes) {
        // Silence unused structured binding name when only the node pointer is needed.
        (void) name;
        // Read the branch watermark requirement from the ExecutionNode virtual interface.
        const int32_t w = node->required_prefix_tokens();
        // Positive milestones require PrefillNode to freeze KV at commit_watermark(w).
        if (w > 0) {
            // Register milestone before threads start so commits can store immutable snapshots.
            state->register_snapshot_milestone(w);
            // Count branch consumers so the prefiller can pause and release VRAM at each milestone.
            state->register_milestone_consumer(w);
        }
    }
    // completion_futures allows multiple downstream nodes to wait on the same upstream completion.
    std::unordered_map<std::string, std::shared_future<void>> completion_futures;
    // completion_promises are fulfilled by worker threads when execute() finishes (or throws).
    std::unordered_map<std::string, std::promise<void>> completion_promises;
    // worker_tasks retains std::future handles returned by std::async for joining at the end.
    std::vector<std::future<void>> worker_tasks;
    // Initialize promise/future pairs for every registered node before spawning threads.
    for (const auto & [name, node] : nodes) {
        // node pointer unused in this loop; only the name key matters for promise map setup.
        (void) node;
        // Create default promise for this node name.
        completion_promises[name] = std::promise<void>();
        // Share the future so multiple dependents can wait without transferring ownership.
        completion_futures[name] = completion_promises[name].get_future().share();
    }
    // Spawn one async worker per node; dependency order enforced inside each lambda.
    for (const auto & [name, node] : nodes) {
        // node is captured by the orchestrator map lookup inside lambda via nodes.at(name).
        (void) node;
        // Read dependency list for this node from reverse adjacency map.
        auto dependencies = get_dependencies_for_node(name);
        // Launch async worker that waits on dependencies/watermarks then calls execute().
        worker_tasks.push_back(std::async(
            std::launch::async,
            [this, name, state, dependencies, &completion_promises, &completion_futures]() {
                // Read this node's watermark requirement once for dependency gating decisions.
                const int32_t req = nodes.at(name)->required_prefix_tokens();
                // Wait for each upstream dependency according to V2 watermark rules.
                for (const auto & dep_name : dependencies) {
                    // Resolve upstream node pointer for prefill provider detection.
                    ExecutionNode * dep = nodes.at(dep_name).get();
                    // If upstream is prefiller and this branch uses watermark gating, wait on watermark.
                    if (dep->is_prefill_provider() && req >= 0) {
                        // Block until PipelineState watermark >= required_prefix_tokens (speculative start).
                        state->wait_for_watermark(req);
                    } else {
                        // Otherwise preserve V1 behavior: wait until upstream node thread completes.
                        completion_futures.at(dep_name).wait();
                    }
                }
                // Build llama_context_params with orchestrator default n_ctx budget.
                llama_context_params params = llama_context_default_params();
                // Lift n_ctx to SwarmKV default pipeline context for multi-k token documents.
                params.n_ctx = kSwarmkvDefaultPipelineCtx;
                // Bundle model/pool/name into OrchestratorContext for node execute().
                OrchestratorContext ctx = {
                    this->memory_pool->get_model(),
                    params,
                    this->memory_pool,
                    name.c_str(),
                };
                // Run node logic and fulfill promise so dependents can proceed.
                try {
                    // Dispatch to PrefillNode or AnalyticalNode implementation.
                    nodes.at(name)->execute(state, &ctx);
                    // Signal successful completion to shared_future waiters.
                    completion_promises.at(name).set_value();
                } catch (...) {
                    if (req > 0) {
                        state->signal_milestone_consumed(req);
                    }
                    try {
                        completion_promises.at(name).set_exception(std::current_exception());
                    } catch (...) {
                    }
                    throw;
                }
            }));
    }
    // Join every worker task on the main thread before freeing MemoryPool allocations.
    for (auto & task : worker_tasks) {
        // Wait for worker completion to avoid use-after-free on PipelineState buffers.
        task.wait();
    }
    // Free all ggml buffers tracked by the MemoryPool after all nodes have joined.
    memory_pool->free_all();
}

// Orchestrator constructor stores the MemoryPool pointer used for allocation and teardown.
Orchestrator::Orchestrator(MemoryPool * pool) : memory_pool(pool) {
    // Reject null pool because nodes require allocate_prefix_cache and allocate_branch_cache.
    if (!memory_pool) {
        // Throw immediately so callers cannot construct a broken orchestrator instance.
        throw std::runtime_error("Orchestrator initialized with a null MemoryPool.");
    }
}
