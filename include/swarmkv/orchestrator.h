#pragma once

#include "swarmkv/execution_node.h"
#include "swarmkv/memory_pool.h"
#include "swarmkv/pipeline_state.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

// The Orchestrator manages the DAG, validates dependencies, and spawns worker threads.
// This is the reconciled public interface matching our implementation and examples.
class Orchestrator {
public:
    // Constructor: Orchestrator requires a pointer to the MemoryPool for lifecycle management.
    explicit Orchestrator(MemoryPool* pool);

    // Registers a worker node with the orchestrator, transferring ownership.
    void add_node(const std::string& name, std::unique_ptr<ExecutionNode> node);

    // Registers a dependency: 'from_node' must finish before 'to_node' starts.
    void add_dependency(const std::string& from_node, const std::string& to_node);

    // Runs a Depth-First Search to ensure no circular dependencies exist.
    void validate_acyclic();

    // Spawns worker threads and enforces dependency wait-logic via std::future.
    void execute_pipeline(PipelineState* state);

private:
    // Helper to retrieve the list of dependencies for a given node.
    // Used by execute_pipeline to block threads until upstream nodes finish.
    std::vector<std::string> get_dependencies_for_node(const std::string& node_name) const;

    // Pointer to the memory pool; the orchestrator triggers pool cleanup after execution.
    MemoryPool* memory_pool;

    // Registry of all analytical nodes in the pipeline.
    std::unordered_map<std::string, std::unique_ptr<ExecutionNode>> nodes;

    // Adjacency list: maps a node name to the list of nodes that depend on it.
    std::unordered_map<std::string, std::vector<std::string>> adj;
    
    // Reverse adjacency list: maps a node name to its upstream dependencies.
    // Used for easier lookup during thread execution setup.
    std::unordered_map<std::string, std::vector<std::string>> rev_adj;
};