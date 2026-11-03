// Copyright 2021 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/raylet/scheduling/policy/bin_pack_scheduling_policy.h"

#include <functional>

#include "ray/util/container_util.h"
#include "ray/util/util.h"

namespace ray {

namespace raylet_scheduling_policy {

bool BinPackSchedulingPolicy::IsNodeFeasible(
    const scheduling::NodeID &node_id,
    const NodeResources &node_resources,
    const ResourceRequest &resource_request) const {
  if (!is_node_alive_(node_id)) {
    return false;
  }

  return node_resources.IsFeasible(resource_request);
}

scheduling::NodeID BinPackSchedulingPolicy::GetBestNode(
    std::vector<std::pair<scheduling::NodeID, float>> &node_scores) const {
  RAY_CHECK(!node_scores.empty());
  std::sort(
      node_scores.begin(),
      node_scores.end(),
      [](const std::pair<scheduling::NodeID, float> &a,
         const std::pair<scheduling::NodeID, float> &b) { return a.first < b.first; });
  // Then sort nodes by scores and preserve the order of equivalent elements.
  std::stable_sort(
      node_scores.begin(),
      node_scores.end(),
      [](const std::pair<scheduling::NodeID, float> &a,
         const std::pair<scheduling::NodeID, float> &b) { return a.second > b.second; });

  return node_scores[0].first;
}

scheduling::NodeID BinPackSchedulingPolicy::ScheduleImpl(
    const ResourceRequest &resource_request,
    bool avoid_local_node,
    bool require_node_available,
    bool cpu_nodes_priority) {
  // Nodes that are feasible and currently have available resources.
  std::vector<std::pair<scheduling::NodeID, float>> available_nodes;
  std::vector<std::pair<scheduling::NodeID, float>> only_cpu_available_nodes;
  // Nodes that are feasible but currently do not have available resources.
  std::vector<std::pair<scheduling::NodeID, float>> feasible_and_unavailable_nodes;
  std::vector<std::pair<scheduling::NodeID, float>>
      only_cpu_feasible_and_unavailable_nodes;
  for (const auto &pair : nodes_) {
    const auto &node_id = pair.first;
    if (node_id == local_node_id_ && avoid_local_node) {
      continue;
    }
    const auto &node_resources = pair.second.GetLocalView();
    if (IsNodeFeasible(node_id, node_resources, resource_request)) {
      bool ignore_pull_manager_at_capacity = false;
      bool is_available =
          node_resources.IsAvailable(resource_request, ignore_pull_manager_at_capacity);
      float node_score = node_resources.CalculateCriticalResourceUtilization();
      RAY_LOG(DEBUG) << "Node " << node_id.ToInt() << " is "
                     << (is_available ? "available" : "not available") << " for request "
                     << resource_request.DebugString()
                     << " with critical resource utilization " << node_score
                     << " based on local view " << node_resources.DebugString();
      if (is_available) {
        if (cpu_nodes_priority && !resource_request.Has(scheduling::ResourceID("NPU")) &&
            !node_resources.total.Has(scheduling::ResourceID("NPU"))) {
          only_cpu_available_nodes.push_back({node_id, node_score});
        } else {
          available_nodes.push_back({node_id, node_score});
        }
      } else {
        if (cpu_nodes_priority && !resource_request.Has(scheduling::ResourceID("NPU")) &&
            !node_resources.total.Has(scheduling::ResourceID("NPU"))) {
          only_cpu_feasible_and_unavailable_nodes.push_back({node_id, node_score});
        } else {
          feasible_and_unavailable_nodes.push_back({node_id, node_score});
        }
      }
    }
  }

  if (!only_cpu_available_nodes.empty()) {
    return GetBestNode(only_cpu_available_nodes);
  } else if (!available_nodes.empty()) {
    // First prioritize available nodes.
    return GetBestNode(available_nodes);
  } else if (!only_cpu_feasible_and_unavailable_nodes.empty() &&
             !require_node_available) {
    return GetBestNode(only_cpu_feasible_and_unavailable_nodes);
  } else if (!feasible_and_unavailable_nodes.empty() && !require_node_available) {
    // If there are no available nodes, and the caller is okay with an
    // unavailable node, check the feasible nodes next.
    return GetBestNode(feasible_and_unavailable_nodes);
  } else {
    return scheduling::NodeID::Nil();
  }
}

scheduling::NodeID BinPackSchedulingPolicy::Schedule(
    const ResourceRequest &resource_request, SchedulingOptions options) {
  RAY_CHECK(options.scheduling_type == SchedulingType::BINPACK)
      << "BinpPackPolicy policy requires type = BINPACK";
  return ScheduleImpl(resource_request,
                      options.avoid_local_node,
                      options.require_node_available,
                      options.cpu_nodes_priority);
}

}  // namespace raylet_scheduling_policy
}  // namespace ray
