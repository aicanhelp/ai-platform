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

#pragma once

#include <optional>
#include <vector>

#include "ray/raylet/scheduling/policy/scheduling_policy.h"

namespace ray {
namespace raylet_scheduling_policy {

class BinPackSchedulingPolicy : public ISchedulingPolicy {
 public:
  BinPackSchedulingPolicy(scheduling::NodeID local_node_id,
                          const absl::flat_hash_map<scheduling::NodeID, Node> &nodes,
                          std::function<bool(scheduling::NodeID)> is_node_alive)
      : local_node_id_(local_node_id), nodes_(nodes), is_node_alive_(is_node_alive) {}

  scheduling::NodeID Schedule(const ResourceRequest &resource_request,
                              SchedulingOptions options) override;

 private:
  /// Return true if the node is alive and its total resource
  /// satisify the filter and resource requrement.
  bool IsNodeFeasible(const scheduling::NodeID &node_id,
                      const NodeResources &node_resources,
                      const ResourceRequest &resource_request) const;

  /// helper function compute a score between 0-1 indicates
  /// the preference of the node (the lower score,
  /// the more preferable.

  scheduling::NodeID GetBestNode(
      std::vector<std::pair<scheduling::NodeID, float>> &node_scores) const;

  /// \param resource_request: The resource request we're attempting to schedule.
  /// \param require_available: If true, only schedule on nodes who have resources
  /// available to fulfill the request. Otherwise, schedule on nodes whose resources
  /// capacity can fulfill the request, even if the resources are not currently
  /// available.
  ///
  /// \return -1 if the task is unfeasible, otherwise the node id (key in `nodes`) to
  /// schedule on.
  scheduling::NodeID ScheduleImpl(const ResourceRequest &resource_request,
                                  bool avoid_local_node,
                                  bool require_available,
                                  bool cpu_nodes_priority = false);

  /// Identifier of local node.
  const scheduling::NodeID local_node_id_;
  /// List of nodes in the clusters and their resources organized as a map.
  /// The key of the map is the node ID.
  const absl::flat_hash_map<scheduling::NodeID, Node> &nodes_;
  /// Function Checks if node is alive.
  std::function<bool(scheduling::NodeID)> is_node_alive_;
};
}  // namespace raylet_scheduling_policy
}  // namespace ray
