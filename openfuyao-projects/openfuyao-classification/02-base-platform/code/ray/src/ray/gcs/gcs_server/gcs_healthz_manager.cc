// Copyright 2025 The Ray Authors.
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

#include "ray/gcs/gcs_server/gcs_healthz_manager.h"

namespace ray::gcs {

void GcsHealthzManager::HandleCheckAlive(rpc::CheckAliveRequest request,
                                         rpc::CheckAliveReply *reply,
                                         rpc::SendReplyCallback send_reply_callback) {
  reply->set_ray_version(kRayVersion);
  for (const auto &addr : request.raylet_address()) {
    bool is_alive = gcs_node_manager_.CheckAlive(addr);
    reply->mutable_raylet_alive()->Add(is_alive);
  }
  GCS_RPC_SEND_REPLY(send_reply_callback, reply, Status::OK());
}

}  // namespace ray::gcs
