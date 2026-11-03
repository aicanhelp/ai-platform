// Copyright 2017 The Ray Authors.
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

#include "ray/gcs/gcs_server/gcs_submit_manager.h"

#include <nlohmann/json.hpp>

namespace ray {
namespace gcs {

// Please keep this in sync with the definition in ray_constants.py.
const std::string kRayInternalNamespacePrefix = "_ray_internal_";

// Please keep these in sync with the definition in dashboard/modules/job/common.py.
const std::string kJobDataKeyPrefix = kRayInternalNamespacePrefix + "job_info_";
inline std::string JobDataKey(const std::string &submission_id) {
  return kJobDataKeyPrefix + submission_id;
}

inline Status ValidateKey(const std::string &key) {
  constexpr std::string_view kNamespacePrefix = "@namespace_";
  if (absl::StartsWith(key, kNamespacePrefix)) {
    return Status::KeyError(absl::StrCat("Key can't start with ", kNamespacePrefix));
  }
  return Status::OK();
}

void GcsSubmitManager::HandleGetStatus(rpc::GetStatusRequest request,
                                       rpc::GetStatusReply *reply,
                                       rpc::SendReplyCallback send_reply_callback) {
  const std::string &job_key = JobDataKey(request.submission_id());
  auto status = ValidateKey(job_key);
  if (!status.ok()) {
    GCS_RPC_SEND_REPLY(send_reply_callback, reply, status);
  } else {
    auto callback = [reply, send_reply_callback](std::optional<std::string> val) {
      if (val) {
        nlohmann::json job = nlohmann::json::parse(*val);
        reply->set_job_status(job["status"]);
        GCS_RPC_SEND_REPLY(send_reply_callback, reply, Status::OK());
      } else {
        GCS_RPC_SEND_REPLY(
            send_reply_callback, reply, Status::NotFound("Failed to find the job"));
      }
    };
    internal_kv_.Get("job", job_key, {std::move(callback), io_context_});
  }
}

}  // namespace gcs
}  // namespace ray
