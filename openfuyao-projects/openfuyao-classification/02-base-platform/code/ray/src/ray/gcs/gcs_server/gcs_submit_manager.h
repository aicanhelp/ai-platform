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

#pragma once

#include "ray/gcs/gcs_server/gcs_function_manager.h"
#include "ray/gcs/gcs_server/gcs_table_storage.h"
#include "ray/rpc/gcs_server/gcs_rpc_server.h"

namespace ray {
namespace gcs {

/// This implementation class of `SubmitInfoHandler`.
class GcsSubmitManager : public rpc::SubmitInfoHandler {
 public:
  explicit GcsSubmitManager(InternalKVInterface &internal_kv,
                            instrumented_io_context &io_context)
      : internal_kv_(internal_kv), io_context_(io_context) {}

  void HandleGetStatus(rpc::GetStatusRequest request,
                       rpc::GetStatusReply *reply,
                       rpc::SendReplyCallback send_reply_callback) override;

 private:
  InternalKVInterface &internal_kv_;
  instrumented_io_context &io_context_;
};

}  // namespace gcs
}  // namespace ray
