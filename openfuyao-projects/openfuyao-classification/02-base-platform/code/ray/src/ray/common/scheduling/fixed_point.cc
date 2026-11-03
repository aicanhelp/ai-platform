// Copyright 2020-2021 The Ray Authors.
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

#include "ray/common/scheduling/fixed_point.h"

#include <sstream>

std::vector<FixedPoint> FixedPointVectorFromDouble(const std::vector<double> &vector) {
  std::vector<FixedPoint> vector_fp(vector.size());
  for (size_t i = 0; i < vector.size(); i++) {
    vector_fp[i] = vector[i];
  }
  return vector_fp;
}

std::vector<double> FixedPointVectorToDouble(const std::vector<FixedPoint> &vector_fp) {
  std::vector<double> vector(vector_fp.size());
  for (size_t i = 0; i < vector_fp.size(); i++) {
    vector[i] = FixedPoint(vector_fp[i]).Double();
  }
  return vector;
}

std::string FixedPointVectorToString(const std::vector<FixedPoint> &vector) {
  std::stringstream buffer;
  buffer << "[";
  for (size_t i = 0; i < vector.size(); i++) {
    buffer << vector[i];
    if (i < vector.size() - 1) {
      buffer << ", ";
    }
  }
  buffer << "]";
  return buffer.str();
}

std::vector<size_t> SatisfyNPU310duoTopology(const std::vector<FixedPoint> &npu_vector,
                                             double require_num) {
  std::vector<size_t> free_npu_vector;
  size_t target = (size_t)std::ceil(require_num), require = target;
  if (require == 0) {
    return free_npu_vector;
  }
  size_t group_size = 2;

  // Total number of groups available
  // total_group_capacity = fulfill_group_vector.size() + unfulfill_group_vector.size();
  std::vector<size_t> fulfill_group_vector;
  std::vector<size_t> unfulfill_group_vector;

  for (size_t i = 0; i < npu_vector.size(); i += group_size) {
    size_t group_count = 0;
    for (size_t j = i; j < i + group_size; j++) {
      if (npu_vector[j] == 1.) {
        group_count++;
      }
    }

    if (group_count == 0) {
      continue;
    }
    for (size_t k = i; k < i + group_size; k++) {
      if (npu_vector[k] == 1.) {
        if (group_count != group_size) {
          unfulfill_group_vector.push_back(k);
        } else {
          fulfill_group_vector.push_back(k);
        }
      }
    }
  }

  // handle mod
  if ((target % group_size) != 0) {
    for (size_t i = 0; i < std::min(target, unfulfill_group_vector.size()); i++) {
      require--;
      free_npu_vector.push_back(unfulfill_group_vector[i]);
    }
  }

  // handle group
  for (size_t j = 0; j < std::min(require, fulfill_group_vector.size()); j++) {
    free_npu_vector.push_back(fulfill_group_vector[j]);
  }

  return free_npu_vector.size() == target ? free_npu_vector : std::vector<size_t>();
}
