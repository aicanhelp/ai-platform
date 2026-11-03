/*
* Copyright 2016 The Kubernetes Authors.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Original file: https://github.com/kubernetes/kubernetes/blob/master/pkg/kubelet/cm/helpers_linux.go
*
* Modifications:
* - By YuXiang Guo on 2024-11-25: Removed some codes and kept those would be used by this project
*
* This file is part of a project licensed under the Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
 */

package beresource

const (
	// MaxShares defines the maximum allowed CPU shares (CFS shares) for a cgroup
	MaxShares = 262144
	// MinShares defines the minimum allowed CPU shares (CFS shares) for a cgroup
	MinShares = 2
	// SharesPerCPU represents the number of CPU shares allocated per CPU core
	SharesPerCPU = 1024
	// QuotaPeriod is the default CFS quota period (in microseconds) for CPU bandwidth control
	QuotaPeriod = 100000
	// MilliCPUToCPU is the conversion factor from millicores (mCPU) to full CPU units
	MilliCPUToCPU = 1000
	// MinQuotaPeriod is the minimum allowed CFS quota period (in microseconds)
	MinQuotaPeriod = 1000
)

// MilliCPUToShares converts the milliCPU to CFS shares.
func MilliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		return MinShares
	}
	shares := (milliCPU * SharesPerCPU) / MilliCPUToCPU
	if shares < MinShares {
		return MinShares
	} else if shares > MaxShares {
		return MaxShares
	} else {
		return shares
	}
}

// MilliCPUToQuota converts the milliCPU to CFS quota and period values.
func MilliCPUToQuota(milliCPU int64) int64 {
	if milliCPU == 0 {
		return 0
	}
	quota := (milliCPU * int64(QuotaPeriod)) / MilliCPUToCPU
	if quota < 0 {
		return -1
	} else if quota < MinQuotaPeriod {
		return MinQuotaPeriod
	} else {
		return quota
	}
}
