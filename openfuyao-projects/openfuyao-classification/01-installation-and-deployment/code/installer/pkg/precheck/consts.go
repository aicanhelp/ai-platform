/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package precheck

const (
	// AddMasterNeedMasterCount 扩容master节点需要的master节点数量
	AddMasterNeedMasterCount = 2
	// OneLevelClusterWorkerCount is primary cluster worker node count
	OneLevelClusterWorkerCount = 5
	// TwoLevelClusterWorkerCount is secondary cluster worker node count
	TwoLevelClusterWorkerCount = 24
	// ThreeLevelClusterWorkerCount is three-level cluster worker node count
	ThreeLevelClusterWorkerCount = 100
	// FourLevelClusterWorkerCount is four-level cluster worker node count
	FourLevelClusterWorkerCount = 150
	// FiveLevelClusterWorkerCount is five-level cluster worker node count
	FiveLevelClusterWorkerCount = 200
)
