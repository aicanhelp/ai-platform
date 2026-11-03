/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package clusterutil

import (
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
)

// IsImmutableOSMode 判断集群 worker 节点是否使用不可变 OS（KubeOS）
func IsImmutableOSMode(bkeCluster *bkev1beta1.BKECluster) bool {
	if bkeCluster == nil || bkeCluster.Annotations == nil {
		return false
	}
	return bkeCluster.Annotations[bkecommon.ImmutableOSAnnotation] == bkecommon.ImmutableOSAnnotationValue
}

// IsImmutableOSModeConf 判断集群 worker 节点是否使用不可变 OS（KubeOS）
// 接受 bkecommon/v1beta1.BKECluster 类型（agent 端使用）
func IsImmutableOSModeConf(bkeCluster *confv1beta1.BKECluster) bool {
	if bkeCluster == nil || bkeCluster.Annotations == nil {
		return false
	}
	return bkeCluster.Annotations[bkecommon.ImmutableOSAnnotation] == bkecommon.ImmutableOSAnnotationValue
}
