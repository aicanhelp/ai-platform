/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phaseutil

import (
	"strconv"

	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func configUpdateFailCount(obj client.Object, key string) int {
	v, ok := annotation.HasAnnotation(obj, key)
	if !ok || v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// ClearConfigUpdateFailCounts 清除配置更新失败计数（人工 retry 时调用）。
func ClearConfigUpdateFailCounts(cluster *bkev1beta1.BKECluster) {
	if cluster == nil {
		return
	}
	annotation.RemoveAnnotation(cluster, annotation.ControlPlaneConfigUpdateFailCountAnnotationKey)
	annotation.RemoveAnnotation(cluster, annotation.EtcdConfigUpdateFailCountAnnotationKey)
}

// RecordControlPlaneConfigUpdateSuccess 成功后把 desired hash 写到对象上，并清零失败计数。
func RecordControlPlaneConfigUpdateSuccess(cluster *bkev1beta1.BKECluster, desiredHash confv1beta1.ComponentConfigHash) {
	if cluster == nil {
		return
	}
	cluster.Status.ControlPlaneConfigHash = desiredHash
	annotation.RemoveAnnotation(cluster, annotation.ControlPlaneConfigUpdateFailCountAnnotationKey)
}

// RecordControlPlaneConfigUpdateFailure 记录一次调谐失败。
// maxFailures 通常取 statusmanage.ReconcileAllowedFailedCount（来自环境变量 ALLOWED_FAILED_COUNT）。
// 未达 maxFailures：只累加 annotation，不改 Status hash，同 Spec 可继续自动重试。
// 达到 maxFailures（彻底失败）：回写 desired hash。
// maxFailures <= 0 时按 1 处理（首次失败即彻底失败）。
func RecordControlPlaneConfigUpdateFailure(cluster *bkev1beta1.BKECluster, desiredHash confv1beta1.ComponentConfigHash, maxFailures int) bool {
	if cluster == nil {
		return false
	}
	if maxFailures <= 0 {
		maxFailures = 1
	}
	next := configUpdateFailCount(cluster, annotation.ControlPlaneConfigUpdateFailCountAnnotationKey) + 1
	thorough := next >= maxFailures
	annotation.SetAnnotation(cluster, annotation.ControlPlaneConfigUpdateFailCountAnnotationKey, strconv.Itoa(next))
	if thorough {
		cluster.Status.ControlPlaneConfigHash = desiredHash
	}
	return thorough
}

// RecordEtcdConfigUpdateSuccess 成功后把 desired hash 写到对象上，并清零失败计数。
func RecordEtcdConfigUpdateSuccess(cluster *bkev1beta1.BKECluster, desiredHash string) {
	if cluster == nil {
		return
	}
	cluster.Status.EtcdConfigHash = desiredHash
	annotation.RemoveAnnotation(cluster, annotation.EtcdConfigUpdateFailCountAnnotationKey)
}

// RecordEtcdConfigUpdateFailure 同 RecordControlPlaneConfigUpdateFailure，针对 etcd。
func RecordEtcdConfigUpdateFailure(cluster *bkev1beta1.BKECluster, desiredHash string, maxFailures int) bool {
	if cluster == nil {
		return false
	}
	if maxFailures <= 0 {
		maxFailures = 1
	}
	next := configUpdateFailCount(cluster, annotation.EtcdConfigUpdateFailCountAnnotationKey) + 1
	thorough := next >= maxFailures
	annotation.SetAnnotation(cluster, annotation.EtcdConfigUpdateFailCountAnnotationKey, strconv.Itoa(next))
	if thorough {
		cluster.Status.EtcdConfigHash = desiredHash
	}
	return thorough
}
