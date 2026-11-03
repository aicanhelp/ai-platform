/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clusterutil

import (
	"fmt"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

// FullyControlled returns true if the cluster is a full control by BKE, only for cluster from "Bocloud"
func FullyControlled(bkeCluster client.Object) bool {
	if IsBKECluster(bkeCluster) {
		return true
	}
	if IsOtherCluster(bkeCluster) {
		return false
	}
	if IsBocloudCluster(bkeCluster) {
		v, ok := annotation.HasAnnotation(bkeCluster, annotation.KONKFullManagementClusterAnnotationKey)
		return ok && v == "true"
	}
	return false
}

// IsBKECluster returns true if the cluster is a BKE cluster
// if the cluster is a BKE cluster, the annotation "bke.bocloud.com/cluster-from" is set to "bke" or not set
func IsBKECluster(bkeCluster client.Object) bool {
	v, ok := annotation.HasAnnotation(bkeCluster, common.BKEClusterFromAnnotationKey)
	return !ok || v == common.BKEClusterFromAnnotationValueBKE || v == ""
}

// IsBocloudCluster returns true if the cluster is a BKE cluster
func IsBocloudCluster(bkeCluster client.Object) bool {
	v, ok := annotation.HasAnnotation(bkeCluster, common.BKEClusterFromAnnotationKey)
	return ok && v == common.BKEClusterFromAnnotationValueBocloud
}

// IsOtherCluster returns true if the cluster is a other cluster
func IsOtherCluster(bkeCluster client.Object) bool {
	v, ok := annotation.HasAnnotation(bkeCluster, common.BKEClusterFromAnnotationKey)
	return ok && v == common.BKEClusterFromAnnotationValueOther
}

func GetClusterType(bkeCluster client.Object) string {
	v, ok := annotation.HasAnnotation(bkeCluster, common.BKEClusterFromAnnotationKey)
	if !ok || v == "" {
		return "bke"
	}
	return v
}

// ClusterInfoHasCollected returns true if the cluster information has been collected
func ClusterInfoHasCollected(bkeCluster client.Object) bool {
	return ClusterBaseInfoHasCollected(bkeCluster) || ClusterAgentInfoHasCollected(bkeCluster)
}

func ClusterBaseInfoHasCollected(bkeCluster client.Object) bool {
	v, ok := annotation.HasAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey)
	if !ok {
		return false
	}
	scope := strings.Split(v, ",")
	return utils.ContainsString(scope, "base")
}

func ClusterAgentInfoHasCollected(bkeCluster client.Object) bool {
	v, ok := annotation.HasAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey)
	if !ok {
		return false
	}
	scope := strings.Split(v, ",")
	return utils.ContainsString(scope, "agent")
}

// MarkClusterBaseInfoCollected marks the cluster Base information has been collected
func MarkClusterBaseInfoCollected(bkeCluster client.Object) {
	if ClusterBaseInfoHasCollected(bkeCluster) {
		return
	}
	if v, ok := annotation.HasAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey); !ok {
		annotation.SetAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey, "base")
	} else {
		v = fmt.Sprintf("%s,%s", v, "base")
		annotation.SetAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey, v)
	}
}

func MarkClusterAgentInfoCollected(bkeCluster client.Object) {
	if ClusterAgentInfoHasCollected(bkeCluster) {
		return
	}
	if v, ok := annotation.HasAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey); !ok {
		annotation.SetAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey, "agent")
	} else {
		v = fmt.Sprintf("%s,%s", v, "agent")
		annotation.SetAnnotation(bkeCluster, annotation.ClusterCollectdAnnotationKey, v)
	}
}

func MarkClusterFullyControlled(bkeCluster client.Object) {
	annotation.SetAnnotation(bkeCluster, annotation.KONKFullManagementClusterAnnotationKey, "true")
}
