/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package bkecommon

const (
	// BKEEventAnnotationKey is the annotation key for BKE event
	BKEEventAnnotationKey = "bke.bocloud.com/event"

	// BKEFinishEventAnnotationKey is the annotation Key for BKE complete event
	BKEFinishEventAnnotationKey = "bke.bocloud.com/complete"

	BKEAgentListenerAnnotationKey = "bke.bocloud.com/bkeagent-listener"

	BKEAgentListenerCurrent    = "current"
	BKEAgentListenerBkecluster = "bkecluster"

	BKEClusterFromAnnotationKey          = "bke.bocloud.com/cluster-from"
	BKEClusterFromAnnotationValueBKE     = "bke"
	BKEClusterFromAnnotationValueBocloud = "bocloud"
	BKEClusterFromAnnotationValueOther   = "other"

	BKEClusterConfigFileName = "bke-config"
)

const (
	ImageRegistryKubernetes = "kubernetes"
	ImageRegistryBoc        = "boc"
	ImageRegistryABSSYS     = "abcsys"
	ImageRegistrykube       = "kube"
	ImageRegistryPublic     = "public"
	ImageRegistryMesh       = "mesh"
	ImageRegistryBeyondmesh = "beyondmesh"
	ImageRegistryPaas       = "paas"
	ImageRegistryBMM        = "bmm"
	ImageRegistryBigdata    = "bigdata"
)

// bootstrap phases
const (
	InitControlPlane    = "InitControlPlane"
	JoinControlPlane    = "JoinControlPlane"
	JoinWorker          = "JoinWorker"
	UpgradeControlPlane = "UpgradeControlPlane"
	UpgradeWorker       = "UpgradeWorker"
)
