/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package common

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

	// ClusterAPIManagerAppliedAnnotationKey marks that cluster-api 003-manage.yaml
	// has been applied (deferred to postprocess completion).
	ClusterAPIManagerAppliedAnnotationKey = "bke.bocloud.com/cluster-api-manager-applied"

	// 不可变 OS 适配
	ImmutableOSAnnotation      = "openfuyao.io/immutable-os"
	ImmutableOSAnnotationValue = "kubeos"
	// ImmutableOSCustomExtraKey 用于在 BKEConfig 层面传递不可变 OS 标志到 agent 端 plugin
	ImmutableOSCustomExtraKey = "immutable-os"

	// KubeOS 节点标签，用于 addon 调度和 KubeOS OS CR 节点选择
	LabelImmutableOSKey           = "node.openfuyao.io/os"
	LabelImmutableOSValue         = "kubeos"
	LabelImmutableRoleKey         = "node.openfuyao.io/role"
	LabelImmutableRoleWorkerValue = "worker"
	// LabelImmutableNodeSelectorKey KubeOS OS CR 节点选择标签
	LabelImmutableNodeSelectorKey   = "upgrade.openeuler.org/node-selector"
	LabelImmutableNodeSelectorValue = "default-worker-pool"
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
