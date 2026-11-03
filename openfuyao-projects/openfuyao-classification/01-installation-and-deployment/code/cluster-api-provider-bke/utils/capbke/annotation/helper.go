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

package annotation

import (
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
)

const (
	BKEClusterDryRunAnnotationKey          = "bke.bocloud.com/dryrun"
	BKEClusterPauseAnnotationKey           = "bke.bocloud.com/pause"
	CommandReconciledAnnotationKey         = "bke.bocloud.com/command-reconciled"
	ClusterCollectdAnnotationKey           = "bke.bocloud.com/collectd"
	KONKFullManagementClusterAnnotationKey = "bke.bocloud.com/full-management"
	LastUpdateConfigurationAnnotationKey   = "bke.bocloud.com/last-update-configuration"
	AtPrecheckPhaseAnnotationKey           = "bke.bocloud.com/at-precheck-phase"
	BKEMachineProviderIDAnnotationKey      = "bke.bocloud.com/providerID"
	EtcdAdvertiseClientUrlsAnnotationKey   = "kubeadm.kubernetes.io/etcd.advertise-client-urls"

	StatusRecordAnnotationKey = "bke.bocloud.com/status-record"
)

// BKE operations and internal workflow annotations.
const (
	ManageClusterEtcdCertDirAnnotationKey = "etcd-cert-dir"
	DeployActionAnnotationKey             = "deployAction"
	DeployActionK8sUpgrade                = "k8s_upgrade"
	ProviderRestartedAtAnnotationKey      = "bke.openfuyao.cn/restartedAt"
)

// feature gates annotation key
const (
	// RetryAnnotationKey is the annotation key for retry
	RetryAnnotationKey = "bke.bocloud.com/retry"
	// ControlPlaneConfigUpdateFailCountAnnotationKey is the annotation key for control plane config update reconcile failure count.
	ControlPlaneConfigUpdateFailCountAnnotationKey = "bke.bocloud.com/controlplane-config-update-fail-count"
	// EtcdConfigUpdateFailCountAnnotationKey is the annotation key for etcd config update reconcile failure count.
	EtcdConfigUpdateFailCountAnnotationKey = "bke.bocloud.com/etcd-config-update-fail-count"

	// DeepRestoreNodeAnnotationKey is the annotation key for deep restore node
	DeepRestoreNodeAnnotationKey = "bke.bocloud.com/deep-restore-node"
	// MasterSchedulableAnnotationKey is the annotation key for master schedulable
	MasterSchedulableAnnotationKey = "bke.bocloud.com/master-schedulable"
	// DeleteIgnoreNamespaceAnnotationKey is the annotation key for delete ignore namespace
	DeleteIgnoreNamespaceAnnotationKey = "bke.bocloud.com/ignore-namespace-delete"
	// DeleteIgnoreTargetClusterAnnotationKey is the annotation key for delete ignore target cluster
	DeleteIgnoreTargetClusterAnnotationKey = "bke.bocloud.com/ignore-target-cluster-delete"
	// AppointmentDeletedNodesAnnotationKey is the annotation key for appointment deleted nodes
	AppointmentDeletedNodesAnnotationKey = "bke.bocloud.com/appointment-deleted-nodes"
	// AppointmentAddNodesAnnotationKey is the annotation key for appointment add nodes
	AppointmentAddNodesAnnotationKey = "bke.bocloud.com/appointment-add-nodes"

	// NodeBootWaitTimeOutAnnotationKey is the annotation key for node boot wait timeout
	NodeBootWaitTimeOutAnnotationKey = "bke.bocloud.com/node-boot-wait-timeout"
	// ClusterTrackerHealthyCheckFailedAnnotationKey is the annotation key for cluster tracker healthy check faild
	ClusterTrackerHealthyCheckFailedAnnotationKey = "bke.bocloud.com/cluster-tracker-healthy-check-failed"

	// AddonBootWaitTimeOutAnnotationKey is the annotation key for addon boot wait timeout
	AddonBootWaitTimeOutAnnotationKey = "bke.bocloud.com/addon-boot-wait-timeout"
	// LauncherWaitTimeOutAnnotationKey is the annotation key for bkeagent-launcher pod ready wait timeout
	LauncherWaitTimeOutAnnotationKey = "bke.bocloud.com/launcher-wait-timeout"
)

// ClusterVersion orchestration annotations (written by ClusterVersion reconciler).
const (
	CVOUpgradeReadyAnnotationKey            = "cvo.openfuyao.cn/upgrade-ready"
	CVOClusterVersionAnnotationKey          = "cvo.openfuyao.cn/cluster-version"
	CVOUpgradePathAnnotationKey             = "cvo.openfuyao.cn/upgrade-path"
	UpgradePathOCIDigestAnnotationKey       = "config.openfuyao.com/oci-digest"
	ReleaseImageBKEClusterNameAnnotationKey = "cvo.openfuyao.cn/bkecluster-name"
)

// Kubernetes and OCI protocol annotations used by this project.
const (
	KubernetesConfigHashAnnotationKey = "kubernetes.io/config.hash"
	OCIImageTitleAnnotationKey        = "org.opencontainers.image.title"
)

func SetBKEClusterDefaultAnnotation(bkeCluster client.Object) {
	if _, ok := HasAnnotation(bkeCluster, DeleteIgnoreTargetClusterAnnotationKey); !ok {
		SetAnnotation(bkeCluster, DeleteIgnoreTargetClusterAnnotationKey, "true")
	}
	if _, ok := HasAnnotation(bkeCluster, DeleteIgnoreNamespaceAnnotationKey); !ok {
		SetAnnotation(bkeCluster, DeleteIgnoreNamespaceAnnotationKey, "true")
	}
	if _, ok := HasAnnotation(bkeCluster, common.BKEAgentListenerAnnotationKey); !ok {
		SetAnnotation(bkeCluster, common.BKEAgentListenerAnnotationKey, common.BKEAgentListenerCurrent)
	}
	if _, ok := HasAnnotation(bkeCluster, common.BKEClusterFromAnnotationKey); !ok {
		SetAnnotation(bkeCluster, common.BKEClusterFromAnnotationKey, "")
	}
	if _, ok := HasAnnotation(bkeCluster, DeepRestoreNodeAnnotationKey); !ok {
		SetAnnotation(bkeCluster, DeepRestoreNodeAnnotationKey, "true")
	}
	if _, ok := HasAnnotation(bkeCluster, MasterSchedulableAnnotationKey); !ok {
		SetAnnotation(bkeCluster, MasterSchedulableAnnotationKey, "false")
	}
	if _, ok := HasAnnotation(bkeCluster, NodeBootWaitTimeOutAnnotationKey); !ok {
		SetAnnotation(bkeCluster, NodeBootWaitTimeOutAnnotationKey, "10m")
	}
	if _, ok := HasAnnotation(bkeCluster, LauncherWaitTimeOutAnnotationKey); !ok {
		SetAnnotation(bkeCluster, LauncherWaitTimeOutAnnotationKey, "5m")
	}
}

func HasAnnotation(obj client.Object, key string) (string, bool) {
	annotations := obj.GetAnnotations()
	if annotations == nil {
		return "", false
	}
	v, ok := annotations[key]
	return v, ok
}

func SetAnnotation(obj client.Object, key, value string) {
	annotations := obj.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[key] = value
	obj.SetAnnotations(annotations)
}

func RemoveAnnotation(obj client.Object, key string) {
	annotations := obj.GetAnnotations()
	if annotations == nil {
		return
	}
	delete(annotations, key)
	obj.SetAnnotations(annotations)
}

func BKENormalEventAnnotation() map[string]string {
	return map[string]string{
		common.BKEEventAnnotationKey: "",
	}
}

func BKEFinishEventAnnotation() map[string]string {
	return map[string]string{
		common.BKEEventAnnotationKey:       "",
		common.BKEFinishEventAnnotationKey: "",
	}
}
