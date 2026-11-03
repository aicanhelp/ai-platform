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

package constant

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
)

const (
	// LocalKubeConfigName defines the secret name of local kube config .
	LocalKubeConfigName = "localkubeconfig"
	// LeastPrivilegeKubeConfigName defines the secret name of least privilege kube config.
	LeastPrivilegeKubeConfigName = "leastprivilegekubeconfig"

	KubeadmConfigKey = "kubeadm-config"
	MockData         = ""

	K8sManifestsDir = "/manifests/kubernetes"
	K8sScriptsDir   = "/manifests/bkeinitscript"

	MasterHADomain = "master.bocloud.com"

	BocVersionEnvKey  = "BOC_VERSION"
	DefaultBocVersion = "boc4.0"

	AddonManifestsDir      = "/etc/kubernetes/addon"
	BootstrapKubeConfigDir = "/etc/rancher/k3s/k3s.yaml"
	ConcurrencyEnvKey      = "CONCURRENCY"
	DefaultConcurrency     = 10
	WorkerNodeSkipReason   = "FailedWorkerNodeSkip"
)

// GetLocalKubeConfigObjectKey returns the client.ObjectKey of local kube config.
func GetLocalKubeConfigObjectKey() client.ObjectKey {
	return client.ObjectKey{
		Namespace: metav1.NamespaceSystem,
		Name:      LocalKubeConfigName,
	}
}

// GetLocalConfigMapObjectKey returns the client.ObjectKey of cluster-system/bke-cluster config map.
func GetLocalConfigMapObjectKey() client.ObjectKey {
	return client.ObjectKey{
		Namespace: "cluster-system",
		Name:      common.BKEClusterConfigFileName,
	}
}

// condition/event reason constants
const (
	// DryRunReason (Severity=Info) documents a DemoMachine in dry-run mode.
	DryRunReason = "DryRun"
	// BKEClusterPausedReason (Severity=Info) documents a BKECluster in paused mode.
	BKEClusterPausedReason = "Paused"

	// NodesInfoReadyReason (Severity=Info) documents that the nodes info is ready
	NodesInfoReadyReason = "NodesInfoReady"
	// NodesInfoNotReadyReason (Severity=Warning) documents that the nodes info is not ready
	NodesInfoNotReadyReason = "NodesINfoNotReady"
	// NodeInfoUpdatingReason (Severity=Info) documents that the nodes info is updating
	NodeInfoUpdatingReason = "NodeInfoUpdating"

	// BKEAgentReadyReason (Severity=Info) documents that the BKEAgent is ready
	BKEAgentReadyReason = "BKEAgentReady"
	// BKEAgentNotReadyReason (Severity=Warning) documents that the BKEAgent is not ready
	BKEAgentNotReadyReason = "BKEAgentNotReady"
	// BKEAgentUnknownReason (Severity=Warning) documents that the BKEAgent is unknow
	BKEAgentUnknownReason = "BKEAgentUnknown"
	// BKEAgentUpdatingReason (Severity=Info) documents that the BKEAgent is updating
	BKEAgentUpdatingReason = "BKEAgentPushing"
	// BKEAgentSkipFailedNodesReason (Severity=Info) documents that failed nodes are skipped after NeedSkip
	BKEAgentSkipFailedNodesReason = "BKEAgentSkipFailedNodes"

	// LoadBalancerReadyReason (Severity=Info) documents that the LoadBalancer is ready
	LoadBalancerReadyReason = "LoadBalancerReady"
	// LoadBalancerNotReadyReason (Severity=Warning) documents that the LoadBalancer is not ready
	LoadBalancerNotReadyReason = "LoadBalancerNotReady"
	// LoadBalancerCreatingReason (Severity=Info) documents that the LoadBalancer is creating
	LoadBalancerCreatingReason = "LoadBalancerCreating"
	// LoadBalancerNotConfiguredReason (Severity=Info) documents that the LoadBalancer is not configured
	LoadBalancerNotConfiguredReason = "LoadBalancerNotConfigured"
	// LoadBalancerUpdatingReason (Severity=Info) documents that the LoadBalancer is updating
	LoadBalancerUpdatingReason = "LoadBalancerUpdating"

	// NodesEnvReadyReason (Severity=Info) documents that the NodesEnv is ready
	NodesEnvReadyReason = "NodesEnvReady"
	// NodesEnvNotReadyReason (Severity=Warning) documents that the NodesEnv is not ready
	NodesEnvNotReadyReason = "NodesEnvNotReady"
	// NodesEnvCheckingReason (Severity=Info) documents that the NodesEnv is Checking
	NodesEnvCheckingReason = "NodesEnvChecking"
	// NodesEnvUpdatingReason (Severity=Info) documents that the NodesEnv is updating
	NodesEnvUpdatingReason = "NodesEnvUpdating"
	// NodesPostProcessReadyReason (Severity=Info) documents that postprocess is ready
	NodesPostProcessReadyReason = "NodesPostProcessReady"
	// NodesPostProcessNotReadyReason (Severity=Warning) documents that postprocess is not ready
	NodesPostProcessNotReadyReason = "NodesPostProcessNotReady"
	// NodesPostProcessCheckingReason (Severity=Info) documents that postprocess is checking
	NodesPostProcessCheckingReason = "NodesPostProcessChecking"

	// NodeBootStrapFailedReason (Severity=Info) documents that the node bootstrap failed
	NodeBootStrapFailedReason = "NodeBootStrapFailed"

	// TargetClusterReadyReason (Severity=Info) documents that the target cluster is ready
	TargetClusterReadyReason    = "TargetClusterReady"
	TargetClusterNotReadyReason = "TargetClusterNotReady"

	// TargetClusterDeletedReason (Severity=Info) documents that the target cluster is deleted
	TargetClusterDeletedReason = "TargetClusterDeleted"

	// TargetClusterBootReadyReason (Severity=Info) documents that the target cluster is ready
	TargetClusterBootReadyReason = "TargetClusterBootReady"
	// TargetClusterBootNotReadyReason (Severity=Info) documents that the target cluster is not ready
	TargetClusterBootNotReadyReason = "TargetClusterBootNotReady"
	// TargetClusterBootingReason (Severity=Info) documents that the target cluster is booting
	TargetClusterBootingReason = "TargetClusterBooting"
	// TargetClusterBootingFailedReason (Severity=Warning) documents that the target cluster is boot failed
	TargetClusterBootingFailedReason = "TargetClusterBootingFailed"

	// AddonDeployingReason (Severity=Info) documents that the addon is deploying
	AddonDeployingReason       = "AddonDeploying"
	AddonDeployFailedReason    = "AddonDeployFailed"
	AddonDeploySucceededReason = "AddonDeploySucceeded"
	AddonDeployedReason        = "AddonDeployed"

	AddonUpdatingReason       = "AddonUpdating"
	AddonUpdatedSuccessReason = "AddonUpdatedSuccess"

	// SwitchClusterSuccessReason (Severity=Info) documents that the switch cluster is success
	SwitchClusterSuccessReason = "SwitchClusterSuccess"
	// SwitchClusterFailedReason (Severity=Warning) documents that the switch cluster is failed
	SwitchClusterFailedReason = "SwitchClusterFailed"

	// UpdateClusterFailedReason (Severity=Warning) documents that the update cluster is failed
	UpdateClusterFailedReason = "UpdateClusterFailed"

	// ClusterAPIObjReadyReason (Severity=Info) documents that the ClusterAPIObj is ready
	ClusterAPIObjReadyReason = "ClusterAPIObjReady"
	// ClusterAPIObjNotReadyReason (Severity=Warning) documents that the ClusterAPIObj is not ready
	ClusterAPIObjNotReadyReason = "ClusterAPIObjNotReady"
	// ClusterAPIObjCreatingReason (Severity=Info) documents that the ClusterAPIObj is Checking
	ClusterAPIObjCreatingReason = "ClusterAPIObjCreating"

	// ReconcileErrorReason (Severity=Info) documents that the reconcile has error
	ReconcileErrorReason = "ReconcileError"

	// UpgradeClusterReason (Severity=Info) documents that the upgrade cluster
	UpgradeClusterReason = "UpgradeCluster"

	// HostNameNotUniqueReason (Severity=Warning) documents that the hostname not unique
	HostNameNotUniqueReason = "HostNameNotUnique"
)

// todo 整理reason
const (
	ClusterDeployFailedReason = "ClusterDeployFailed"

	MasterNotInitReason = "MasterNotInit"
	MasterInitReason    = "MasterInit"

	// MasterNotJoinAllReason (Severity=Info) documents that the master is not join
	MasterNotJoinAllReason = "MasterNotJoinAll"

	MasterJoiningReason     = "MasterJoining"
	MasterJoinedReason      = "MasterJoined"
	MasterJoinFailedReason  = "MasterJoinFailed"
	MasterJoinSucceedReason = "MasterJoinSucceed"

	MasterDeletingReason      = "MasterDeleting"
	MasterDeletedReason       = "MasterDeleted"
	MasterDeleteFailedReason  = "MasterDeleteFailed"
	MasterDeleteSucceedReason = "MasterDeleteSucceed"

	MasterUpgradingReason      = "MasterUpgrading"
	MasterUpgradedReason       = "MasterUpgraded"
	MasterUpgradeFailedReason  = "MasterUpgradeFailed"
	MasterUpgradeSucceedReason = "MasterUpgradeSucceed"

	MasterConfigUpdatingReason      = "MasterConfigUpdating"
	MasterConfigUpdatedReason       = "MasterConfigUpdated"
	MasterConfigUpdateFailedReason  = "MasterConfigUpdateFailed"
	MasterConfigUpdateSucceedReason = "MasterConfigUpdateSucceed"

	WorkerJoiningReason     = "WorkerJoining"
	WorkerJoinedReason      = "WorkerJoined"
	WorkerJoinFailedReason  = "WorkerJoinFailed"
	WorkerJoinSucceedReason = "WorkerJoinSucceed"

	WorkerDeletingReason      = "WorkerDeleting"
	WorkerDeletedReason       = "WorkerDeleted"
	WorkerDeleteFailedReason  = "WorkerDeleteFailed"
	WorkerDeleteSucceedReason = "WorkerDeleteSucceed"

	WorkerUpgradingReason      = "WorkerUpgrading"
	WorkerUpgradedReason       = "WorkerUpgraded"
	WorkerUpgradeFailedReason  = "WorkerUpgradeFailed"
	WorkerUpgradeSucceedReason = "WorkerUpgradeSucceed"

	EtcdUpgradingReason = "EtcdUpgrading"
	EtcdUpgradedReason  = "EtcdUpgraded"
	EtcdUpgradeFailed   = "EtcdUpgradeFailed"
	EtcdUpgradeSuccess  = "EtcdUpgradeSuccess"

	EtcdConfigUpdatingReason      = "EtcdConfigUpdating"
	EtcdConfigUpdatedReason       = "EtcdConfigUpdated"
	EtcdConfigUpdateFailedReason  = "EtcdConfigUpdateFailed"
	EtcdConfigUpdateSucceedReason = "EtcdConfigUpdateSucceed"

	ContainerdUpgradingReason = "ContainerdUpgrading"
	ContainerdUpgradeFailed   = "ContainerdUpgradeFailed"
	ContainerdUpgradeSuccess  = "ContainerdUpgradeSuccess"

	ComponentUpgradingReason = "ComponentUpgrading"
	ComponentUpgradeFailed   = "ComponentUpgradeFailed"
	ComponentUpgradeSuccess  = "ComponentUpgradeSuccess"

	AgentUpgradingReason = "AgentUpgrading"
	AgentUpgradeFailed   = "AgentUpgradeFailed"
	AgentUpgradeSuccess  = "AgentUpgradeSuccess"

	ClusterUnhealthyReason = "ClusterUnhealthy"
	ClusterReadyReason     = "ClusterReady"
	ClusterDeletingReason  = "ClusterDeleting"

	ClusterManagingReason           = "ClusterManaging"
	ClusterManageWarningReason      = "ClusterManageWarning"
	CollectClusterInfoFailedReason  = "CollectClusterInfoFailed"
	CollectClusterInfoSucceedReason = "CollectClusterInfoSucceed"

	CommandCreateFailedReason  = "CommandCreateFailed"
	CommandCreateSuccessReason = "CommandCreateSuccess"
	CommandWaitFailedReason    = "CommandWaitFailedFailed"
	CommandExecFailedReason    = "CommandExecFailed"
	CommandExecSuccessReason   = "CommandExecSuccess"

	ProviderSelfUpgradeReason  = "ProviderSelfUpgrading"
	ProviderSelfUpgradeFailed  = "ProviderSelfUpgradeFailed"
	ProviderSelfUpgradeSuccess = "ProviderSelfUpgradeSuccess"

	DrainNodeReason = "DrainNode"

	InternalErrorReason = "InternalError"

	LostBKEConfigConfigMapReason = "LostBKEConfigConfigMap"

	BKEConfigInvalidReason = "BKEConfigNotValid"

	BocloudClusterDataBackupSuccessReason             = "BocloudClusterDataBackupSuccess"
	BocloudClusterDataBackupFailedReason              = "BocloudClusterDataBackupFailed"
	BocloudClusterMasterCertDistributionSuccessReason = "BocloudClusterMasterCertDistributionSuccess"
	BocloudClusterMasterCertDistributionFailedReason  = "BocloudClusterMasterCertDistributionFailed"
	BocloudClusterWorkerCertDistributionSuccessReason = "BocloudClusterWorkerCertDistributionSuccess"
	BocloudClusterWorkerCertDistributionFailedReason  = "BocloudClusterWorkerCertDistributionFailed"
	BocloudClusterEnvInitSuccessReason                = "BocloudClusterEnvInitSuccess"
	BocloudClusterEnvInitFailedReason                 = "BocloudClusterEnvInitFailed"

	ClusterTracker = "ClusterTracker"

	PhaseRunningReason = "PhaseRunning"

	EnvExtraExecScriptFailed  = "EnvExtraExecScriptFailed"
	EnvExtraExecScriptSuccess = "EnvExtraExecScriptSuccess"
	EnvExtraExecScriptSkip    = "EnvExtraExecScriptSkip"

	// ManualInterventionRequiredReason (Severity=Warning) documents that manual intervention is required
	ManualInterventionRequiredReason = "ManualInterventionRequired"
	// MasterInitTimeoutReason (Severity=Warning) documents that master initialization timed out
	MasterInitTimeoutReason = "MasterInitTimeout"
	// NodeBootstrapTerminalReason (Severity=Warning) documents that node bootstrap failed permanently
	NodeBootstrapTerminalReason = "NodeBootstrapTerminal"
	// RetryExhaustedReason (Severity=Warning) documents that retry limit has been exhausted
	RetryExhaustedReason = "RetryExhausted"
)

const (
	// OpenFuyaoSystemPort defines the port of openfuyao system
	OpenFuyaoSystemPort = "31616"
	// OpenFuyaoSystemController defines the controller name of openfuyao system
	OpenFuyaoSystemController = "openfuyao-system-controller"
)

const (
	// Check your certificate 30 days in advance to see if it is about to expire
	CertExpireAlertDays = 30
)

const (
	// ValuesYamlKey defines the key of values.yaml in chart addon configMap
	ValuesYamlKey = "values.yaml"
	// CaKey defines the name of ca.crt
	CaKey = "ca.crt"
	// CertKey defines the name of cert.crt
	CertKey = "cert.crt"
	// KeyKey defines the name of key.key
	KeyKey = "key.key"
	// UsernameKey defines the name of username
	UsernameKey = "username"
	// PasswordKey defines the name of password
	PasswordKey = "password"
)
