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

package phases

import (
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

// CommonPhases returns the common phase constructors.
func CommonPhases() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return []func(ctx *phaseframe.PhaseContext) phaseframe.Phase{
		NewEnsureFinalizer,
		NewEnsurePaused,
		NewEnsureClusterManage,
		NewEnsureDeleteOrReset,
		NewEnsureDryRun,
	}
}

// DeployPhases returns the deployment phase constructors.
func DeployPhases() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return []func(ctx *phaseframe.PhaseContext) phaseframe.Phase{
		NewEnsureBKEAgent,
		NewEnsureNodesEnv,
		NewEnsureClusterAPIObj,
		NewEnsureCerts,
		NewEnsureLoadBalance,
		NewEnsureMasterInit,
		NewEnsureMasterJoin,
		NewEnsureWorkerJoin,
		NewEnsureAddonDeploy,
		NewEnsureNodesPostProcess,
		NewEnsureAgentSwitch,
	}
}

// LegacyManifestUpgradePhases are superseded by bke-manifests YAML in declarative upgrade
// (provider, kube-proxy, coredns). Kept for the legacy PhaseFlow path.
func LegacyManifestUpgradePhases() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return []func(ctx *phaseframe.PhaseContext) phaseframe.Phase{
		NewEnsureProviderSelfUpgrade,
		NewEnsureComponentUpgrade,
	}
}

// DeclarativeInlineUpgradePhases are resolved via componentfactory.ComponentFactory when
// ReleaseImage DAG runs inline handlers (EnsureAgentUpgrade, EnsureEtcdUpgrade, etc.).
func DeclarativeInlineUpgradePhases() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return []func(ctx *phaseframe.PhaseContext) phaseframe.Phase{
		NewEnsurePreUpgradeResources,
		NewEnsureAgentUpgrade,
		NewEnsureContainerdUpgrade,
		NewEnsureEtcdUpgrade,
		NewEnsureMasterUpgrade,
		NewEnsureWorkerUpgrade,
	}
}

// PostDeployPhases post deploy phases
func PostDeployPhases() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return []func(ctx *phaseframe.PhaseContext) phaseframe.Phase{
		// Legacy manifest-backed upgrades (declarative path uses YAML instead)
		NewEnsureProviderSelfUpgrade,
		NewEnsureAgentUpgrade,
		NewEnsureContainerdUpgrade,
		NewEnsureEtcdUpgrade,
		NewEnsureWorkerUpgrade,
		NewEnsureMasterUpgrade,
		NewEnsureEtcdConfigUpdate,
		NewEnsureMasterConfigUpdate,
		NewEnsureWorkerDelete,
		NewEnsureMasterDelete,
		NewEnsureComponentUpgrade,
		NewEnsureClusterAPIManagerManifest,
		NewEnsureCluster,
	}
}

// DeletePhases delete phases
func DeletePhases() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	return []func(ctx *phaseframe.PhaseContext) phaseframe.Phase{
		NewEnsurePaused,
		NewEnsureDeleteOrReset,
	}
}

// ClusterInitPhaseNames returns the phases used to initialize a cluster.
func ClusterInitPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureFinalizerName,
		EnsureCertsName,
		EnsureClusterAPIObjName,
		EnsureMasterInitName,
		EnsureBKEAgentName,
		EnsureNodesEnvName,
		EnsureLoadBalanceName,
		EnsureAgentSwitchName,
		EnsureNodesPostProcessName,
		EnsureClusterAPIManagerManifestName,
	}
}

// ClusterUpgradePhaseNames tracks legacy PhaseFlow upgrade phases for cluster status.
func ClusterUpgradePhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureAgentUpgradeName,
		EnsureContainerdUpgradeName,
		EnsureMasterUpgradeName,
		EnsureWorkerUpgradeName,
		EnsureComponentUpgradeName,
		EnsureEtcdConfigUpdateName,
		EnsureMasterConfigUpdateName,
	}
}

// DeclarativeClusterUpgradePhaseNames tracks inline DAG-driven upgrade phases.
func DeclarativeClusterUpgradePhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsurePreUpgradeResourcesName,
		EnsureAgentUpgradeName,
		EnsureEtcdUpgradeName,
		EnsureContainerdUpgradeName,
		EnsureMasterUpgradeName,
		EnsureWorkerUpgradeName,
		EnsureProviderSelfUpgradeName,
	}
}

// ClusterScaleMasterDownPhaseNames returns the phases used to remove control-plane nodes.
func ClusterScaleMasterDownPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureMasterDeleteName,
	}
}

// ClusterScaleMasterUpPhaseNames returns the phases used to add control-plane nodes.
func ClusterScaleMasterUpPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureMasterJoinName,
	}
}

// ClusterScaleWorkerDownPhaseNames returns the phases used to remove worker nodes.
func ClusterScaleWorkerDownPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureWorkerDeleteName,
	}
}

// ClusterScaleWorkerUpPhaseNames returns the phases used to add worker nodes.
func ClusterScaleWorkerUpPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureWorkerJoinName,
	}
}

// ClusterAddonsPhaseNames returns the phases used to deploy addons.
func ClusterAddonsPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureAddonDeployName,
	}
}

// ClusterDeletePhaseNames returns the phases used to delete a cluster.
func ClusterDeletePhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureDeleteOrResetName,
	}
}

// ClusterDryRunPhaseNames returns the phases used for a dry run.
func ClusterDryRunPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureDryRunName,
	}
}

// ClusterPausedPhaseNames returns the phases used while a cluster is paused.
func ClusterPausedPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsurePausedName,
	}
}

// ClusterManagePhaseNames returns the phases used to manage an existing cluster.
func ClusterManagePhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureClusterManageName,
	}
}

// CustomSetStatusPhaseNames returns phases that update custom cluster status.
func CustomSetStatusPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsureClusterName,
	}
}

// ClusterDeleteResetPhaseNames returns the phases used to reset or delete a cluster.
func ClusterDeleteResetPhaseNames() []confv1beta1.BKEClusterPhase {
	return []confv1beta1.BKEClusterPhase{
		EnsurePausedName,
		EnsureDeleteOrResetName,
	}
}

func phaseNameCNMap() map[confv1beta1.BKEClusterPhase]string {
	return map[confv1beta1.BKEClusterPhase]string{
		EnsureFinalizerName:                 "部署任务创建",
		EnsurePausedName:                    "集群管理暂停",
		EnsureDeleteOrResetName:             "集群删除",
		EnsureClusterManageName:             "纳管现有集群",
		EnsureDryRunName:                    "DryRun部署",
		EnsureBKEAgentName:                  "推送Agent",
		EnsureNodesEnvName:                  "节点环境准备",
		EnsureClusterAPIObjName:             "ClusterAPI对接",
		EnsureCertsName:                     "集群证书创建",
		EnsureLoadBalanceName:               "集群入口配置",
		EnsureMasterInitName:                "Master初始化",
		EnsureMasterJoinName:                "Master加入",
		EnsureWorkerJoinName:                "Worker加入",
		EnsureAgentSwitchName:               "Agent监听切换",
		EnsureAddonDeployName:               "集群组件部署",
		EnsureNodesPostProcessName:          "后置脚本处理",
		EnsureClusterAPIManagerManifestName: "Cluster-API Manager部署",
		EnsureProviderSelfUpgradeName:       "provider自升级",
		EnsurePreUpgradeResourcesName:       "升级前资源预创建",
		EnsureAgentUpgradeName:              "Agent升级",
		EnsureContainerdUpgradeName:         "Containerd升级",
		EnsureWorkerUpgradeName:             "Worker升级",
		EnsureMasterUpgradeName:             "Master升级",
		EnsureMasterConfigUpdateName:        "Master静态Pod配置更新",
		EnsureEtcdConfigUpdateName:          "Etcd静态Pod配置更新",
		EnsureComponentUpgradeName:          "openFuyao核心组件升级",
		EnsureWorkerDeleteName:              "Worker删除",
		EnsureMasterDeleteName:              "Master删除",
		EnsureClusterName:                   "集群健康检查",
	}
}

// IsDeclarativeInlineUpgradePhase reports whether the phase is handled by the declarative upgrade DAG.
func IsDeclarativeInlineUpgradePhase(name confv1beta1.BKEClusterPhase) bool {
	for _, phaseName := range DeclarativeClusterUpgradePhaseNames() {
		if phaseName == name {
			return true
		}
	}
	return false
}

func ConvertPhaseNameToCN(phase string) string {
	if v, ok := phaseNameCNMap()[confv1beta1.BKEClusterPhase(phase)]; ok {
		return v
	}
	return phase
}
