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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCommonPhases(t *testing.T) {
	assert.NotNil(t, CommonPhases())
	assert.Equal(t, 5, len(CommonPhases()))
}

func TestDeployPhases(t *testing.T) {
	assert.NotNil(t, DeployPhases())
	assert.Equal(t, 11, len(DeployPhases()))
}

func TestPostDeployPhases(t *testing.T) {
	assert.NotNil(t, PostDeployPhases())
	assert.Equal(t, 13, len(PostDeployPhases()))
}

func TestDeletePhases(t *testing.T) {
	assert.NotNil(t, DeletePhases())
	assert.Equal(t, 2, len(DeletePhases()))
}

func TestClusterInitPhaseNames(t *testing.T) {
	phases := ClusterInitPhaseNames()
	assert.NotNil(t, phases)
	assert.Equal(t, 10, len(phases))
	assert.Contains(t, phases, EnsureFinalizerName)
	assert.Contains(t, phases, EnsureMasterInitName)
	assert.Contains(t, phases, EnsureNodesPostProcessName)
	assert.Contains(t, phases, EnsureClusterAPIManagerManifestName)
}

func TestClusterUpgradePhaseNames(t *testing.T) {
	phases := ClusterUpgradePhaseNames()
	assert.NotNil(t, phases)
	assert.Equal(t, 7, len(phases))
	assert.Contains(t, phases, EnsureMasterUpgradeName)
	assert.Contains(t, phases, EnsureWorkerUpgradeName)
}

func TestDeclarativeClusterUpgradePhaseNames(t *testing.T) {
	phases := DeclarativeClusterUpgradePhaseNames()
	assert.NotNil(t, phases)
	assert.Equal(t, 7, len(phases))
	assert.Contains(t, phases, EnsurePreUpgradeResourcesName)
	assert.Contains(t, phases, EnsureAgentUpgradeName)
	assert.Contains(t, phases, EnsureEtcdUpgradeName)
	assert.Contains(t, phases, EnsureContainerdUpgradeName)
	assert.Contains(t, phases, EnsureProviderSelfUpgradeName)
}

func TestIsDeclarativeInlineUpgradePhase(t *testing.T) {
	assert.True(t, IsDeclarativeInlineUpgradePhase(EnsureEtcdUpgradeName))
	assert.True(t, IsDeclarativeInlineUpgradePhase(EnsureAgentUpgradeName))
	assert.True(t, IsDeclarativeInlineUpgradePhase(EnsureProviderSelfUpgradeName))
	assert.False(t, IsDeclarativeInlineUpgradePhase(EnsureClusterName))
	assert.False(t, IsDeclarativeInlineUpgradePhase(EnsureAddonDeployName))
}

func TestClusterScaleMasterDownPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterScaleMasterDownPhaseNames())
	assert.Equal(t, 1, len(ClusterScaleMasterDownPhaseNames()))
	assert.Contains(t, ClusterScaleMasterDownPhaseNames(), EnsureMasterDeleteName)
}

func TestClusterScaleMasterUpPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterScaleMasterUpPhaseNames())
	assert.Equal(t, 1, len(ClusterScaleMasterUpPhaseNames()))
	assert.Contains(t, ClusterScaleMasterUpPhaseNames(), EnsureMasterJoinName)
}

func TestClusterScaleWorkerDownPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterScaleWorkerDownPhaseNames())
	assert.Equal(t, 1, len(ClusterScaleWorkerDownPhaseNames()))
	assert.Contains(t, ClusterScaleWorkerDownPhaseNames(), EnsureWorkerDeleteName)
}

func TestClusterScaleWorkerUpPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterScaleWorkerUpPhaseNames())
	assert.Equal(t, 1, len(ClusterScaleWorkerUpPhaseNames()))
	assert.Contains(t, ClusterScaleWorkerUpPhaseNames(), EnsureWorkerJoinName)
}

func TestClusterAddonsPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterAddonsPhaseNames())
	assert.Equal(t, 1, len(ClusterAddonsPhaseNames()))
	assert.Contains(t, ClusterAddonsPhaseNames(), EnsureAddonDeployName)
}

func TestClusterDeletePhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterDeletePhaseNames())
	assert.Equal(t, 1, len(ClusterDeletePhaseNames()))
	assert.Contains(t, ClusterDeletePhaseNames(), EnsureDeleteOrResetName)
}

func TestClusterDryRunPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterDryRunPhaseNames())
	assert.Equal(t, 1, len(ClusterDryRunPhaseNames()))
	assert.Contains(t, ClusterDryRunPhaseNames(), EnsureDryRunName)
}

func TestClusterPausedPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterPausedPhaseNames())
	assert.Equal(t, 1, len(ClusterPausedPhaseNames()))
	assert.Contains(t, ClusterPausedPhaseNames(), EnsurePausedName)
}

func TestClusterManagePhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterManagePhaseNames())
	assert.Equal(t, 1, len(ClusterManagePhaseNames()))
	assert.Contains(t, ClusterManagePhaseNames(), EnsureClusterManageName)
}

func TestCustomSetStatusPhaseNames(t *testing.T) {
	assert.NotNil(t, CustomSetStatusPhaseNames())
	assert.Equal(t, 1, len(CustomSetStatusPhaseNames()))
	assert.Contains(t, CustomSetStatusPhaseNames(), EnsureClusterName)
}

func TestClusterDeleteResetPhaseNames(t *testing.T) {
	assert.NotNil(t, ClusterDeleteResetPhaseNames())
	assert.Equal(t, 2, len(ClusterDeleteResetPhaseNames()))
	assert.Contains(t, ClusterDeleteResetPhaseNames(), EnsurePausedName)
	assert.Contains(t, ClusterDeleteResetPhaseNames(), EnsureDeleteOrResetName)
}

func TestPhaseNameCNMap(t *testing.T) {
	phaseNames := phaseNameCNMap()
	assert.NotNil(t, phaseNames)
	assert.Greater(t, len(phaseNames), 0)

	assert.Equal(t, "部署任务创建", phaseNames[EnsureFinalizerName])
	assert.Equal(t, "集群管理暂停", phaseNames[EnsurePausedName])
	assert.Equal(t, "集群删除", phaseNames[EnsureDeleteOrResetName])
	assert.Equal(t, "Master初始化", phaseNames[EnsureMasterInitName])
	assert.Equal(t, "Worker加入", phaseNames[EnsureWorkerJoinName])
	assert.Equal(t, "升级前资源预创建", phaseNames[EnsurePreUpgradeResourcesName])
}

func TestConvertPhaseNameToCN_ExistingPhase(t *testing.T) {
	result := ConvertPhaseNameToCN(string(EnsureFinalizerName))
	assert.Equal(t, "部署任务创建", result)

	result = ConvertPhaseNameToCN(string(EnsureMasterInitName))
	assert.Equal(t, "Master初始化", result)

	result = ConvertPhaseNameToCN(string(EnsureWorkerJoinName))
	assert.Equal(t, "Worker加入", result)
}

func TestConvertPhaseNameToCN_NonExistingPhase(t *testing.T) {
	unknownPhase := "UnknownPhase"
	result := ConvertPhaseNameToCN(unknownPhase)
	assert.Equal(t, unknownPhase, result)
}

func TestConvertPhaseNameToCN_EmptyString(t *testing.T) {
	result := ConvertPhaseNameToCN("")
	assert.Equal(t, "", result)
}

func TestConvertPhaseNameToCN_AllMappedPhases(t *testing.T) {
	for phase, cnName := range phaseNameCNMap() {
		result := ConvertPhaseNameToCN(string(phase))
		assert.Equal(t, cnName, result)
	}
}
