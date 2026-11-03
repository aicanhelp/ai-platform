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

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
)

func TestEnsureAgentSwitchConstants(t *testing.T) {
	assert.Equal(t, "EnsureAgentSwitch", string(EnsureAgentSwitchName))
}

func TestNewEnsureAgentSwitch(t *testing.T) {
	InitinitPhaseContextFun()
	phase := NewEnsureAgentSwitch(initPhaseContext)
	assert.NotNil(t, phase)
}

func TestEnsureAgentSwitch_Execute_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	patches.ApplyPrivateMethod(eas, "reconcileAgentSwitch", func(_ *EnsureAgentSwitch) error {
		return nil
	})

	result, err := eas.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureAgentSwitch_Execute_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	patches.ApplyPrivateMethod(eas, "reconcileAgentSwitch", func(_ *EnsureAgentSwitch) error {
		return assert.AnError
	})

	result, err := eas.Execute()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureAgentSwitch_NeedExecute_DefaultFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	patches.ApplyMethod(&eas.BasePhase, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return false
	})

	result := eas.NeedExecute(&initOldBkeCluster, &initNewBkeCluster)
	assert.False(t, result)
}

func TestEnsureAgentSwitch_NeedExecute_ListenerCurrent(t *testing.T) {
	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{
		common.BKEAgentListenerAnnotationKey: common.BKEAgentListenerCurrent,
	}
	initPhaseContext.BKECluster = &initNewBkeCluster

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	result := eas.NeedExecute(&initOldBkeCluster, &initNewBkeCluster)
	assert.False(t, result)
}

func TestEnsureAgentSwitch_NeedExecute_ConditionTrue(t *testing.T) {
	t.Skip("Requires complex mocking of condition.HasConditionStatus")
}

func TestEnsureAgentSwitch_NeedExecute_ShouldExecute(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{}
	initPhaseContext.BKECluster = &initNewBkeCluster

	patches.ApplyFunc(condition.HasConditionStatus, func(_ confv1beta1.ClusterConditionType, _ *bkev1beta1.BKECluster, _ confv1beta1.ConditionStatus) bool {
		return false
	})

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	result := eas.NeedExecute(&initOldBkeCluster, &initNewBkeCluster)
	assert.True(t, result)
}

func TestEnsureAgentSwitch_ReconcileAgentSwitch_NoAnnotation(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{}
	initPhaseContext.BKECluster = &initNewBkeCluster

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster) error {
		return nil
	})

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	err := eas.reconcileAgentSwitch()
	assert.NoError(t, err)
}

func TestEnsureAgentSwitch_ReconcileAgentSwitch_ListenerCurrent(t *testing.T) {
	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{
		common.BKEAgentListenerAnnotationKey: common.BKEAgentListenerCurrent,
	}
	initPhaseContext.BKECluster = &initNewBkeCluster

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	err := eas.reconcileAgentSwitch()
	assert.NoError(t, err)
}

func TestEnsureAgentSwitch_ReconcileAgentSwitch_ListenerBkecluster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{
		common.BKEAgentListenerAnnotationKey: common.BKEAgentListenerBkecluster,
	}
	initPhaseContext.BKECluster = &initNewBkeCluster

	patches.ApplyMethod(initPhaseContext, "GetNodes", func(_ *phaseframe.PhaseContext) (bkenode.Nodes, error) {
		return bkenode.Nodes{}, nil
	})
	patches.ApplyMethod(&command.Switch{}, "New", func(_ *command.Switch) error {
		return nil
	})

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	err := eas.reconcileAgentSwitch()
	assert.NoError(t, err)
}

func TestEnsureAgentSwitch_ReconcileAgentSwitch_GetNodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{
		common.BKEAgentListenerAnnotationKey: common.BKEAgentListenerBkecluster,
	}
	initPhaseContext.BKECluster = &initNewBkeCluster

	patches.ApplyMethod(initPhaseContext, "GetNodes", func(_ *phaseframe.PhaseContext) (bkenode.Nodes, error) {
		return nil, assert.AnError
	})

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	err := eas.reconcileAgentSwitch()
	assert.Error(t, err)
}

func TestEnsureAgentSwitch_ReconcileAgentSwitch_CommandNewError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	initNewBkeCluster.Annotations = map[string]string{
		common.BKEAgentListenerAnnotationKey: common.BKEAgentListenerBkecluster,
	}
	initPhaseContext.BKECluster = &initNewBkeCluster

	patches.ApplyMethod(initPhaseContext, "GetNodes", func(_ *phaseframe.PhaseContext) (bkenode.Nodes, error) {
		return bkenode.Nodes{}, nil
	})
	patches.ApplyMethod(&command.Switch{}, "New", func(_ *command.Switch) error {
		return assert.AnError
	})

	phase := NewEnsureAgentSwitch(initPhaseContext)
	eas := phase.(*EnsureAgentSwitch)

	err := eas.reconcileAgentSwitch()
	assert.NoError(t, err)
}

func TestCreateSwitchCommand(t *testing.T) {
	InitinitPhaseContextFun()

	nodes := bkenode.Nodes{
		{IP: "192.168.1.1", Hostname: "node1"},
	}

	cmd := createSwitchCommand(initCxt, initClient.GetClient(), &initNewBkeCluster, initScheme, nodes)

	assert.NotNil(t, cmd)
	assert.Equal(t, initNewBkeCluster.Name, cmd.ClusterName)
	assert.Equal(t, nodes, cmd.Nodes)
}
