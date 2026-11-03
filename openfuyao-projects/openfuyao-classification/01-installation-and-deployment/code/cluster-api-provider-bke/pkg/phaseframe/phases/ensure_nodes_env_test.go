/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/scriptshelper"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// createBKENodeWithFlags creates a BKENode with the specified flags set
func createBKENodeWithFlags(namespace, clusterName, ip, hostname string, roles []string, flags ...int) *confv1beta1.BKENode {
	stateCode := 0
	for _, flag := range flags {
		stateCode |= flag
	}

	return &confv1beta1.BKENode{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hostname,
			Namespace: namespace,
			Labels: map[string]string{
				nodeutil.ClusterNameLabel: clusterName,
			},
		},
		Spec: confv1beta1.BKENodeSpec{
			IP:       ip,
			Hostname: hostname,
			Role:     roles,
		},
		Status: confv1beta1.BKENodeStatus{
			StateCode: stateCode,
		},
	}
}

func TestEnsure_nodes_env_get_nodes_to_init_env(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create BKENode resources instead of NodesStatus
	bkeNode1 := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		bkev1beta1.NodeAgentReadyFlag,
	)
	bkeNode2 := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.2", "node2", []string{bkenode.WorkerNodeRole},
		bkev1beta1.NodeAgentReadyFlag, bkev1beta1.NodeEnvFlag,
	)

	// Create fake client with BKENode resources
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode1, bkeNode2).
		WithStatusSubresource(bkeNode1, bkeNode2).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	nodes := e.getNodesToInitEnv()
	// Only node1 should be returned (has AgentReady but not EnvFlag)
	require.Equal(t, 1, nodes.Length())
	require.Equal(t, "10.0.0.1", nodes[0].IP)
}

func TestEnsure_nodes_env_handle_success_nodes(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()

	// Create BKENode resource
	bkeNode1 := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "master1", []string{bkenode.MasterNodeRole},
	)

	// Create fake client with BKENode resources
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode1).
		WithStatusSubresource(bkeNode1).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	e.nodes = bkenode.Nodes{{IP: "10.0.0.1", Hostname: "master1", Role: []string{bkenode.MasterNodeRole}}}

	e.handleSuccessNodes([]string{"master1/10.0.0.1"})

	// Verify the flag was set by reading the BKENode
	updatedNode := &confv1beta1.BKENode{}
	err := fakeClient.Get(context.Background(), client.ObjectKey{
		Namespace: bkeCluster.Namespace,
		Name:      "master1",
	}, updatedNode)
	require.NoError(t, err)
	require.True(t, phaseutil.GetNodeStateFlag(updatedNode, "10.0.0.1", bkev1beta1.NodeEnvFlag))
	require.Equal(t, 1, e.nodes.Length())
}

func TestEnsure_nodes_env_check_preprocess_config_exists_global(t *testing.T) {
	InitinitPhaseContextFun()

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "preprocess-all-config",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"a.sh","order":1}]}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjects(t, cm)

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	ok := e.checkPreprocessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.True(t, ok)
}

func TestEnsure_nodes_env_check_preprocess_config_exists_batch(t *testing.T) {
	InitinitPhaseContextFun()

	mapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "preprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": `{"10.0.0.1":"001"}`,
		},
	}
	batch := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "preprocess-config-batch-001",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"b.sh","order":1}]}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjects(t, mapping, batch)

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	ok := e.checkPreprocessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	require.True(t, ok)
}

func TestEnsure_nodes_env_get_nodes_ips_by_script(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeCluster.Spec.ClusterConfig.CustomExtra = map[string]string{
		"pipelineServer":                  "10.0.0.9",
		"pipelineServerEnableCleanImages": "true",
		"host":                            "10.0.0.2",
	}
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	e.nodes = bkenode.Nodes{
		{IP: "10.0.0.1", Hostname: "node1", Role: []string{bkenode.MasterNodeRole}},
		{IP: "10.0.0.2", Hostname: "node2", Role: []string{bkenode.WorkerNodeRole}},
	}

	got, err := e.getNodesIpsByScript("install-nfsutils.sh")
	require.NoError(t, err)
	require.Equal(t, "10.0.0.9", got)

	got, err = e.getNodesIpsByScript("clean-docker-images.py")
	require.NoError(t, err)
	require.Equal(t, "10.0.0.9", got)

	got, err = e.getNodesIpsByScript("update-runc.sh")
	require.NoError(t, err)
	require.Equal(t, "10.0.0.1", got)
}

func TestEnsure_nodes_env_get_nodes_ips_by_script_errors(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeCluster.Spec.ClusterConfig.CustomExtra = map[string]string{}
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	e.nodes = bkenode.Nodes{
		{IP: "10.0.0.1", Hostname: "node1", Role: []string{bkenode.MasterNodeRole}},
	}

	_, err := e.getNodesIpsByScript("install-nfsutils.sh")
	require.Error(t, err)

	_, err = e.getNodesIpsByScript("clean-docker-images.py")
	require.Error(t, err)
}

func newFakeClientWithObjects(t *testing.T, objs ...client.Object) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

func TestEnsureNodesEnv_Constants(t *testing.T) {
	assert.Equal(t, "EnsureNodesEnv", string(EnsureNodesEnvName))
	assert.Len(t, defaultEnvExtraExecScripts, 7)
	assert.Len(t, commonEnvExtraExecScripts, 2)
}

func TestEnsureNodesEnv_NewEnsureNodesEnv(t *testing.T) {
	InitinitPhaseContextFun()
	phase := NewEnsureNodesEnv(initPhaseContext)
	assert.NotNil(t, phase)
	_, ok := phase.(*EnsureNodesEnv)
	assert.True(t, ok)
}

func TestEnsureNodesEnv_NeedExecute_DefaultNeedExecuteFalse(t *testing.T) {
	InitinitPhaseContextFun()

	now := metav1.Now()
	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeCluster.DeletionTimestamp = &now

	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func TestEnsureNodesEnv_NeedExecute_GetBKENodesError(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	// Mock to return error when getting BKENodes
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(ctx *phaseframe.PhaseContext, cluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
		return nil, false
	})

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func TestEnsureNodesEnv_NeedExecute_NoNodesNeedEnv(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	// Mock to return nodes without NodeEnvFlag
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(ctx *phaseframe.PhaseContext, cluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
		return bkev1beta1.BKENodes{}, true
	})

	patches.ApplyFunc(phaseutil.HasNodesNeedingPhase, func(nodes bkev1beta1.BKENodes, flag int) bool {
		return false
	})

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func TestEnsureNodesEnv_shouldUseDeepRestore_True(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeCluster.Annotations = map[string]string{"deepRestoreNode": "true"}
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	result := e.shouldUseDeepRestore(bkeCluster)
	assert.True(t, result)
}

func TestEnsureNodesEnv_shouldUseDeepRestore_False(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeCluster.Annotations = map[string]string{"bke.bocloud.com/deep-restore-node": "false"}
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	result := e.shouldUseDeepRestore(bkeCluster)
	assert.False(t, result)
}

func TestEnsureNodesEnv_shouldUseDeepRestore_NoAnnotation(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	result := e.shouldUseDeepRestore(bkeCluster)
	assert.True(t, result)
}

func TestBuildCommonEnvCommandParams_Structure(t *testing.T) {
	params := BuildCommonEnvCommandParams{
		Ctx:            context.Background(),
		Client:         &fakeClient{},
		BKECluster:     &bkev1beta1.BKECluster{},
		Scheme:         runtime.NewScheme(),
		ExceptEnvNodes: bkenode.Nodes{{IP: "10.0.0.1"}},
		Extra:          []string{"extra1"},
		ExtraHosts:     []string{"host1"},
		DryRun:         false,
		DeepRestore:    true,
		Log:            createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Scheme)
	assert.Len(t, params.ExceptEnvNodes, 1)
	assert.Len(t, params.Extra, 1)
	assert.Len(t, params.ExtraHosts, 1)
	assert.False(t, params.DryRun)
	assert.True(t, params.DeepRestore)
	assert.NotNil(t, params.Log)
}

func TestInstallScriptParams_Structure(t *testing.T) {
	params := InstallScriptParams{
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
		ScriptsLi:  []string{"script1", "script2"},
	}
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
	assert.Len(t, params.ScriptsLi, 2)
}

func TestInstallOtherScriptParams_Structure(t *testing.T) {
	params := InstallOtherScriptParams{
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
		ScriptsLi:  []string{"script1"},
		Cfg:        bkeinit.BkeConfig{},
	}
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
	assert.Len(t, params.ScriptsLi, 1)
}

func TestEnsureNodesEnv_createAddonTransfer(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	result := e.createAddonTransfer("test-script.sh", map[string]string{"key": "value"}, true)
	assert.NotNil(t, result)
	assert.Equal(t, "clusterextra", result.Addon.Name)
	assert.Equal(t, "test-script.sh", result.Addon.Version)
	assert.Equal(t, map[string]string{"key": "value"}, result.Addon.Param)
	assert.True(t, result.Addon.Block)
}

func TestEnsureNodesEnv_handleFileDownloaderScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	e.nodes = bkenode.Nodes{
		{IP: "10.0.0.1"},
		{IP: "10.0.0.2"},
	}

	result, err := e.handleFileDownloaderScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handlePackageDownloaderScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	e.nodes = bkenode.Nodes{
		{IP: "10.0.0.1"},
		{IP: "10.0.0.2"},
	}

	result, err := e.handlePackageDownloaderScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handleInstallLxcfsScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	result, err := e.handleInstallLxcfsScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handleInstallEtcdctlScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	result, err := e.handleInstallEtcdctlScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handleInstallHelmScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	result, err := e.handleInstallHelmScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handleInstallCalicoctlScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	result, err := e.handleInstallCalicoctlScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handleDefaultScript(t *testing.T) {
	InitinitPhaseContextFun()

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	result, err := e.handleDefaultScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_handleUpdateRuncScript_NoHostFilter(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeCluster.Spec.ClusterConfig.CustomExtra = map[string]string{}
	initPhaseContext.BKECluster = bkeCluster

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	result, err := e.handleUpdateRuncScript([]string{"10.0.0.1", "10.0.0.2"})
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.1,10.0.0.2", result)
}

func TestEnsureNodesEnv_checkPreprocessConfigExists_NodeConfig(t *testing.T) {
	InitinitPhaseContextFun()

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "preprocess-config-node-10.0.0.1",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"config.json": `{"scripts":[{"scriptName":"c.sh","order":1}]}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjects(t, cm)

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	ok := e.checkPreprocessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	assert.True(t, ok)
}

func TestEnsureNodesEnv_checkPreprocessConfigExists_NotFound(t *testing.T) {
	InitinitPhaseContextFun()

	initPhaseContext.Client = newFakeClientWithObjects(t)

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	ok := e.checkPreprocessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	assert.False(t, ok)
}

func TestEnsureNodesEnv_BuildCommonEnvCommand(t *testing.T) {
	InitinitPhaseContextFun()

	bkeCluster := initNewBkeCluster.DeepCopy()
	initPhaseContext.BKECluster = bkeCluster

	params := BuildCommonEnvCommandParams{
		Ctx:            context.Background(),
		Client:         &fakeClient{},
		BKECluster:     bkeCluster,
		Scheme:         runtime.NewScheme(),
		ExceptEnvNodes: bkenode.Nodes{{IP: "10.0.0.1"}},
		Extra:          []string{"extra1"},
		ExtraHosts:     []string{"host1"},
		DryRun:         false,
		DeepRestore:    false,
		Log:            initLog,
	}

	envCmd, err := BuildCommonEnvCommand(params)
	require.NoError(t, err)
	assert.NotNil(t, envCmd)
	assert.Equal(t, bkeCluster.Name, envCmd.BkeConfigName)
}

func TestEnsureNodesEnv_getNodesToInitEnv_NoNodes(t *testing.T) {
	InitinitPhaseContextFun()

	// Create BKENode with EnvFlag already set
	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeNode := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		bkev1beta1.NodeAgentReadyFlag, bkev1beta1.NodeEnvFlag,
	)

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode).
		WithStatusSubresource(bkeNode).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	nodes := e.getNodesToInitEnv()
	require.Equal(t, 0, nodes.Length())
}

func TestEnsureNodesEnv_getNodesToInitEnv_AgentNotReady(t *testing.T) {
	InitinitPhaseContextFun()

	// Create BKENode without AgentReadyFlag
	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeNode := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		// No NodeAgentReadyFlag
	)

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode).
		WithStatusSubresource(bkeNode).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	nodes := e.getNodesToInitEnv()
	require.Equal(t, 0, nodes.Length())
}

func TestEnsureNodesEnv_getNodesToInitEnv_NodeFailed(t *testing.T) {
	InitinitPhaseContextFun()

	// Create BKENode with Failed flag
	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeNode := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		bkev1beta1.NodeAgentReadyFlag, bkev1beta1.NodeFailedFlag,
	)

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode).
		WithStatusSubresource(bkeNode).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	nodes := e.getNodesToInitEnv()
	require.Equal(t, 0, nodes.Length())
}

func TestEnsureNodesEnv_getNodesToInitEnv_NodeDeleting(t *testing.T) {
	InitinitPhaseContextFun()

	// Create BKENode with Deleting flag
	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeNode := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		bkev1beta1.NodeAgentReadyFlag, bkev1beta1.NodeDeletingFlag,
	)

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode).
		WithStatusSubresource(bkeNode).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	nodes := e.getNodesToInitEnv()
	require.Equal(t, 0, nodes.Length())
}

func TestEnsureNodesEnv_getNodesToInitEnv_NodeNeedSkip(t *testing.T) {
	InitinitPhaseContextFun()

	// Create BKENode with NeedSkip
	bkeCluster := initNewBkeCluster.DeepCopy()
	bkeNode := createBKENodeWithFlags(
		bkeCluster.Namespace, bkeCluster.Name,
		"10.0.0.1", "node1", []string{bkenode.MasterNodeRole},
		bkev1beta1.NodeAgentReadyFlag,
	)
	bkeNode.Status.NeedSkip = true

	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bkeNode).
		WithStatusSubresource(bkeNode).
		Build()

	initPhaseContext.BKECluster = bkeCluster
	initPhaseContext.Client = fakeClient

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)

	nodes := e.getNodesToInitEnv()
	require.Equal(t, 0, nodes.Length())
}

func TestEnsureNodesEnv_checkPreprocessConfigExists_BatchMappingNotFound(t *testing.T) {
	InitinitPhaseContextFun()

	// Only create batch mapping CM, not the batch config CM
	mapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "preprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": `{"10.0.0.1":"001"}`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjects(t, mapping)

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	ok := e.checkPreprocessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	assert.False(t, ok)
}

func TestEnsureNodesEnv_checkPreprocessConfigExists_BatchMappingInvalid(t *testing.T) {
	InitinitPhaseContextFun()

	// Invalid mapping JSON
	mapping := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "preprocess-node-batch-mapping",
			Namespace: "user-system",
		},
		Data: map[string]string{
			"mapping.json": `invalid json`,
		},
	}
	initPhaseContext.Client = newFakeClientWithObjects(t, mapping)

	e := NewEnsureNodesEnv(initPhaseContext).(*EnsureNodesEnv)
	ok := e.checkPreprocessConfigExists(context.Background(), initPhaseContext.Client, initPhaseContext.Log, "10.0.0.1")
	assert.False(t, ok)
}

func newNodesEnvPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureNodesEnv {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	c := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureNodesEnv{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// stubRemoteClient for nodes_env script-install operations.
type nodesEnvStubRemoteClient struct {
	kube.RemoteKubeClient
	installErr error
}

func (s nodesEnvStubRemoteClient) InstallAddon(*bkev1beta1.BKECluster, *bkeaddon.AddonTransfer, *kube.AddonRecorder, client.Client, bkenode.Nodes) error {
	return s.installErr
}

// ---- setupClusterConditionAndSync ----

func TestEnsureNodesEnvSetupClusterConditionAndSync(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		require.NoError(t, e.setupClusterConditionAndSync())
	})

	t.Run("sync error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return assertErr("sync failed")
		})
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		err := e.setupClusterConditionAndSync()
		require.Error(t, err)
	})
}

// ---- getExtraAndExtraHosts ----

func TestEnsureNodesEnvGetExtraAndExtraHosts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1"}}}, nil
	})

	t.Run("HA vip adds extra + ingress", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ControlPlaneEndpoint: confv1beta1.APIEndpoint{Host: "192.168.99.99", Port: 6443},
				ClusterConfig:        &confv1beta1.BKEConfig{Addons: []confv1beta1.Product{{Name: "beyondELB", Param: map[string]string{"lbVIP": "192.168.99.100"}}}},
			},
		}
		e := newNodesEnvPhaseCov(t, cluster)
		extra, extraHosts := e.getExtraAndExtraHosts(cluster)
		assert.Contains(t, extra, "192.168.99.99")
		assert.NotEmpty(t, extraHosts)
		assert.Contains(t, extra, "192.168.99.100")
	})

	t.Run("non-HA no extra", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ControlPlaneEndpoint: confv1beta1.APIEndpoint{Host: "10.0.0.1", Port: 6443},
				ClusterConfig:        &confv1beta1.BKEConfig{},
			},
		}
		e := newNodesEnvPhaseCov(t, cluster)
		extra, extraHosts := e.getExtraAndExtraHosts(cluster)
		assert.Empty(t, extra)
		assert.Empty(t, extraHosts)
	})
}

// ---- buildEnvCommand ----

func TestEnsureNodesEnvBuildEnvCommand(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}}})
		patches.ApplyMethod(&command.ENV{}, "New", func(_ *command.ENV) error { return nil })
		envCmd, err := e.buildEnvCommand(bkenode.Nodes{{IP: "10.0.0.1"}})
		require.NoError(t, err)
		assert.NotNil(t, envCmd)
	})

	t.Run("new error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}}})
		patches.ApplyMethod(&command.ENV{}, "New", func(_ *command.ENV) error { return assertErr("new failed") })
		_, err := e.buildEnvCommand(bkenode.Nodes{{IP: "10.0.0.1"}})
		require.Error(t, err)
	})
}

// ---- executeEnvCommand ----

func TestEnsureNodesEnvExecuteEnvCommand(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
		return assertErr("wait failed"), []string{}, []string{"10.0.0.1"}
	})
	err, success, failed := e.executeEnvCommand(&command.ENV{})
	require.Error(t, err)
	assert.Contains(t, failed, "10.0.0.1")
	assert.Empty(t, success)
}

// ---- handleFailedNodes ----

func TestEnsureNodesEnvHandleFailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"worker"}}}}, nil
	})
	patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
		return nil
	})
	patches.ApplyFunc(phaseutil.LogCommandFailed, func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
		return nil, nil
	})
	patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
	envCmd := &command.ENV{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}
	// upstream change: non-empty failedNodes now returns an error.
	require.Error(t, e.handleFailedNodes(envCmd, []string{"node 10.0.0.1: fail"}))
}

// ---- finalDecisionAndCleanup ----

func TestEnsureNodesEnvFinalDecisionAndCleanup(t *testing.T) {
	t.Run("no success nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		_, err := e.finalDecisionAndCleanup(nil, []string{"10.0.0.1"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "all nodes")
	})

	t.Run("success path", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyPrivateMethod(e, "initClusterExtra", func(_ *EnsureNodesEnv) {})
		patches.ApplyPrivateMethod(e, "executeNodePreprocessScripts", func(_ *EnsureNodesEnv) error { return nil })
		_, err := e.finalDecisionAndCleanup([]string{"10.0.0.1"}, nil)
		require.NoError(t, err)
	})

	t.Run("sync error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return assertErr("sync failed")
		})
		_, err := e.finalDecisionAndCleanup([]string{"10.0.0.1"}, nil)
		require.Error(t, err)
	})
}

// ---- CheckOrInitNodesEnv + Execute ----

func TestEnsureNodesEnvCheckOrInitNodesEnv(t *testing.T) {
	t.Run("no nodes returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "getNodesToInitEnv", func(_ *EnsureNodesEnv) bkenode.Nodes { return nil })
		_, err := e.CheckOrInitNodesEnv()
		require.NoError(t, err)
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "getNodesToInitEnv", func(_ *EnsureNodesEnv) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		patches.ApplyPrivateMethod(e, "setupClusterConditionAndSync", func(_ *EnsureNodesEnv) error { return nil })
		patches.ApplyPrivateMethod(e, "buildEnvCommand", func(_ *EnsureNodesEnv, _ bkenode.Nodes) (*command.ENV, error) {
			return &command.ENV{}, nil
		})
		patches.ApplyPrivateMethod(e, "executeEnvCommand", func(_ *EnsureNodesEnv, _ *command.ENV) (error, []string, []string) {
			return nil, []string{"10.0.0.1"}, nil
		})
		patches.ApplyPrivateMethod(e, "handleSuccessNodes", func(_ *EnsureNodesEnv, _ []string) {})
		patches.ApplyPrivateMethod(e, "handleFailedNodes", func(_ *EnsureNodesEnv, _ *command.ENV, _ []string) error { return nil })
		patches.ApplyPrivateMethod(e, "finalDecisionAndCleanup", func(_ *EnsureNodesEnv, _, _ []string) (ctrl.Result, error) { return ctrl.Result{}, nil })
		_, err := e.CheckOrInitNodesEnv()
		require.NoError(t, err)
	})
}

func TestEnsureNodesEnvExecute(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
	patches.ApplyPrivateMethod(e, "CheckOrInitNodesEnv", func(_ *EnsureNodesEnv) (ctrl.Result, error) { return ctrl.Result{}, assertErr("check failed") })
	_, err := e.Execute()
	require.Error(t, err)
}

// ---- initClusterExtra ----

func TestEnsureNodesEnvInitClusterExtra(t *testing.T) {
	t.Run("new client error skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}}})
		patches.ApplyFunc(kube.NewClientFromRestConfig, func(_ context.Context, _ interface{}) (kube.RemoteKubeClient, error) {
			return nil, assertErr("client failed")
		})
		assert.NotPanics(t, func() { e.initClusterExtra() })
	})

	t.Run("success installs scripts", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}}})
		patches.ApplyFunc(kube.NewClientFromRestConfig, func(_ context.Context, _ interface{}) (kube.RemoteKubeClient, error) {
			return nodesEnvStubRemoteClient{}, nil
		})
		patches.ApplyFunc(scriptshelper.ListScriptsConfigMaps, func(_ client.Client) ([]string, error) {
			return commonEnvExtraExecScripts, nil
		})
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() { e.initClusterExtra() })
	})

	t.Run("list configmaps error skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}}})
		patches.ApplyFunc(kube.NewClientFromRestConfig, func(_ context.Context, _ interface{}) (kube.RemoteKubeClient, error) {
			return nodesEnvStubRemoteClient{}, nil
		})
		patches.ApplyFunc(scriptshelper.ListScriptsConfigMaps, func(_ client.Client) ([]string, error) {
			return nil, assertErr("list failed")
		})
		assert.NotPanics(t, func() { e.initClusterExtra() })
	})
}

// ---- installCommonScripts / installOtherCustomScripts (real body) ----

func TestEnsureNodesEnvInstallScripts(t *testing.T) {
	t.Run("common scripts installed", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() {
			e.installCommonScripts(InstallScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   commonEnvExtraExecScripts,
			})
		})
	})

	t.Run("common script not in list skips", func(t *testing.T) {
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		assert.NotPanics(t, func() {
			e.installCommonScripts(InstallScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   []string{},
			})
		})
	})

	t.Run("other custom scripts installed", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newNodesEnvPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}}})
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() {
			e.installOtherCustomScripts(InstallOtherScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   defaultEnvExtraExecScripts,
				Cfg:         bkeinit.BkeConfig(*e.Ctx.BKECluster.Spec.ClusterConfig),
			})
		})
	})
}

func nodesEnvGaps2Phase(t *testing.T) *EnsureNodesEnv {
	t.Helper()
	cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	ctx := newAdditionalPhaseContext(t, cluster)
	return &EnsureNodesEnv{BasePhase: phaseframe.NewBasePhase(ctx, EnsureNodesEnvName)}
}

// TestNodesEnvGaps2ExecuteNodePreprocessScripts covers the previously uncovered
// executeNodePreprocessScripts across all of its branches.
func TestNodesEnvGaps2ExecuteNodePreprocessScripts(t *testing.T) {
	patchHasConfig := func(patches *gomonkey.Patches, e *EnsureNodesEnv, hasConfig bool) {
		patches.ApplyPrivateMethod(e, "checkPreprocessConfigExists",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return hasConfig
			})
	}
	patchCreateCmd := func(patches *gomonkey.Patches, e *EnsureNodesEnv, cmd *command.Custom, err error) {
		patches.ApplyPrivateMethod(e, "createPreprocessCommand",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ *runtime.Scheme, _ bkenode.Nodes) (*command.Custom, error) {
				return cmd, err
			})
	}
	patchWait := func(patches *gomonkey.Patches, waitErr error, successNodes, failedNodes []string) {
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return waitErr, successNodes, failedNodes
		})
	}
	patchLogInfo := func(patches *gomonkey.Patches) {
		patches.ApplyFunc(phaseutil.LogCommandInfo, func(_ agentv1beta1.Command, _ *bkev1beta1.BKELogger, _ string) {})
	}

	t.Run("empty_ip_skipped_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGaps2Phase(t)
		e.nodes = bkenode.Nodes{{IP: ""}}
		require.NoError(t, e.executeNodePreprocessScripts())
	})

	t.Run("no_config_all_skipped_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGaps2Phase(t)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
		patchHasConfig(patches, e, false)
		require.NoError(t, e.executeNodePreprocessScripts())
	})

	t.Run("create_command_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGaps2Phase(t)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patchHasConfig(patches, e, true)
		patchCreateCmd(patches, e, nil, errors.New("create failed"))
		err := e.executeNodePreprocessScripts()
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to create preprocess Command resource")
	})

	t.Run("wait_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGaps2Phase(t)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patchHasConfig(patches, e, true)
		patchCreateCmd(patches, e, &command.Custom{}, nil)
		patchWait(patches, errors.New("wait failed"), nil, []string{"10.0.0.1"})
		patchLogInfo(patches)
		err := e.executeNodePreprocessScripts()
		require.Error(t, err)
		require.Contains(t, err.Error(), "preprocess execution failed")
	})

	t.Run("wait_failed_nodes_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGaps2Phase(t)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patchHasConfig(patches, e, true)
		patchCreateCmd(patches, e, &command.Custom{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}, nil)
		patchWait(patches, nil, nil, []string{"10.0.0.1"})
		patchLogInfo(patches)
		err := e.executeNodePreprocessScripts()
		require.Error(t, err)
		require.Contains(t, err.Error(), "preprocess execution failed")
	})

	t.Run("success_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGaps2Phase(t)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patchHasConfig(patches, e, true)
		patchCreateCmd(patches, e, &command.Custom{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}, nil)
		patchWait(patches, nil, []string{"10.0.0.1"}, nil)
		patchLogInfo(patches)
		require.NoError(t, e.executeNodePreprocessScripts())
	})
}

// nodesEnvGapsPhase builds an EnsureNodesEnv backed by a full PhaseContext.
func nodesEnvGapsPhase(t *testing.T, cluster *bkev1beta1.BKECluster) *EnsureNodesEnv {
	t.Helper()
	if cluster == nil {
		cluster = &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	}
	if cluster.Spec.ClusterConfig == nil {
		cluster.Spec.ClusterConfig = &confv1beta1.BKEConfig{}
	}
	ctx := newAdditionalPhaseContext(t, cluster)
	return &EnsureNodesEnv{BasePhase: phaseframe.NewBasePhase(ctx, EnsureNodesEnvName)}
}

// ---- executeNodePreprocessScripts (0%) ----

func TestNodesEnvGapsExecuteNodePreprocessScripts(t *testing.T) {
	t.Run("all_no_config_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
		patches.ApplyPrivateMethod(e, "checkPreprocessConfigExists",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return false
			})
		require.NoError(t, e.executeNodePreprocessScripts())
	})

	t.Run("empty_ip_skipped", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: ""}}
		require.NoError(t, e.executeNodePreprocessScripts())
	})

	t.Run("create_command_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "checkPreprocessConfigExists",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return true
			})
		patches.ApplyMethod(&command.Custom{}, "New", func(_ *command.Custom) error {
			return assertErr("new failed")
		})
		err := e.executeNodePreprocessScripts()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create preprocess Command resource")
	})

	t.Run("wait_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "checkPreprocessConfigExists",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return true
			})
		patches.ApplyMethod(&command.Custom{}, "New", func(_ *command.Custom) error { return nil })
		patches.ApplyMethod(&command.Custom{}, "Wait", func(_ *command.Custom) (error, []string, []string) {
			return assertErr("wait failed"), nil, nil
		})
		err := e.executeNodePreprocessScripts()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "preprocess execution failed")
	})

	t.Run("wait_failed_nodes_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "checkPreprocessConfigExists",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return true
			})
		patches.ApplyMethod(&command.Custom{}, "New", func(_ *command.Custom) error { return nil })
		patches.ApplyMethod(&command.Custom{}, "Wait", func(_ *command.Custom) (error, []string, []string) {
			return nil, nil, []string{"10.0.0.1"}
		})
		err := e.executeNodePreprocessScripts()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "preprocess execution failed")
	})

	t.Run("success_returns_nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "checkPreprocessConfigExists",
			func(_ *EnsureNodesEnv, _ context.Context, _ client.Client, _ *bkev1beta1.BKELogger, _ string) bool {
				return true
			})
		// Set Command non-nil so LogCommandInfo branch is covered.
		patches.ApplyMethod(&command.Custom{}, "New", func(c *command.Custom) error {
			c.Command = &agentv1beta1.Command{}
			return nil
		})
		patches.ApplyMethod(&command.Custom{}, "Wait", func(_ *command.Custom) (error, []string, []string) {
			return nil, []string{"10.0.0.1"}, nil
		})
		require.NoError(t, e.executeNodePreprocessScripts())
	})
}

// ---- handleFailedNodes uncovered branches (68.4%) ----

func TestNodesEnvGapsHandleFailedNodes(t *testing.T) {
	t.Run("get_nodes_error_still_processed", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return nil, assertErr("get nodes failed")
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
				return nil
			})
		patches.ApplyFunc(phaseutil.LogCommandFailed,
			func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
		envCmd := &command.ENV{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}
		require.Error(t, e.handleFailedNodes(envCmd, []string{"10.0.0.1"}))
	})

	t.Run("update_status_error_logged", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"worker"}}}}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
				return assertErr("update failed")
			})
		patches.ApplyFunc(phaseutil.LogCommandFailed,
			func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
		envCmd := &command.ENV{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}
		require.Error(t, e.handleFailedNodes(envCmd, []string{"10.0.0.1"}))
	})

	t.Run("empty_failed_nodes_no_error_log", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1"}}}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
				return nil
			})
		patches.ApplyFunc(phaseutil.LogCommandFailed,
			func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
		envCmd := &command.ENV{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}
		require.NoError(t, e.handleFailedNodes(envCmd, nil))
	})

	t.Run("worker_node_callback_sets_need_skip", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		var capturedNeedSkip bool
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"node"}}}}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, cb func(*confv1beta1.BKENodeStatus)) error {
				status := &confv1beta1.BKENodeStatus{}
				cb(status)
				capturedNeedSkip = status.NeedSkip
				return nil
			})
		patches.ApplyFunc(phaseutil.LogCommandFailed,
			func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
		envCmd := &command.ENV{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}
		require.Error(t, e.handleFailedNodes(envCmd, []string{"10.0.0.1"}))
		assert.True(t, capturedNeedSkip)
	})

	t.Run("non_worker_node_callback_no_skip", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		var capturedNeedSkip bool
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
				return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"master"}}}}, nil
			})
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, cb func(*confv1beta1.BKENodeStatus)) error {
				status := &confv1beta1.BKENodeStatus{}
				cb(status)
				capturedNeedSkip = status.NeedSkip
				return nil
			})
		patches.ApplyFunc(phaseutil.LogCommandFailed,
			func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
				return nil, nil
			})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
			func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})
		envCmd := &command.ENV{BaseCommand: command.BaseCommand{Command: &agentv1beta1.Command{}}}
		require.Error(t, e.handleFailedNodes(envCmd, []string{"10.0.0.1"}))
		assert.False(t, capturedNeedSkip)
	})
}

// ---- finalDecisionAndCleanup uncovered branches (71.4%) ----

func TestNodesEnvGapsFinalDecisionAndCleanup(t *testing.T) {
	t.Run("preprocess_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyPrivateMethod(e, "initClusterExtra", func(_ *EnsureNodesEnv) {})
		patches.ApplyPrivateMethod(e, "executeNodePreprocessScripts", func(_ *EnsureNodesEnv) error {
			return assertErr("preprocess failed")
		})
		_, err := e.finalDecisionAndCleanup([]string{"10.0.0.1"}, nil)
		require.Error(t, err)
	})

	t.Run("deploying_state_with_failed_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}},
			Status: confv1beta1.BKEClusterStatus{
				ClusterStatus: bkev1beta1.ClusterInitializing,
			},
		}
		e := nodesEnvGapsPhase(t, cluster)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
			func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyPrivateMethod(e, "initClusterExtra", func(_ *EnsureNodesEnv) {})
		patches.ApplyPrivateMethod(e, "executeNodePreprocessScripts", func(_ *EnsureNodesEnv) error { return nil })
		patches.ApplyFunc(phaseutil.GetNotSkipFailedNode,
			func(_ *bkev1beta1.BKECluster, _ []string) int { return 1 })
		_, err := e.finalDecisionAndCleanup([]string{"10.0.0.1"}, []string{"10.0.0.2"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Deploying")
	})
}

// ---- CheckOrInitNodesEnv uncovered branches (69.6%) ----

func TestNodesEnvGapsCheckOrInitNodesEnv(t *testing.T) {
	t.Run("setup_cluster_condition_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyPrivateMethod(e, "getNodesToInitEnv", func(_ *EnsureNodesEnv) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		patches.ApplyPrivateMethod(e, "setupClusterConditionAndSync", func(_ *EnsureNodesEnv) error {
			return assertErr("setup failed")
		})
		_, err := e.CheckOrInitNodesEnv()
		require.Error(t, err)
	})

	t.Run("build_env_command_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyPrivateMethod(e, "getNodesToInitEnv", func(_ *EnsureNodesEnv) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		patches.ApplyPrivateMethod(e, "setupClusterConditionAndSync", func(_ *EnsureNodesEnv) error { return nil })
		patches.ApplyPrivateMethod(e, "buildEnvCommand", func(_ *EnsureNodesEnv, _ bkenode.Nodes) (*command.ENV, error) {
			return nil, assertErr("build failed")
		})
		_, err := e.CheckOrInitNodesEnv()
		require.Error(t, err)
	})

	t.Run("execute_env_command_error_returns_err", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyPrivateMethod(e, "getNodesToInitEnv", func(_ *EnsureNodesEnv) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		patches.ApplyPrivateMethod(e, "setupClusterConditionAndSync", func(_ *EnsureNodesEnv) error { return nil })
		patches.ApplyPrivateMethod(e, "buildEnvCommand", func(_ *EnsureNodesEnv, _ bkenode.Nodes) (*command.ENV, error) {
			return &command.ENV{}, nil
		})
		patches.ApplyPrivateMethod(e, "executeEnvCommand", func(_ *EnsureNodesEnv, _ *command.ENV) (error, []string, []string) {
			return assertErr("exec failed"), nil, nil
		})
		_, err := e.CheckOrInitNodesEnv()
		require.Error(t, err)
	})

	t.Run("handle_failed_nodes_error_still_returns_final", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		patches.ApplyPrivateMethod(e, "getNodesToInitEnv", func(_ *EnsureNodesEnv) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		patches.ApplyPrivateMethod(e, "setupClusterConditionAndSync", func(_ *EnsureNodesEnv) error { return nil })
		patches.ApplyPrivateMethod(e, "buildEnvCommand", func(_ *EnsureNodesEnv, _ bkenode.Nodes) (*command.ENV, error) {
			return &command.ENV{}, nil
		})
		patches.ApplyPrivateMethod(e, "executeEnvCommand", func(_ *EnsureNodesEnv, _ *command.ENV) (error, []string, []string) {
			return nil, []string{"10.0.0.1"}, []string{"10.0.0.2"}
		})
		patches.ApplyPrivateMethod(e, "handleSuccessNodes", func(_ *EnsureNodesEnv, _ []string) {})
		patches.ApplyPrivateMethod(e, "handleFailedNodes", func(_ *EnsureNodesEnv, _ *command.ENV, _ []string) error {
			return assertErr("handle failed")
		})
		patches.ApplyPrivateMethod(e, "finalDecisionAndCleanup", func(_ *EnsureNodesEnv, _, _ []string) (ctrl.Result, error) {
			return ctrl.Result{}, nil
		})
		_, err := e.CheckOrInitNodesEnv()
		require.NoError(t, err)
	})
}

// ---- installCommonScripts uncovered branches (64.7%) ----

func TestNodesEnvGapsInstallCommonScripts(t *testing.T) {
	t.Run("get_nodes_ips_error_skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "", assertErr("get ips failed")
		})
		assert.NotPanics(t, func() {
			e.installCommonScripts(InstallScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   commonEnvExtraExecScripts,
			})
		})
	})

	t.Run("empty_nodes_ips_skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "", nil
		})
		assert.NotPanics(t, func() {
			e.installCommonScripts(InstallScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   commonEnvExtraExecScripts,
			})
		})
	})

	t.Run("install_addon_error_skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() {
			e.installCommonScripts(InstallScriptParams{
				LocalClient: nodesEnvStubRemoteClient{installErr: assertErr("install failed")},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   commonEnvExtraExecScripts,
			})
		})
	})
}

// ---- installOtherCustomScripts uncovered branches (65.4%) ----

func TestNodesEnvGapsInstallOtherCustomScripts(t *testing.T) {
	t.Run("custom_scripts_from_cfg", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					CustomExtra: map[string]string{"envExtraExecScripts": "install-lxcfs.sh"},
				},
			},
		}
		e := nodesEnvGapsPhase(t, cluster)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() {
			e.installOtherCustomScripts(InstallOtherScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   []string{"install-lxcfs.sh"},
				Cfg:         bkeinit.BkeConfig(*e.Ctx.BKECluster.Spec.ClusterConfig),
			})
		})
	})

	t.Run("update_runc_containerd_skipped", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec: confv1beta1.BKEClusterSpec{
				ClusterConfig: &confv1beta1.BKEConfig{
					Cluster: confv1beta1.Cluster{
						ContainerRuntime: confv1beta1.ContainerRuntime{CRI: bkeinit.CRIContainerd},
					},
				},
			},
		}
		e := nodesEnvGapsPhase(t, cluster)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() {
			e.installOtherCustomScripts(InstallOtherScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   defaultEnvExtraExecScripts,
				Cfg:         bkeinit.BkeConfig(*e.Ctx.BKECluster.Spec.ClusterConfig),
			})
		})
	})

	t.Run("get_nodes_ips_error_skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "", assertErr("get ips failed")
		})
		assert.NotPanics(t, func() {
			e.installOtherCustomScripts(InstallOtherScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   defaultEnvExtraExecScripts,
				Cfg:         bkeinit.BkeConfig(*e.Ctx.BKECluster.Spec.ClusterConfig),
			})
		})
	})

	t.Run("empty_nodes_ips_skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "", nil
		})
		assert.NotPanics(t, func() {
			e.installOtherCustomScripts(InstallOtherScriptParams{
				LocalClient: nodesEnvStubRemoteClient{},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   defaultEnvExtraExecScripts,
				Cfg:         bkeinit.BkeConfig(*e.Ctx.BKECluster.Spec.ClusterConfig),
			})
		})
	})

	t.Run("install_addon_error_skips", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := nodesEnvGapsPhase(t, nil)
		e.nodes = bkenode.Nodes{{IP: "10.0.0.1"}}
		patches.ApplyPrivateMethod(e, "getNodesIpsByScript", func(_ *EnsureNodesEnv, _ string) (string, error) {
			return "10.0.0.1", nil
		})
		assert.NotPanics(t, func() {
			e.installOtherCustomScripts(InstallOtherScriptParams{
				LocalClient: nodesEnvStubRemoteClient{installErr: assertErr("install failed")},
				BKECluster:  e.Ctx.BKECluster,
				Log:         e.Ctx.Log,
				ScriptsLi:   defaultEnvExtraExecScripts,
				Cfg:         bkeinit.BkeConfig(*e.Ctx.BKECluster.Spec.ClusterConfig),
			})
		})
	})
}
