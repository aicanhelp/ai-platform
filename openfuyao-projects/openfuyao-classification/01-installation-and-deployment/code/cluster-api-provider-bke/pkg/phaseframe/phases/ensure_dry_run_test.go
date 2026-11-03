/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
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
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/remote"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

// RFC 5737 documentation addresses and RFC 2606 reserved names for unit tests (SCA-safe fixtures).
const (
	testDryRunNodeIP1          = "203.0.113.10"
	testDryRunNodeIP2          = "203.0.113.11"
	testDryRunControlPlaneHost = "203.0.113.100"
	testDryRunNTPServer        = "ntp.invalid"
)

func newDryRunClusterWithNTP() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{NTPServer: testDryRunNTPServer},
			},
		},
	}
}

func createTestEnsureDryRun() *EnsureDryRun {
	logger := createTestLogger()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName,
			Namespace: testNamespace,
		},
		Spec: confv1beta1.BKEClusterSpec{
			DryRun: true,
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					NTPServer: testDryRunNTPServer,
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Log:        logger,
	}

	return &EnsureDryRun{
		BasePhase: phaseframe.NewBasePhase(ctx, EnsureDryRunName),
	}
}

func TestNewEnsureDryRun(t *testing.T) {
	logger := createTestLogger()
	ctx := &phaseframe.PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: testNamespace},
		},
		Scheme: runtime.NewScheme(),
		Log:    logger,
	}

	phase := NewEnsureDryRun(ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureDryRun{}, phase)
}

func TestEnsureDryRun_NeedExecute_DeletionTimestamp(t *testing.T) {
	e := createTestEnsureDryRun()
	now := metav1.Now()
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			DeletionTimestamp: &now,
		},
		Spec: confv1beta1.BKEClusterSpec{
			DryRun: true,
		},
	}

	result := e.NeedExecute(old, new)
	assert.False(t, result)
}

func TestEnsureDryRun_NeedExecute_Paused(t *testing.T) {
	e := createTestEnsureDryRun()
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			Pause:  true,
			DryRun: true,
		},
	}

	result := e.NeedExecute(old, new)
	assert.False(t, result)
}

func TestEnsureDryRun_NeedExecute_DryRunDisabled(t *testing.T) {
	e := createTestEnsureDryRun()
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			DryRun: false,
		},
	}

	result := e.NeedExecute(old, new)
	assert.False(t, result)
}

func TestEnsureDryRun_NeedExecute_Success(t *testing.T) {
	e := createTestEnsureDryRun()
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			DryRun: true,
		},
	}

	result := e.NeedExecute(old, new)
	assert.True(t, result)
}

func TestEnsureDryRun_HandleDryRunDisabled_WithAnnotation(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annotation.BKEClusterDryRunAnnotationKey: testDryRunNodeIP1 + ",",
			},
		},
	}

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return nil
	})

	err := e.handleDryRunDisabled(&fakeClient{}, bkeCluster, bkeCluster.Annotations, createTestLogger())
	assert.NoError(t, err)
	assert.NotContains(t, bkeCluster.Annotations, annotation.BKEClusterDryRunAnnotationKey)
}

func TestEnsureDryRun_HandleDryRunDisabled_WithoutAnnotation(t *testing.T) {
	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
	}

	err := e.handleDryRunDisabled(&fakeClient{}, bkeCluster, bkeCluster.Annotations, createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_HandleDryRunDisabled_SyncError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annotation.BKEClusterDryRunAnnotationKey: testDryRunNodeIP1 + ",",
			},
		},
	}

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return assert.AnError
	})

	err := e.handleDryRunDisabled(&fakeClient{}, bkeCluster, bkeCluster.Annotations, createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_GetDryRunNodes_NoAnnotation(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{}

	testNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: testDryRunNodeIP1}},
		{Spec: confv1beta1.BKENodeSpec{IP: testDryRunNodeIP2}},
	}

	expectedNodes := bkenode.Nodes{
		{IP: testDryRunNodeIP1},
		{IP: testDryRunNodeIP2},
	}

	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
		return testNodes, nil
	})

	patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(cluster *bkev1beta1.BKECluster, nodes bkev1beta1.BKENodes) bkenode.Nodes {
		return expectedNodes
	})

	nodes, err := e.getDryRunNodes(bkeCluster, map[string]string{}, createTestLogger())
	assert.NoError(t, err)
	assert.Len(t, nodes, 2)
}

func TestEnsureDryRun_GetDryRunNodes_WithAnnotation(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{}

	testNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: testDryRunNodeIP1}},
		{Spec: confv1beta1.BKENodeSpec{IP: testDryRunNodeIP2}},
	}

	expectedNodes := bkenode.Nodes{
		{IP: testDryRunNodeIP1},
		{IP: testDryRunNodeIP2},
	}

	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
		return testNodes, nil
	})

	patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(cluster *bkev1beta1.BKECluster, nodes bkev1beta1.BKENodes) bkenode.Nodes {
		return expectedNodes
	})

	annotations := map[string]string{
		annotation.BKEClusterDryRunAnnotationKey: testDryRunNodeIP1 + ",",
	}

	nodes, err := e.getDryRunNodes(bkeCluster, annotations, createTestLogger())
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
	assert.Equal(t, testDryRunNodeIP2, nodes[0].IP)
}

func TestEnsureDryRun_GetDryRunNodes_IPPrefixBoundary(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{}

	const longIP = "122.235.189.16"
	const shortIP = "122.235.189.1"

	testNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: longIP}},
		{Spec: confv1beta1.BKENodeSpec{IP: shortIP}},
	}
	expectedNodes := bkenode.Nodes{
		{IP: longIP},
		{IP: shortIP},
	}

	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapperForCluster, func(_ *nodeutil.NodeFetcher, ctx context.Context, cluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, error) {
		return testNodes, nil
	})
	patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(cluster *bkev1beta1.BKECluster, nodes bkev1beta1.BKENodes) bkenode.Nodes {
		return expectedNodes
	})

	annotations := map[string]string{
		annotation.BKEClusterDryRunAnnotationKey: longIP + ",",
	}

	nodes, err := e.getDryRunNodes(bkeCluster, annotations, createTestLogger())
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
	assert.Equal(t, shortIP, nodes[0].IP)
}

func TestEnsureDryRun_UpdateDryRunAnnotation_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{}
	nodes := bkenode.Nodes{
		{IP: testDryRunNodeIP1},
		{IP: testDryRunNodeIP2},
	}

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return nil
	})

	err := e.updateDryRunAnnotation(&fakeClient{}, bkeCluster, nodes, map[string]string{}, createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_UpdateDryRunAnnotation_SyncError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{}
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return assert.AnError
	})

	err := e.updateDryRunAnnotation(&fakeClient{}, bkeCluster, nodes, map[string]string{}, createTestLogger())
	assert.Error(t, err)
}

func TestEnsureDryRun_UpdateDryRunAnnotation_NilAnnotations(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	bkeCluster := &bkev1beta1.BKECluster{}
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return nil
	})

	err := e.updateDryRunAnnotation(&fakeClient{}, bkeCluster, nodes, nil, createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_PushAgentWithParams_SecretNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1, Hostname: "node1"}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		return apierrors.NewNotFound(schema.GroupResource{}, "secret")
	})

	params := PushAgentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.pushAgentWithParams(params)
	assert.NoError(t, err)
}

func TestEnsureDryRun_PushAgentWithParams_GetSecretError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1, Hostname: "node1"}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		return assert.AnError
	})

	params := PushAgentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.pushAgentWithParams(params)
	assert.NoError(t, err)
}

func TestEnsureDryRun_PushAgentWithParams_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1, Hostname: "node1"}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if secret, ok := obj.(*corev1.Secret); ok {
			secret.Data = map[string][]byte{"config": []byte("test-config")}
		}
		return nil
	})

	patches.ApplyFunc(phaseutil.NodeToRemoteHost, func(nodes bkenode.Nodes) []remote.Host {
		return []remote.Host{{Address: testDryRunNodeIP1}}
	})

	patches.ApplyFunc(phaseutil.PushAgent, func(hosts []remote.Host, kubeconfig []byte, ntpServer string) []string {
		return []string{}
	})

	params := PushAgentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: newDryRunClusterWithNTP(),
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.pushAgentWithParams(params)
	assert.NoError(t, err)
}

func TestEnsureDryRun_PushAgentWithParams_UsesArchAwareSSHFlow(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1, Hostname: "node1"}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if secret, ok := obj.(*corev1.Secret); ok {
			secret.Data = map[string][]byte{"config": []byte("test-config")}
		}
		return nil
	})

	servicePrepared := false
	sshPushed := false
	patches.ApplyPrivateMethod(&EnsureBKEAgent{}, "prepareServiceFile", func(_ *EnsureBKEAgent, _ *bkev1beta1.BKECluster) (string, error) {
		servicePrepared = true
		return filepath.Join(t.TempDir(), "bkeagent.service"), nil
	})
	patches.ApplyPrivateMethod(&EnsureBKEAgent{}, "sshPushAgent", func(_ *EnsureBKEAgent, _ context.Context, _ []remote.Host, kubeconfig []byte, servicePath string) (map[string]error, error) {
		sshPushed = true
		assert.Equal(t, []byte("test-config"), kubeconfig)
		assert.True(t, strings.HasSuffix(servicePath, "bkeagent.service"))
		return map[string]error{}, nil
	})

	params := PushAgentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: newDryRunClusterWithNTP(),
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.pushAgentWithParams(params)
	assert.NoError(t, err)
	assert.True(t, servicePrepared)
	assert.True(t, sshPushed)
}

func TestEnsureDryRun_PushAgentWithParams_PushFailed(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1, Hostname: "node1"}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if secret, ok := obj.(*corev1.Secret); ok {
			secret.Data = map[string][]byte{"config": []byte("test-config")}
		}
		return nil
	})

	patches.ApplyFunc(phaseutil.NodeToRemoteHost, func(nodes bkenode.Nodes) []remote.Host {
		return []remote.Host{{Address: testDryRunNodeIP1}}
	})

	patches.ApplyFunc(phaseutil.PushAgent, func(hosts []remote.Host, kubeconfig []byte, ntpServer string) []string {
		return []string{testDryRunNodeIP1}
	})

	params := PushAgentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: newDryRunClusterWithNTP(),
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.pushAgentWithParams(params)
	assert.NoError(t, err)
}

func TestEnsureDryRun_CheckBKEAgentStatus_PingError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()

	patches.ApplyFunc(phaseutil.PingBKEAgent, func(ctx context.Context, c client.Client, scheme *runtime.Scheme, bkeCluster *bkev1beta1.BKECluster) (error, []string, []string) {
		return assert.AnError, nil, nil
	})

	err := e.checkBKEAgentStatus(createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_CheckBKEAgentStatus_FailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()

	patches.ApplyFunc(phaseutil.PingBKEAgent, func(ctx context.Context, c client.Client, scheme *runtime.Scheme, bkeCluster *bkev1beta1.BKECluster) (error, []string, []string) {
		return nil, []string{}, []string{testDryRunNodeIP1}
	})

	err := e.checkBKEAgentStatus(createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_CheckBKEAgentStatus_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()

	patches.ApplyFunc(phaseutil.PingBKEAgent, func(ctx context.Context, c client.Client, scheme *runtime.Scheme, bkeCluster *bkev1beta1.BKECluster) (error, []string, []string) {
		return nil, []string{testDryRunNodeIP1}, []string{}
	})

	err := e.checkBKEAgentStatus(createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRun_PingBKEAgent(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()

	patches.ApplyFunc(phaseutil.PingBKEAgent, func(ctx context.Context, c client.Client, scheme *runtime.Scheme, bkeCluster *bkev1beta1.BKECluster) (error, []string, []string) {
		return nil, []string{"node1"}, []string{}
	})

	err, success, failed := e.pingBKEAgent()
	assert.NoError(t, err)
	assert.Len(t, success, 1)
	assert.Empty(t, failed)
}

func TestEnsureDryRun_CheckNodeEnvironmentWithParams_NewCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
	})

	patches.ApplyFunc(clusterutil.AvailableLoadBalancerEndPoint, func(endpoint confv1beta1.APIEndpoint, nodes []confv1beta1.Node) bool {
		return false
	})

	patches.ApplyMethod(&command.ENV{}, "New", func(_ *command.ENV) error {
		return assert.AnError
	})

	params := CheckNodeEnvironmentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Scheme:     runtime.NewScheme(),
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.checkNodeEnvironmentWithParams(params)
	assert.Error(t, err)
}

func TestEnsureDryRun_CheckNodeEnvironmentWithParams_WaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
	})

	patches.ApplyMethod(&command.ENV{}, "New", func(_ *command.ENV) error {
		return nil
	})

	patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
		return assert.AnError, nil, nil
	})

	params := CheckNodeEnvironmentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Scheme:     runtime.NewScheme(),
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.checkNodeEnvironmentWithParams(params)
	assert.Error(t, err)
}

func TestEnsureDryRun_CheckNodeEnvironmentWithParams_FailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{}}, nil
	})

	patches.ApplyMethod(&command.ENV{}, "New", func(_ *command.ENV) error {
		return nil
	})

	patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
		return nil, []string{}, []string{testDryRunNodeIP1}
	})

	params := CheckNodeEnvironmentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Scheme:     runtime.NewScheme(),
		Nodes:      nodes,
		Log:        createTestLogger(),
	}

	err := e.checkNodeEnvironmentWithParams(params)
	assert.NoError(t, err)
}

func TestEnsureDryRun_CheckNodeEnvironmentWithParams_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: testDryRunNodeIP1}}}, nil
	})

	patches.ApplyFunc(clusterutil.AvailableLoadBalancerEndPoint, func(endpoint confv1beta1.APIEndpoint, nodes []confv1beta1.Node) bool {
		return true
	})

	patches.ApplyMethod(&command.ENV{}, "New", func(_ *command.ENV) error {
		return nil
	})

	patches.ApplyMethod(&command.ENV{}, "Wait", func(_ *command.ENV) (error, []string, []string) {
		return nil, []string{testDryRunNodeIP1}, []string{}
	})

	params := CheckNodeEnvironmentParams{
		Ctx:    context.Background(),
		Client: &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			Spec: confv1beta1.BKEClusterSpec{
				ControlPlaneEndpoint: confv1beta1.APIEndpoint{
					Host: testDryRunControlPlaneHost,
				},
			},
		},
		Scheme: runtime.NewScheme(),
		Nodes:  nodes,
		Log:    createTestLogger(),
	}

	err := e.checkNodeEnvironmentWithParams(params)
	assert.NoError(t, err)
}

func TestEnsureDryRun_ReconcileDryRun_DryRunDisabled(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	e.Ctx.BKECluster.Spec.DryRun = false

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return nil
	})

	err := e.reconcileDryRun()
	assert.NoError(t, err)
}

func TestEnsureDryRun_ReconcileDryRun_NoNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()

	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
		return bkev1beta1.BKENodes{}, nil
	})

	patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(cluster *bkev1beta1.BKECluster, nodes bkev1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{}
	})

	err := e.reconcileDryRun()
	assert.NoError(t, err)
}

func TestEnsureDryRun_ReconcileDryRun_UpdateAnnotationError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()

	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
		return bkev1beta1.BKENodes{{Spec: confv1beta1.BKENodeSpec{IP: testDryRunNodeIP1}}}, nil
	})

	patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(cluster *bkev1beta1.BKECluster, nodes bkev1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{{IP: testDryRunNodeIP1}}
	})

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(c client.Client, bkeCluster *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		return assert.AnError
	})

	err := e.reconcileDryRun()
	assert.Error(t, err)
}

func TestEnsureDryRun_PushAgent(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureDryRun()
	nodes := bkenode.Nodes{{IP: testDryRunNodeIP1}}

	patches.ApplyMethod(&fakeClient{}, "Get", func(_ *fakeClient, ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if secret, ok := obj.(*corev1.Secret); ok {
			secret.Data = map[string][]byte{"config": []byte("test")}
		}
		return nil
	})

	patches.ApplyFunc(phaseutil.NodeToRemoteHost, func(nodes bkenode.Nodes) []remote.Host {
		return []remote.Host{{Address: testDryRunNodeIP1}}
	})

	patches.ApplyFunc(phaseutil.PushAgent, func(hosts []remote.Host, kubeconfig []byte, ntpServer string) []string {
		return []string{}
	})

	err := e.pushAgent(context.Background(), &fakeClient{}, newDryRunClusterWithNTP(), nodes, createTestLogger())
	assert.NoError(t, err)
}

func TestEnsureDryRunName_Constant(t *testing.T) {
	assert.Equal(t, confv1beta1.BKEClusterPhase("EnsureDryRun"), EnsureDryRunName)
}

func TestPushAgentParams_Structure(t *testing.T) {
	params := PushAgentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Nodes:      bkenode.Nodes{{IP: testDryRunNodeIP1}},
		Log:        createTestLogger(),
	}

	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.Len(t, params.Nodes, 1)
	assert.NotNil(t, params.Log)
}

func TestCheckNodeEnvironmentParams_Structure(t *testing.T) {
	params := CheckNodeEnvironmentParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Scheme:     runtime.NewScheme(),
		Nodes:      bkenode.Nodes{{IP: testDryRunNodeIP1}},
		Log:        createTestLogger(),
	}

	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Scheme)
	assert.Len(t, params.Nodes, 1)
	assert.NotNil(t, params.Log)
}
