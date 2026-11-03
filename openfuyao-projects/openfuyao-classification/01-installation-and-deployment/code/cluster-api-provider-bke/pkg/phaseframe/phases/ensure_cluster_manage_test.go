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
	"context"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/certs"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	nodeutil "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	controlv1beta1 "sigs.k8s.io/cluster-api/controlplane/kubeadm/api/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureClusterManageConstants(t *testing.T) {
	assert.Equal(t, "EnsureClusterManage", string(EnsureClusterManageName))
}

func TestNewEnsureClusterManage(t *testing.T) {
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
		Scheme:     runtime.NewScheme(),
	}
	phase := NewEnsureClusterManage(ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureClusterManage{}, phase)
}

func TestEnsureClusterManage_NeedExecute(t *testing.T) {
	t.Skip("Requires complex mocking of DefaultNeedExecute")
}

func TestEnsureClusterManage_Execute_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
	}
	e := NewEnsureClusterManage(ctx).(*EnsureClusterManage)
	patches.ApplyPrivateMethod(e, "collectBaseInfo", func(_ *EnsureClusterManage) error {
		return assert.AnError
	})
	result, err := e.Execute()
	assert.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureClusterManage_CheckAgentNeedPush(t *testing.T) {
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
	}
	e := NewEnsureClusterManage(ctx).(*EnsureClusterManage)
	// Empty node list: no nodes to push, so no push needed.
	result := e.checkAgentNeedPush(bkenode.Nodes{})
	assert.False(t, result)
}

func TestEnsureClusterManage_DistributeMasterNodesCerts(t *testing.T) {
	t.Skip("Requires complex setup with remote client")
}

func TestEnsureClusterManage_DistributeWorkerNodesCerts(t *testing.T) {
	t.Skip("Requires complex setup with remote client")
}

func TestEnsureClusterManage_BackupBocloudClusterData(t *testing.T) {
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
	}
	e := NewEnsureClusterManage(ctx).(*EnsureClusterManage)
	err := e.backupBocloudClusterData(bkenode.Nodes{})
	assert.Error(t, err)
}

func TestCreateCertCommandSpec(t *testing.T) {
	params := CreateCertCommandSpecParams{
		CertPluginName:  "test",
		ClusterName:     "test",
		Namespace:       "default",
		CertificatesDir: "/etc/kubernetes/pki",
	}
	spec := createCertCommandSpec(params)
	assert.NotNil(t, spec)
}

func TestNewBaseCommandParams(t *testing.T) {
	params := newBaseCommandParams(context.Background(), &fakeClient{}, &bkev1beta1.BKECluster{}, runtime.NewScheme())
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.OwnerObj)
	assert.NotNil(t, params.Scheme)
}

func cmScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, agentv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, clusterv1.AddToScheme(scheme))
	return scheme
}

func newClusterManageCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster, objs ...client.Object) *EnsureClusterManage {
	t.Helper()
	scheme := cmScheme(t)
	builder := ctrlfake.NewClientBuilder().WithScheme(scheme)
	if bkeCluster != nil {
		builder = builder.WithObjects(bkeCluster)
	}
	for _, o := range objs {
		builder = builder.WithObjects(o)
	}
	c := builder.Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureClusterManage{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

const (
	cmClusterFromKey     = "bke.bocloud.com/cluster-from"
	cmClusterFromBocloud = "bocloud"
	cmClusterFromBKE     = "bke"
	cmCollectedAnnoKey   = "bke.bocloud.com/collectd"
)

func cmBocloudCluster() *bkev1beta1.BKECluster {
	c := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "cm-c1", Namespace: "ns", UID: "cm-uid"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					KubernetesVersion: "v1.30.0",
					CertificatesDir:   "/etc/kubernetes/pki",
					ContainerRuntime:  confv1beta1.ContainerRuntime{CRI: "containerd"},
					ImageRepo:         confv1beta1.Repo{Domain: "registry.k8s.io", Port: "443"},
					Kubelet:           &confv1beta1.Kubelet{},
				},
			},
		},
	}
	annotation.SetAnnotation(c, cmClusterFromKey, cmClusterFromBocloud)
	return c
}

func cmMarkBaseCollected(c *bkev1beta1.BKECluster) {
	annotation.SetAnnotation(c, cmCollectedAnnoKey, "base")
}
func cmMarkAgentCollected(c *bkev1beta1.BKECluster) {
	annotation.SetAnnotation(c, cmCollectedAnnoKey, "agent")
}

func cmSyncCallingPatch() func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
	return func(_ client.Client, bc *bkev1beta1.BKECluster, pfs ...mergecluster.PatchFunc) error {
		for _, pf := range pfs {
			if pf != nil {
				pf(bc)
			}
		}
		return nil
	}
}

func cmPatchSyncNoop(patches *gomonkey.Patches) {
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
		return nil
	})
}

func cmPatchGetNodes(patches *gomonkey.Patches, nodes bkenode.Nodes, err error) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: nodes}, err
	})
}

type cmFakeRemoteClient struct {
	kube.RemoteKubeClient
	collectResult *kube.CollectResult
	collectErrs   []error
	collectWarns  []error
}

func (f *cmFakeRemoteClient) Collect() (*kube.CollectResult, []error, []error) {
	return f.collectResult, f.collectWarns, f.collectErrs
}

func (f *cmFakeRemoteClient) KubeClient() (*kubernetes.Clientset, dynamic.Interface) { return nil, nil }

func TestClusterManageGetContainerRuntimeConfigFromCollectCommand(t *testing.T) {
	makeCollectCmd := func(stdout []string) *agentv1beta1.Command {
		return &agentv1beta1.Command{
			Status: map[string]*agentv1beta1.CommandStatus{
				"node1": {Conditions: []*agentv1beta1.Condition{{StdOut: stdout}}},
			},
		}
	}

	t.Run("sets ExtraVolumes when kubelet has none", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cluster.Spec.ClusterConfig.Cluster.Kubelet = &confv1beta1.Kubelet{}
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, cmSyncCallingPatch())
		cmd := makeCollectCmd([]string{"containerd", "systemd", "/var/lib/containerd", "bocloud", "/var/lib/kubelet"})
		require.NoError(t, e.getContainerRuntimeConfigFromCollectCommand(cmd))
		assert.True(t, clusterutil.ClusterAgentInfoHasCollected(cluster))
		require.NotNil(t, cluster.Spec.ClusterConfig.Cluster.Kubelet)
		require.Len(t, cluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes, 1)
		assert.Equal(t, "kubelet-root-dir", cluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes[0].Name)
	})

	t.Run("updates existing kubelet-root-dir ExtraVolumes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cluster.Spec.ClusterConfig.Cluster.Kubelet = &confv1beta1.Kubelet{}
		cluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes = []confv1beta1.HostPathMount{
			{Name: "kubelet-root-dir", HostPath: "/old/dir"},
			{Name: "other", HostPath: "/other"},
		}
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, cmSyncCallingPatch())
		cmd := makeCollectCmd([]string{"docker", "cgroupfs", "/var/lib/docker", "bke", "/custom/kubelet"})
		require.NoError(t, e.getContainerRuntimeConfigFromCollectCommand(cmd))
		require.Len(t, cluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes, 2)
		assert.Equal(t, "/custom/kubelet", cluster.Spec.ClusterConfig.Cluster.Kubelet.ExtraVolumes[0].HostPath)
	})

	t.Run("handles empty stdout entries", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, cmSyncCallingPatch())
		cmd := makeCollectCmd([]string{"", "", "", "", ""})
		require.NoError(t, e.getContainerRuntimeConfigFromCollectCommand(cmd))
	})

	t.Run("sync error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return assertErr("sync failed")
		})
		cmd := makeCollectCmd([]string{"containerd", "systemd", "/data", "bocloud", "/kubelet"})
		err := e.getContainerRuntimeConfigFromCollectCommand(cmd)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync failed")
	})
}

func TestClusterManageWaitForNodesBootstrap(t *testing.T) {
	t.Run("nil successNodes map returns error", func(t *testing.T) {
		err := waitForNodesBootstrap(nodesBootstrapInput{
			ctx: context.Background(), cluster: cmBocloudCluster(),
			nodes: bkenode.Nodes{{IP: "1.1.1.1"}}, expectedCount: 1, log: createTestLogger(),
		})
		require.Error(t, err)
	})

	t.Run("all nodes bootstrapped returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return &clusterv1.Machine{Status: clusterv1.MachineStatus{NodeRef: &corev1.ObjectReference{Name: "n1"}}}, nil
		})
		success := map[int]confv1beta1.Node{}
		err := waitForNodesBootstrap(nodesBootstrapInput{
			ctx: context.Background(), cluster: cluster, nodes: bkenode.Nodes{{IP: "1.1.1.1"}},
			successNodes: success, expectedCount: 1, log: createTestLogger(),
		})
		require.NoError(t, err)
		assert.Len(t, success, 1)
	})

	t.Run("not all ready returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			return &clusterv1.Machine{Status: clusterv1.MachineStatus{}}, nil
		})
		err := waitForNodesBootstrap(nodesBootstrapInput{
			ctx: context.Background(), cluster: cluster,
			nodes:        bkenode.Nodes{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			successNodes: map[int]confv1beta1.Node{}, expectedCount: 2, log: createTestLogger(),
		})
		require.Error(t, err)
	})

	t.Run("NodeToMachine error skips node", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, n confv1beta1.Node) (*clusterv1.Machine, error) {
			if n.IP == "1.1.1.1" {
				return nil, assertErr("not found")
			}
			return &clusterv1.Machine{Status: clusterv1.MachineStatus{NodeRef: &corev1.ObjectReference{Name: "n2"}}}, nil
		})
		err := waitForNodesBootstrap(nodesBootstrapInput{
			ctx: context.Background(), cluster: cluster,
			nodes:        bkenode.Nodes{{IP: "1.1.1.1"}, {IP: "2.2.2.2"}},
			successNodes: map[int]confv1beta1.Node{}, expectedCount: 1, log: createTestLogger(),
		})
		require.NoError(t, err)
	})

	t.Run("already processed node skipped", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		called := false
		patches.ApplyFunc(phaseutil.NodeToMachine, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ confv1beta1.Node) (*clusterv1.Machine, error) {
			called = true
			return &clusterv1.Machine{}, nil
		})
		success := map[int]confv1beta1.Node{0: {IP: "1.1.1.1"}}
		err := waitForNodesBootstrap(nodesBootstrapInput{
			ctx: context.Background(), cluster: cluster, nodes: bkenode.Nodes{{IP: "1.1.1.1"}},
			successNodes: success, expectedCount: 1, log: createTestLogger(),
		})
		require.NoError(t, err)
		assert.False(t, called)
	})
}

func TestClusterManageCreateCustomCommand(t *testing.T) {
	params := CreateCustomCommandParams{
		BaseCommand:  command.BaseCommand{ClusterName: "c1"},
		Nodes:        bkenode.Nodes{{IP: "1.1.1.1", Hostname: "n1"}},
		CommandName:  "cert-cmd",
		CommandSpec:  command.GenerateDefaultCommandSpec(),
		CommandLabel: command.BKEClusterLabel,
	}
	c := createCustomCommand(params)
	assert.Equal(t, "cert-cmd", c.CommandName)
	assert.Equal(t, command.BKEClusterLabel, c.CommandLabel)
	assert.Len(t, c.Nodes, 1)
}

func TestClusterManageExecuteCommandAndWait(t *testing.T) {
	t.Run("New error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(_ *command.Custom) error { return assertErr("new failed") })
		params := ExecuteCommandAndWaitParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: cluster, Log: e.Ctx.Log,
			Command: command.Custom{CommandName: "x"},
		}
		err := executeCommandAndWait(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "new failed")
	})

	t.Run("Wait error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return assertErr("wait failed"), nil, nil
		})
		params := ExecuteCommandAndWaitParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: cluster, Log: e.Ctx.Log,
			Command: command.Custom{CommandName: "x"},
		}
		err := executeCommandAndWait(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait failed")
	})

	t.Run("failed nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return nil, nil, []string{"node1"}
		})
		patches.ApplyFunc(phaseutil.LogCommandFailed, func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
			return map[string][]string{"1.1.1.1": {"err"}}, nil
		})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs, func(context.Context, client.Client, *bkev1beta1.BKECluster, map[string][]string) {})
		params := ExecuteCommandAndWaitParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: cluster, Log: e.Ctx.Log,
			Command:       command.Custom{CommandName: "x"},
			ConditionType: bkev1beta1.BocloudClusterMasterCertDistributionCondition,
		}
		err := executeCommandAndWait(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "node1")
	})

	t.Run("success marks condition", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return nil, []string{"node1"}, nil
		})
		params := ExecuteCommandAndWaitParams{
			Ctx: context.Background(), Client: e.Ctx.Client, BKECluster: cluster, Log: e.Ctx.Log,
			Command:       command.Custom{CommandName: "x"},
			ConditionType: bkev1beta1.BocloudClusterMasterCertDistributionCondition,
			SuccessReason: "ok", FailedReason: "bad", SuccessMessage: "done",
		}
		require.NoError(t, executeCommandAndWait(params))
	})
}

func TestClusterManageProcessAgentPingResults(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	cluster := cmBocloudCluster()
	e := newClusterManageCov(t, cluster)
	patches.ApplyFunc(phaseutil.GetNodeIPFromCommandWaitResult, func(result string) string { return result })
	updateCalled := 0
	patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
		updateCalled++
		return nil
	})
	assert.NotPanics(t, func() {
		e.processAgentPingResults(context.Background(), cluster, []string{"10.0.0.1"}, []string{"10.0.0.2"}, e.Ctx.Log)
	})
	assert.Equal(t, 2, updateCalled)
}

func TestClusterManageMarkNodesBootstrapSuccess(t *testing.T) {
	t.Run("updates nodes and swallows error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
			return assertErr("update failed")
		})
		nodes := map[int]confv1beta1.Node{0: {IP: "10.0.0.1"}, 1: {IP: "10.0.0.2"}}
		assert.NotPanics(t, func() {
			e.markNodesBootstrapSuccess(context.Background(), nodes, e.Ctx.Log)
		})
	})

	t.Run("empty nodes no-op", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		assert.NotPanics(t, func() {
			e.markNodesBootstrapSuccess(context.Background(), map[int]confv1beta1.Node{}, e.Ctx.Log)
		})
	})
}

func TestClusterManageWaitAgentPushedFlagVisible(t *testing.T) {
	t.Run("GetBootTimeOut error logged, wait still invoked", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		nf := e.Ctx.NodeFetcher()
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) {
			return 0, assertErr("timeout err")
		})
		waitCalled := false
		patches.ApplyFunc(phaseutil.WaitNodesStateFlagVisible, func(_ phaseutil.WaitNodesStateFlagVisibleInput) error {
			waitCalled = true
			return nil
		})
		err := e.waitAgentPushedFlagVisible(context.Background(), nf, cluster, bkenode.Nodes{{IP: "10.0.0.1"}}, e.Ctx.Log)
		require.NoError(t, err)
		assert.True(t, waitCalled)
	})

	t.Run("wait error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		nf := e.Ctx.NodeFetcher()
		patches.ApplyFunc(phaseutil.GetBootTimeOut, func(_ *bkev1beta1.BKECluster) (time.Duration, error) { return time.Minute, nil })
		patches.ApplyFunc(phaseutil.WaitNodesStateFlagVisible, func(_ phaseutil.WaitNodesStateFlagVisibleInput) error {
			return assertErr("not visible")
		})
		err := e.waitAgentPushedFlagVisible(context.Background(), nf, cluster, bkenode.Nodes{{IP: "10.0.0.1"}}, e.Ctx.Log)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not visible")
	})
}

func TestClusterManageCollectBaseInfo(t *testing.T) {
	t.Run("already collected returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		cmMarkBaseCollected(cluster)
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.collectBaseInfo())
	})

	t.Run("collect errors returns aggregated error", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.remoteClient = &cmFakeRemoteClient{collectErrs: []error{assertErr("collect failed")}}
		err := e.collectBaseInfo()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "collect failed")
	})

	t.Run("collect success with warnings and nodes", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.remoteClient = &cmFakeRemoteClient{
			collectResult: &kube.CollectResult{
				EtcdCertificatesDir:  "/etc/etcd/ssl",
				Networking:           confv1beta1.Networking{ServiceSubnet: "10.96.0.0/12"},
				ControlPlaneEndpoint: confv1beta1.APIEndpoint{Host: "1.1.1.1", Port: 6443},
				KubernetesVersion:    "v1.30.0",
				ContainerRuntime:     confv1beta1.ContainerRuntime{CRI: "containerd"},
				Nodes:                bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}},
			},
			collectWarns: []error{assertErr("warn1")},
		}
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, cmSyncCallingPatch())
		require.NoError(t, e.collectBaseInfo())
		assert.Equal(t, "v1.30.0", cluster.Status.KubernetesVersion)
	})
}

func TestClusterManageCollectAgentInfo(t *testing.T) {
	t.Run("agent info already collected returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		cmMarkAgentCollected(cluster)
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.collectAgentInfo())
	})

	t.Run("base info not collected returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.collectAgentInfo())
	})

	t.Run("no master nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cmMarkBaseCollected(cluster)
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"worker"}}}, nil)
		err := e.collectAgentInfo()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no master nodes")
	})

	t.Run("no nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cmMarkBaseCollected(cluster)
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{}, nil)
		err := e.collectAgentInfo()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no BKENode resources")
	})

	t.Run("collect command wait error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cmMarkBaseCollected(cluster)
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyFunc((*command.Collect).New, func(c *command.Collect) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Collect).Wait, func(_ *command.Collect) (error, []string, []string) {
			return assertErr("wait failed"), nil, nil
		})
		err := e.collectAgentInfo()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait failed")
	})

	t.Run("collect command failed nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cmMarkBaseCollected(cluster)
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyFunc((*command.Collect).New, func(c *command.Collect) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Collect).Wait, func(_ *command.Collect) (error, []string, []string) {
			return nil, nil, []string{"m1"}
		})
		patches.ApplyFunc(phaseutil.LogCommandFailed, func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
			return nil, nil
		})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs, func(context.Context, client.Client, *bkev1beta1.BKECluster, map[string][]string) {})
		err := e.collectAgentInfo()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "m1")
	})

	t.Run("collect success invokes getContainerRuntimeConfig", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cmMarkBaseCollected(cluster)
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyFunc((*command.Collect).New, func(c *command.Collect) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Collect).Wait, func(_ *command.Collect) (error, []string, []string) {
			return nil, []string{"m1"}, nil
		})
		patches.ApplyPrivateMethod(e, "getContainerRuntimeConfigFromCollectCommand", func(_ *EnsureClusterManage, _ *agentv1beta1.Command) error { return nil })
		require.NoError(t, e.collectAgentInfo())
	})
}

func TestClusterManageUpdateKubeadmControlPlaneReplicas(t *testing.T) {
	t.Run("GetKCP error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyFunc(phaseutil.GetClusterAPIKubeadmControlPlane, func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*controlv1beta1.KubeadmControlPlane, error) {
			return nil, assertErr("kcp not found")
		})
		err := e.updateKubeadmControlPlaneReplicas(context.Background(), e.Ctx.Client, int32(3))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "kcp not found")
	})

	t.Run("success updates replicas", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		kcp := &controlv1beta1.KubeadmControlPlane{Spec: controlv1beta1.KubeadmControlPlaneSpec{Replicas: int32p(1)}}
		patches.ApplyFunc(phaseutil.GetClusterAPIKubeadmControlPlane, func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*controlv1beta1.KubeadmControlPlane, error) {
			return kcp, nil
		})
		called := false
		patches.ApplyFunc(phaseutil.ResumeAndUpdateKubeadmControlPlaneReplicas, func(_ context.Context, _ client.Client, _ *controlv1beta1.KubeadmControlPlane, replicas int32) error {
			called = true
			assert.Equal(t, int32(3), replicas)
			return nil
		})
		require.NoError(t, e.updateKubeadmControlPlaneReplicas(context.Background(), e.Ctx.Client, int32(3)))
		assert.True(t, called)
	})
}

func TestClusterManageWaitForClusterInfrastructureReady(t *testing.T) {
	t.Run("sync error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error {
			return assertErr("sync failed")
		})
		err := e.waitForClusterInfrastructureReady(context.Background(), e.Ctx.Client, cluster)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync failed")
	})

	t.Run("infrastructure ready returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{Status: clusterv1.ClusterStatus{InfrastructureReady: true}}
		cmPatchSyncNoop(patches)
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		require.NoError(t, e.waitForClusterInfrastructureReady(context.Background(), e.Ctx.Client, cluster))
	})

	t.Run("refresh error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		cmPatchSyncNoop(patches)
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error {
			return assertErr("refresh failed")
		})
		err := e.waitForClusterInfrastructureReady(context.Background(), e.Ctx.Client, cluster)
		require.Error(t, err)
	})
}

func TestClusterManageDistributeMasterNodesCerts(t *testing.T) {
	t.Run("already distributed returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		cluster.Status.Conditions = append(cluster.Status.Conditions, confv1beta1.ClusterCondition{
			Type: bkev1beta1.BocloudClusterMasterCertDistributionCondition, Status: confv1beta1.ConditionTrue,
		})
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.distributeMasterNodesCerts(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}))
	})

	t.Run("executeCommandAndWait error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(executeCommandAndWait, func(_ ExecuteCommandAndWaitParams) error { return assertErr("exec failed") })
		err := e.distributeMasterNodesCerts(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "exec failed")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(executeCommandAndWait, func(_ ExecuteCommandAndWaitParams) error { return nil })
		require.NoError(t, e.distributeMasterNodesCerts(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}))
	})
}

func TestClusterManageDistributeWorkerNodesCerts(t *testing.T) {
	t.Run("already distributed returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		cluster.Status.Conditions = append(cluster.Status.Conditions, confv1beta1.ClusterCondition{
			Type: bkev1beta1.BocloudClusterWorkerCertDistributionCondition, Status: confv1beta1.ConditionTrue,
		})
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.distributeWorkerNodesCerts(bkenode.Nodes{{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}}}))
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(executeCommandAndWait, func(_ ExecuteCommandAndWaitParams) error { return nil })
		require.NoError(t, e.distributeWorkerNodesCerts(bkenode.Nodes{{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}}}))
	})
}

func TestClusterManageDistributeTargetClusterCerts(t *testing.T) {
	t.Run("guess BKE cluster type skips distribution", func(t *testing.T) {
		cluster := cmBocloudCluster()
		cluster.Status.Conditions = append(cluster.Status.Conditions, confv1beta1.ClusterCondition{
			Type: bkev1beta1.TypeOfManagementClusterGuessCondition, Status: confv1beta1.ConditionTrue,
			Reason: cmClusterFromBKE,
		})
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.distributeTargetClusterCerts(bkenode.Nodes{{IP: "10.0.0.1"}}))
	})

	t.Run("master cert error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyPrivateMethod(e, "distributeMasterNodesCerts", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return assertErr("master cert failed") })
		err := e.distributeTargetClusterCerts(bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"master"}}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "master cert failed")
	})

	t.Run("distributes master and worker certs", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyPrivateMethod(e, "distributeMasterNodesCerts", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return nil })
		patches.ApplyPrivateMethod(e, "distributeWorkerNodesCerts", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return nil })
		require.NoError(t, e.distributeTargetClusterCerts(bkenode.Nodes{
			{IP: "10.0.0.1", Role: []string{"master"}}, {IP: "10.0.0.2", Role: []string{"worker"}},
		}))
	})
}

func TestClusterManageCompatibilityPatch(t *testing.T) {
	t.Run("no etcd pods returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.mockClient = k8sfake.NewSimpleClientset()
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		require.NoError(t, e.compatibilityPatch())
	})

	t.Run("etcd pod matching node updates annotation", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		etcdPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "etcd-m1", Namespace: metav1.NamespaceSystem, Labels: map[string]string{"component": "etcd"}},
			Spec:       corev1.PodSpec{NodeName: "m1"},
		}
		e.mockClient = k8sfake.NewSimpleClientset(etcdPod)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		patches.ApplyFunc(phaseutil.GetClientURLByIP, func(_ string) string { return "https://10.0.0.1:2379" })
		patches.ApplyFunc(phaseutil.RetryOnConflict, func(_ func() error) error { return nil })
		require.NoError(t, e.compatibilityPatch())
	})

	t.Run("etcd pod with no matching node warns and continues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		etcdPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "etcd-unknown", Namespace: metav1.NamespaceSystem,
				Labels:      map[string]string{"component": "etcd"},
				Annotations: map[string]string{annotation.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.1.1.1:2379"},
			},
			Spec: corev1.PodSpec{NodeName: "unknown-node"},
		}
		e.mockClient = k8sfake.NewSimpleClientset(etcdPod)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		require.NoError(t, e.compatibilityPatch())
	})
}

func TestClusterManageWaitForLauncherPodsComplete(t *testing.T) {
	t.Run("canceled context returns context error", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.mockClient = k8sfake.NewSimpleClientset()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := e.waitForLauncherPodsComplete(ctx, cluster)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "context canceled")
	})

	t.Run("all pods running returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		start := metav1.NewTime(time.Now().Add(-1 * time.Minute))
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: launcherDaemonSetName, Namespace: launcherNamespace},
			Spec:       appsv1.DaemonSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "launcher"}}},
			Status:     appsv1.DaemonSetStatus{DesiredNumberScheduled: 1, NumberReady: 1},
		}
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "launcher-pod", Namespace: launcherNamespace, Labels: map[string]string{"app": "launcher"}},
			Status: corev1.PodStatus{
				Phase:     corev1.PodRunning,
				StartTime: &start,
				Conditions: []corev1.PodCondition{
					{Type: corev1.PodReady, Status: corev1.ConditionTrue},
				},
			},
		}
		e.mockClient = k8sfake.NewSimpleClientset(ds, pod)
		_, err := e.waitForLauncherPodsComplete(context.Background(), cluster)
		require.NoError(t, err)
	})
}

func TestClusterManageFakeBootstrapWorker(t *testing.T) {
	t.Run("no worker nodes returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Role: []string{"master"}}}, nil)
		require.NoError(t, e.fakeBootstrapWorker())
	})

	t.Run("with worker nodes delegates to doFakeBootstrapWorker", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.2", Role: []string{"worker"}}}, nil)
		patches.ApplyPrivateMethod(e, "doFakeBootstrapWorker", func(_ *EnsureClusterManage, _ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ bkenode.Nodes, _ int32, _ *bkev1beta1.BKELogger) error {
			return nil
		})
		require.NoError(t, e.fakeBootstrapWorker())
	})
}

func TestClusterManageReconcileFakeBootstrap(t *testing.T) {
	t.Run("not bocloud cluster returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		annotation.SetAnnotation(cluster, cmClusterFromKey, cmClusterFromBKE)
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.reconcileFakeBootstrap())
	})

	t.Run("fully controlled returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		clusterutil.MarkClusterFullyControlled(cluster)
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.reconcileFakeBootstrap())
	})

	t.Run("no owner references returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.reconcileFakeBootstrap())
	})

	t.Run("cluster nil after refresh returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cluster.OwnerReferences = []metav1.OwnerReference{{Kind: "Cluster"}}
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = nil
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		require.NoError(t, e.reconcileFakeBootstrap())
	})

	t.Run("master bootstrap error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cluster.OwnerReferences = []metav1.OwnerReference{{Kind: "Cluster"}}
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyPrivateMethod(e, "fakeBootstrapMaster", func(_ *EnsureClusterManage) error { return assertErr("master failed") })
		err := e.reconcileFakeBootstrap()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to fake bootstrap master")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cluster.OwnerReferences = []metav1.OwnerReference{{Kind: "Cluster"}}
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyPrivateMethod(e, "fakeBootstrapMaster", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "fakeBootstrapWorker", func(_ *EnsureClusterManage) error { return nil })
		require.NoError(t, e.reconcileFakeBootstrap())
	})
}

func TestClusterManageBackupBocloudClusterData(t *testing.T) {
	t.Run("already backed up returns nil", func(t *testing.T) {
		cluster := cmBocloudCluster()
		cluster.Status.Conditions = append(cluster.Status.Conditions, confv1beta1.ClusterCondition{
			Type: bkev1beta1.BocloudClusterDataBackupCondition, Status: confv1beta1.ConditionTrue,
		})
		e := newClusterManageCov(t, cluster)
		require.NoError(t, e.backupBocloudClusterData(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}))
	})

	t.Run("New error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(_ *command.Custom) error { return assertErr("new failed") })
		err := e.backupBocloudClusterData(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "new failed")
	})

	t.Run("Wait error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return assertErr("wait failed"), nil, nil
		})
		err := e.backupBocloudClusterData(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait failed")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return nil, []string{"m1"}, nil
		})
		require.NoError(t, e.backupBocloudClusterData(bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}))
	})
}

func TestClusterManageExecute(t *testing.T) {
	t.Run("getRemoteClient error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return nil, assertErr("no remote client")
		})
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no remote client")
	})

	t.Run("collectBaseInfo error returns", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "collectBaseInfo", func(_ *EnsureClusterManage) error { return assertErr("collect failed") })
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "collect failed")
	})

	t.Run("not bocloud cluster returns nil after collect", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		annotation.SetAnnotation(cluster, cmClusterFromKey, cmClusterFromBKE)
		e := newClusterManageCov(t, cluster)
		patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "collectBaseInfo", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "pushAgent", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "collectAgentInfo", func(_ *EnsureClusterManage) error { return nil })
		_, err := e.Execute()
		require.NoError(t, err)
	})

	t.Run("bocloud cluster manage success requeues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		cluster.OwnerReferences = []metav1.OwnerReference{{Kind: "Cluster"}}
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyPrivateMethod(e, "getRemoteClient", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "collectBaseInfo", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "pushAgent", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "collectAgentInfo", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster", func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
		patches.ApplyPrivateMethod(e, "bocloudClusterManagePrepare", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "reconcileFakeBootstrap", func(_ *EnsureClusterManage) error { return nil })
		patches.ApplyPrivateMethod(e, "compatibilityPatch", func(_ *EnsureClusterManage) error { return nil })
		res, err := e.Execute()
		require.NoError(t, err)
		assert.True(t, res.Requeue)
	})
}

func TestClusterManageNeedExecute(t *testing.T) {
	t.Run("bke cluster returns false", func(t *testing.T) {
		cluster := cmBocloudCluster()
		annotation.SetAnnotation(cluster, cmClusterFromKey, cmClusterFromBKE)
		e := newClusterManageCov(t, cluster)
		assert.False(t, e.NeedExecute(cluster, cluster))
	})

	t.Run("fully controlled returns false", func(t *testing.T) {
		cluster := cmBocloudCluster()
		clusterutil.MarkClusterFullyControlled(cluster)
		e := newClusterManageCov(t, cluster)
		assert.False(t, e.NeedExecute(cluster, cluster))
	})

	t.Run("bocloud cluster returns true", func(t *testing.T) {
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		assert.True(t, e.NeedExecute(cluster, cluster))
	})
}

func TestClusterManageInitBocloudClusterEnv(t *testing.T) {
	patchBKENodes := func(patches *gomonkey.Patches, nodes bkev1beta1.BKENodes, err error) {
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nodes, err
		})
	}

	t.Run("GetBKENodes error returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patchBKENodes(patches, nil, assertErr("no bke nodes"))
		require.NoError(t, e.initBocloudClusterEnv())
	})

	t.Run("no nodes need init returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patchBKENodes(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{}
		})
		require.NoError(t, e.initBocloudClusterEnv())
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patchBKENodes(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}
		})
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return nil, []string{"m1"}, nil
		})
		patches.ApplyFunc(phaseutil.GetNodeIPFromCommandWaitResult, func(result string) string { return result })
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
			return nil
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).MarkNodeStateFlagForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) error {
			return nil
		})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		require.NoError(t, e.initBocloudClusterEnv())
	})

	t.Run("failed nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		patchBKENodes(patches, bkev1beta1.BKENodes{}, nil)
		patches.ApplyFunc(phaseutil.GetNeedInitEnvNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}
		})
		patches.ApplyFunc((*command.Custom).New, func(c *command.Custom) error { c.Command = &agentv1beta1.Command{}; return nil })
		patches.ApplyFunc((*command.Custom).Wait, func(_ *command.Custom) (error, []string, []string) {
			return nil, nil, []string{"m1"}
		})
		patches.ApplyFunc(phaseutil.GetNodeIPFromCommandWaitResult, func(result string) string { return result })
		patches.ApplyFunc((*nodeutil.NodeFetcher).UpdateNodeStatusByIP, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ func(*confv1beta1.BKENodeStatus)) error {
			return nil
		})
		patches.ApplyFunc(
			(*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster,
				_ string, _ confv1beta1.NodeState, _ string,
			) error {
				return nil
			},
		)
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })
		patches.ApplyFunc(phaseutil.LogCommandFailed, func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
			return nil, nil
		})
		patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs, func(context.Context, client.Client, *bkev1beta1.BKECluster, map[string][]string) {})
		err := e.initBocloudClusterEnv()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to init bocloud cluster env")
	})
}

func TestClusterManageBocloudClusterManagePrepare(t *testing.T) {
	t.Run("certs LookUpOrGenerate error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyFunc(certs.NewKubernetesCertGeneratorWithCache, func(_ context.Context, _ client.Client, _ cache.Cache, _ *bkev1beta1.BKECluster) *certs.BKEKubernetesCertGenerator {
			return &certs.BKEKubernetesCertGenerator{}
		})
		patches.ApplyFunc((*certs.BKEKubernetesCertGenerator).LookUpOrGenerate, func(_ *certs.BKEKubernetesCertGenerator) error {
			return assertErr("cert failed")
		})
		err := e.bocloudClusterManagePrepare()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cert failed")
	})

	t.Run("backup error swallowed continues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyFunc(certs.NewKubernetesCertGeneratorWithCache, func(_ context.Context, _ client.Client, _ cache.Cache, _ *bkev1beta1.BKECluster) *certs.BKEKubernetesCertGenerator {
			return &certs.BKEKubernetesCertGenerator{}
		})
		patches.ApplyFunc((*certs.BKEKubernetesCertGenerator).LookUpOrGenerate, func(_ *certs.BKEKubernetesCertGenerator) error { return nil })
		patches.ApplyPrivateMethod(e, "backupBocloudClusterData", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return assertErr("backup failed") })
		patches.ApplyPrivateMethod(e, "distributeTargetClusterCerts", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return nil })
		patches.ApplyPrivateMethod(e, "initBocloudClusterEnv", func(_ *EnsureClusterManage) error { return nil })
		require.NoError(t, e.bocloudClusterManagePrepare())
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{
			{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}},
			{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}},
		}, nil)
		patches.ApplyFunc(certs.NewKubernetesCertGeneratorWithCache, func(_ context.Context, _ client.Client, _ cache.Cache, _ *bkev1beta1.BKECluster) *certs.BKEKubernetesCertGenerator {
			return &certs.BKEKubernetesCertGenerator{}
		})
		patches.ApplyFunc((*certs.BKEKubernetesCertGenerator).LookUpOrGenerate, func(_ *certs.BKEKubernetesCertGenerator) error { return nil })
		patches.ApplyPrivateMethod(e, "backupBocloudClusterData", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return nil })
		patches.ApplyPrivateMethod(e, "distributeTargetClusterCerts", func(_ *EnsureClusterManage, _ bkenode.Nodes) error { return nil })
		patches.ApplyPrivateMethod(e, "initBocloudClusterEnv", func(_ *EnsureClusterManage) error { return nil })
		require.NoError(t, e.bocloudClusterManagePrepare())
	})
}

// cmGapsPatchGetNodeStateFlag patches the deep (*nodeutil.NodeFetcher).GetNodeStateFlag
// function so that GetNodeStateFlagForCluster (short wrapper) returns the configured
// values. We never patch the short wrapper directly to avoid arm64 trampoline issues.
func cmGapsPatchGetNodeStateFlag(patches *gomonkey.Patches, hasFlag bool, err error) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ int) (bool, error) {
			return hasFlag, err
		})
}

// cmGapsPatchMarkNodeStateFlag patches the MarkNodeStateFlagForCluster method on
// NodeFetcher so that node state flag marking does not hit the API server.
func cmGapsPatchMarkNodeStateFlag(patches *gomonkey.Patches, err error) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).MarkNodeStateFlagForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) error {
			return err
		})
}

// cmGapsPatchRefreshCtxBKECluster patches the PhaseContext.RefreshCtxBKECluster method
// so it does not hit the API server.
func cmGapsPatchRefreshCtxBKECluster(patches *gomonkey.Patches) {
	patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxBKECluster",
		func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
}

// cmGapsPatchRefreshCtxCluster patches the PhaseContext.RefreshCtxCluster method
// so it does not hit the API server.
func cmGapsPatchRefreshCtxCluster(patches *gomonkey.Patches) {
	patches.ApplyMethod(&phaseframe.PhaseContext{}, "RefreshCtxCluster",
		func(_ *phaseframe.PhaseContext, _ ...context.Context) error { return nil })
}

// cmGapsFakeRemoteClient implements the InstallAddon method of kube.RemoteKubeClient
// so that pushAgent can exercise the InstallAddon call without a real remote cluster.
type cmGapsFakeRemoteClient struct {
	kube.RemoteKubeClient
	installAddonErr error
	installCalls    int
}

func (f *cmGapsFakeRemoteClient) InstallAddon(_ *bkev1beta1.BKECluster, _ *bkeaddon.AddonTransfer,
	_ *kube.AddonRecorder, _ client.Client, _ bkenode.Nodes) error {
	f.installCalls++
	return f.installAddonErr
}

// cmGapsControlPlaneInitializedCluster returns a clusterv1.Cluster with the
// ControlPlaneInitializedCondition set to True.
func cmGapsControlPlaneInitializedCluster() *clusterv1.Cluster {
	c := &clusterv1.Cluster{}
	c.Status.Conditions = append(c.Status.Conditions, clusterv1.Condition{
		Type:   clusterv1.ControlPlaneInitializedCondition,
		Status: corev1.ConditionTrue,
	})
	return c
}

// ---------------------------------------------------------------------------
// pushAgent
// ---------------------------------------------------------------------------

func TestClusterManagePushAgent(t *testing.T) {
	t.Run("get nodes error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, nil, assertErr("get nodes failed"))
		err := e.pushAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get nodes")
	})

	t.Run("agent already pushed returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		cmGapsPatchGetNodeStateFlag(patches, true, nil) // already pushed
		require.NoError(t, e.pushAgent())
	})

	t.Run("local kubeconfig not found returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		cmGapsPatchGetNodeStateFlag(patches, false, nil) // not pushed
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig,
			func(_ context.Context, _ client.Client) ([]byte, error) {
				return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "secrets"}, "local-kubeconfig")
			})
		err := e.pushAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "local kubeconfig secret not found")
	})

	t.Run("local kubeconfig other error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		cmGapsPatchGetNodeStateFlag(patches, false, nil)
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig,
			func(_ context.Context, _ client.Client) ([]byte, error) {
				return nil, assertErr("secret get failed")
			})
		err := e.pushAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get local kubeconfig secret")
	})

	t.Run("install addon error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		cmGapsPatchGetNodeStateFlag(patches, false, nil)
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig,
			func(_ context.Context, _ client.Client) ([]byte, error) {
				return []byte("kubeconfig-data"), nil
			})
		e.remoteClient = &cmGapsFakeRemoteClient{installAddonErr: assertErr("install failed")}
		err := e.pushAgent()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "install failed")
	})

	t.Run("success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1"}}, nil)
		cmGapsPatchGetNodeStateFlag(patches, false, nil)
		patches.ApplyFunc(phaseutil.GetLocalKubeConfig,
			func(_ context.Context, _ client.Client) ([]byte, error) {
				return []byte("kubeconfig-data"), nil
			})
		e.remoteClient = &cmGapsFakeRemoteClient{installAddonErr: nil}
		// Patch private helpers that are not the unit under test.
		patches.ApplyPrivateMethod(e, "waitForLauncherPodsComplete",
			func(_ *EnsureClusterManage, _ context.Context, _ *bkev1beta1.BKECluster) error { return nil })
		cmGapsPatchMarkNodeStateFlag(patches, nil)
		patches.ApplyPrivateMethod(e, "waitAgentPushedFlagVisible",
			func(_ *EnsureClusterManage, _ context.Context, _ phaseutil.NodeStateFlagReader,
				_ *bkev1beta1.BKECluster, _ bkenode.Nodes, _ *bkev1beta1.BKELogger) error {
				return nil
			})
		patches.ApplyFunc(phaseutil.PingBKEAgent,
			func(_ context.Context, _ client.Client, _ *runtime.Scheme,
				_ *bkev1beta1.BKECluster) (error, []string, []string) {
				return nil, []string{"10.0.0.1"}, nil
			})
		patches.ApplyPrivateMethod(e, "processAgentPingResults",
			func(_ *EnsureClusterManage, _ context.Context, _ *bkev1beta1.BKECluster,
				_ []string, _ []string, _ *bkev1beta1.BKELogger) {
			})
		cmPatchSyncNoop(patches)
		require.NoError(t, e.pushAgent())
	})
}

// ---------------------------------------------------------------------------
// fakeBootstrapMaster
// ---------------------------------------------------------------------------

func TestClusterManageFakeBootstrapMaster(t *testing.T) {
	t.Run("get nodes error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, nil, assertErr("get nodes failed"))
		err := e.fakeBootstrapMaster()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get nodes")
	})

	t.Run("update kcp replicas error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyPrivateMethod(e, "updateKubeadmControlPlaneReplicas",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client, _ int32) error {
				return assertErr("kcp update failed")
			})
		err := e.fakeBootstrapMaster()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "kcp update failed")
	})

	t.Run("wait infra ready error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyPrivateMethod(e, "updateKubeadmControlPlaneReplicas",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client, _ int32) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForClusterInfrastructureReady",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client,
				_ *bkev1beta1.BKECluster) error {
				return assertErr("infra not ready")
			})
		err := e.fakeBootstrapMaster()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "infra not ready")
	})

	t.Run("wait master nodes bootstrap error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyPrivateMethod(e, "updateKubeadmControlPlaneReplicas",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client, _ int32) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForClusterInfrastructureReady",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client,
				_ *bkev1beta1.BKECluster) error {
				return nil
			})
		patches.ApplyPrivateMethod(e, "waitForMasterNodesBootstrap",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client,
				_ *bkev1beta1.BKECluster, _ bkenode.Nodes, _ int32,
				_ *bkev1beta1.BKELogger) (map[int]confv1beta1.Node, error) {
				return nil, assertErr("master bootstrap failed")
			})
		err := e.fakeBootstrapMaster()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "master bootstrap failed")
	})

	t.Run("success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		cmPatchGetNodes(patches, bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}, nil)
		patches.ApplyPrivateMethod(e, "updateKubeadmControlPlaneReplicas",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client, _ int32) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForClusterInfrastructureReady",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client,
				_ *bkev1beta1.BKECluster) error {
				return nil
			})
		patches.ApplyPrivateMethod(e, "waitForMasterNodesBootstrap",
			func(_ *EnsureClusterManage, _ context.Context, _ client.Client,
				_ *bkev1beta1.BKECluster, _ bkenode.Nodes, _ int32,
				_ *bkev1beta1.BKELogger) (map[int]confv1beta1.Node, error) {
				return map[int]confv1beta1.Node{0: {IP: "10.0.0.1"}}, nil
			})
		cmGapsPatchRefreshCtxBKECluster(patches)
		patches.ApplyPrivateMethod(e, "markNodesBootstrapSuccess",
			func(_ *EnsureClusterManage, _ context.Context,
				_ map[int]confv1beta1.Node, _ *bkev1beta1.BKELogger) {
			})
		cmPatchSyncNoop(patches)
		require.NoError(t, e.fakeBootstrapMaster())
	})
}

// ---------------------------------------------------------------------------
// waitForMasterNodesBootstrap
// ---------------------------------------------------------------------------

func TestClusterManageWaitForMasterNodesBootstrap(t *testing.T) {
	t.Run("success returns nodes map", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = cmGapsControlPlaneInitializedCluster()
		cmGapsPatchRefreshCtxCluster(patches)
		patches.ApplyFunc(waitForNodesBootstrap,
			func(_ nodesBootstrapInput) error {
				return nil
			})
		masterNodes := bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}
		result, err := e.waitForMasterNodesBootstrap(nodesBootstrapInput{
			ctx: context.Background(), client: e.Ctx.Client, cluster: cluster,
			nodes: masterNodes, expectedCount: 1, log: e.Ctx.Log,
		})
		require.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("timeout returns error when control plane not initialized", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{} // no conditions
		cmGapsPatchRefreshCtxCluster(patches)
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // immediately cancel so the poll times out after one condition call
		masterNodes := bkenode.Nodes{{IP: "10.0.0.1", Hostname: "m1", Role: []string{"master"}}}
		result, err := e.waitForMasterNodesBootstrap(nodesBootstrapInput{
			ctx: ctx, client: e.Ctx.Client, cluster: cluster,
			nodes: masterNodes, expectedCount: 1, log: e.Ctx.Log,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to wait for cluster")
		assert.Nil(t, result)
	})
}

// ---------------------------------------------------------------------------
// doFakeBootstrapWorker
// ---------------------------------------------------------------------------

func TestClusterManageDoFakeBootstrapWorker(t *testing.T) {
	t.Run("get machine deployment error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyFunc(phaseutil.GetClusterAPIMachineDeployment,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*clusterv1.MachineDeployment, error) {
				return nil, assertErr("md not found")
			})
		workerNodes := bkenode.Nodes{{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}}}
		err := e.doFakeBootstrapWorker(nodesBootstrapInput{
			ctx: context.Background(), client: e.Ctx.Client, cluster: cluster,
			nodes: workerNodes, expectedCount: 1, log: e.Ctx.Log,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "md not found")
	})

	t.Run("resume cluster api obj error returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyFunc(phaseutil.GetClusterAPIMachineDeployment,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*clusterv1.MachineDeployment, error) {
				return &clusterv1.MachineDeployment{}, nil
			})
		patches.ApplyFunc(phaseutil.ResumeAndUpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, _ *clusterv1.MachineDeployment, _ int32) error {
				return assertErr("resume failed")
			})
		workerNodes := bkenode.Nodes{{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}}}
		err := e.doFakeBootstrapWorker(nodesBootstrapInput{
			ctx: context.Background(), client: e.Ctx.Client, cluster: cluster,
			nodes: workerNodes, expectedCount: 1, log: e.Ctx.Log,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "resume failed")
	})

	t.Run("timeout returns error when nodes not ready", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyFunc(phaseutil.GetClusterAPIMachineDeployment,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*clusterv1.MachineDeployment, error) {
				return &clusterv1.MachineDeployment{}, nil
			})
		patches.ApplyFunc(phaseutil.ResumeAndUpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, _ *clusterv1.MachineDeployment, _ int32) error { return nil })
		patches.ApplyFunc(waitForNodesBootstrap,
			func(_ nodesBootstrapInput) error {
				return assertErr("not ready")
			})
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // immediately cancel so the poll times out after one condition call
		workerNodes := bkenode.Nodes{{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}}}
		err := e.doFakeBootstrapWorker(nodesBootstrapInput{
			ctx: ctx, client: e.Ctx.Client, cluster: cluster,
			nodes: workerNodes, expectedCount: 1, log: e.Ctx.Log,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to wait for cluster")
	})

	t.Run("success returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := cmBocloudCluster()
		e := newClusterManageCov(t, cluster)
		e.Ctx.Cluster = &clusterv1.Cluster{}
		patches.ApplyFunc(phaseutil.GetClusterAPIMachineDeployment,
			func(_ context.Context, _ client.Client, _ *clusterv1.Cluster) (*clusterv1.MachineDeployment, error) {
				return &clusterv1.MachineDeployment{}, nil
			})
		patches.ApplyFunc(phaseutil.ResumeAndUpdateMachineDeploymentReplicas,
			func(_ context.Context, _ client.Client, _ *clusterv1.MachineDeployment, _ int32) error { return nil })
		patches.ApplyFunc(waitForNodesBootstrap,
			func(_ nodesBootstrapInput) error {
				return nil
			})
		cmGapsPatchRefreshCtxBKECluster(patches)
		patches.ApplyPrivateMethod(e, "markNodesBootstrapSuccess",
			func(_ *EnsureClusterManage, _ context.Context,
				_ map[int]confv1beta1.Node, _ *bkev1beta1.BKELogger) {
			})
		cmPatchSyncNoop(patches)
		workerNodes := bkenode.Nodes{{IP: "10.0.0.2", Hostname: "w1", Role: []string{"worker"}}}
		require.NoError(t, e.doFakeBootstrapWorker(nodesBootstrapInput{
			ctx: context.Background(), client: e.Ctx.Client, cluster: cluster,
			nodes: workerNodes, expectedCount: 1, log: e.Ctx.Log,
		}))
	})
}

func TestEnsureClusterManage_CreateLauncherAddon_AgentLogLevel(t *testing.T) {
	cluster := cmBocloudCluster()
	cluster.Spec.ClusterConfig.Cluster.NTPServer = "ntp.invalid"
	cluster.Spec.ClusterConfig.Cluster.AgentHealthPort = "9443"
	cluster.Spec.ClusterConfig.CustomExtra = map[string]string{
		phaseutil.AgentLogLevelCustomExtraKey: "DEBUG",
	}
	e := newClusterManageCov(t, cluster)

	addon, err := e.createLauncherAddon(cluster, []byte("kubeconfig-data"), e.Ctx.Log)
	require.NoError(t, err)
	require.NotNil(t, addon)
	require.NotNil(t, addon.Addon)
	assert.Equal(t, "DEBUG", addon.Addon.Param["agentLogLevel"])
	assert.Equal(t, "ntp.invalid", addon.Addon.Param["ntpServer"])
	assert.Equal(t, "9443", addon.Addon.Param["agentHealthPort"])
}
