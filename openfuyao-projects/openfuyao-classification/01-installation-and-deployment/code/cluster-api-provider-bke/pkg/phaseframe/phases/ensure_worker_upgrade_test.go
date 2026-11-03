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
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	bkelog "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	kubedrain "k8s.io/kubectl/pkg/drain"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureWorkerUpgradeConstants(t *testing.T) {
	assert.Equal(t, "EnsureWorkerUpgrade", string(EnsureWorkerUpgradeName))
	assert.Equal(t, 2, WorkerNodeHealthCheckPollIntervalSeconds)
	assert.Equal(t, 5, WorkerNodeHealthCheckTimeoutMinutes)
}

func TestNewEnsureWorkerUpgrade(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureWorkerUpgrade(ctx)
	assert.NotNil(t, phase)
}

func TestCreateUpgradeCommand(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()

	node := &confv1beta1.Node{IP: "192.168.1.1", Hostname: "node1"}
	params := CreateUpgradeCommandParams{
		Ctx:         context.Background(),
		Namespace:   "default",
		Client:      c,
		Scheme:      scheme,
		OwnerObj:    bkeCluster,
		ClusterName: "test",
		Node:        node,
		BKEConfig:   "test-config",
		Phase:       bkev1beta1.UpgradeWorker,
	}

	cmd := createUpgradeCommand(params)
	assert.NotNil(t, cmd)
	assert.Equal(t, "default", cmd.NameSpace)
	assert.Equal(t, "test", cmd.ClusterName)
	assert.Equal(t, node, cmd.Node)
	assert.Equal(t, "test-config", cmd.BKEConfig)
	assert.Equal(t, bkev1beta1.UpgradeWorker, cmd.Phase)
	assert.True(t, cmd.Unique)
	assert.NotNil(t, cmd.Ctx)
	assert.NotNil(t, cmd.Client)
	assert.NotNil(t, cmd.Scheme)
	assert.NotNil(t, cmd.OwnerObj)
}

func TestCreateUpgradeCommandParams_Structure(t *testing.T) {
	params := CreateUpgradeCommandParams{
		Ctx:         context.Background(),
		Namespace:   "default",
		Client:      &fakeClient{},
		Scheme:      runtime.NewScheme(),
		OwnerObj:    &bkev1beta1.BKECluster{},
		ClusterName: "test",
		Node:        &confv1beta1.Node{},
		BKEConfig:   "config",
		Phase:       bkev1beta1.UpgradeWorker,
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.Scheme)
	assert.NotNil(t, params.OwnerObj)
	assert.NotNil(t, params.Node)
}

func TestPrepareUpgradeNodesParams_Structure(t *testing.T) {
	params := PrepareUpgradeNodesParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
}

func TestWaitForWorkerNodeHealthCheckParams_Structure(t *testing.T) {
	params := WaitForWorkerNodeHealthCheckParams{
		Ctx:        context.Background(),
		K8sVersion: "v1.28.0",
		Logger:     createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.Equal(t, "v1.28.0", params.K8sVersion)
	assert.NotNil(t, params.Logger)
}

func TestWaitForNodeHealthParams_Structure(t *testing.T) {
	params := WaitForNodeHealthParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Node:       confv1beta1.Node{IP: "192.168.1.1"},
		Log:        createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.Equal(t, "192.168.1.1", params.Node.IP)
	assert.NotNil(t, params.Log)
}

func TestProcessNodeUpgradeParams_Structure(t *testing.T) {
	params := ProcessNodeUpgradeParams{
		Ctx:        context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
		Log:        createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
}

func TestEnsureWorkerUpgrade_NeedExecute_UnhealthyCluster(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterUnhealthy,
		},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureWorkerUpgrade(ctx)
	e := phase.(*EnsureWorkerUpgrade)

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func TestEnsureWorkerUpgrade_NeedExecute_NoUpgradeNodes(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := workerUpgradeTestCluster("v1.28.0", "v1.28.0")
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureWorkerUpgrade(ctx)
	e := phase.(*EnsureWorkerUpgrade)

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func TestEnsureWorkerUpgradeDesiredKubernetesVersion_PrefersVersionContext(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.33.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.33.0"},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent(upgrade.ComponentKubernetesWorker, "v1.33.0")
	vc.SetTarget(upgrade.ComponentKubernetesWorker, "v1.34.1")

	e := &EnsureWorkerUpgrade{BasePhase: phaseframe.BasePhase{Ctx: &phaseframe.PhaseContext{
		BKECluster:     bkeCluster,
		VersionContext: vc,
	}}}

	assert.Equal(t, "v1.34.1", e.desiredKubernetesVersion())
	assert.Equal(t, "v1.33.0", e.currentKubernetesVersion())
}

func TestEnsureWorkerUpgradeSyncLegacyTargetKubernetesVersion(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.33.0"},
			},
		},
	}
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, bc *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
		for _, patch := range patchs {
			patch(bc)
		}
		return nil
	})

	e := NewEnsureWorkerUpgrade(ctx).(*EnsureWorkerUpgrade)
	require.NoError(t, e.syncLegacyTargetKubernetesVersion("v1.34.1"))
	assert.Equal(t, "v1.34.1", bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion)
}

func TestEnsureWorkerUpgradeReconcileWorkerUpgradeUsesVersionContext(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status:     confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.33.0"},
	}
	vc := upgrade.NewVersionContext()
	vc.SetCurrent(upgrade.ComponentKubernetesWorker, "v1.33.0")
	vc.SetTarget(upgrade.ComponentKubernetesWorker, "v1.34.1")

	ctx := &phaseframe.PhaseContext{
		Context:        context.Background(),
		BKECluster:     bkeCluster,
		Log:            bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
		VersionContext: vc,
	}

	e := NewEnsureWorkerUpgrade(ctx).(*EnsureWorkerUpgrade)
	_, err := e.reconcileWorkerUpgrade()
	require.Error(t, err)
	assert.EqualError(t, err, "cluster config is nil")
}

func TestEnsureWorkerUpgrade_ExecutePreHook(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultPreHook", func(_ *phaseframe.BasePhase) error {
		return nil
	})

	phase := NewEnsureWorkerUpgrade(ctx)
	e := phase.(*EnsureWorkerUpgrade)
	err := e.ExecutePreHook()
	assert.NoError(t, err)
}

func TestWaitForWorkerNodeHealthCheck_ContextDone(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	params := WaitForWorkerNodeHealthCheckParams{
		Ctx:        ctx,
		K8sVersion: "v1.28.0",
		Logger:     createTestLogger(),
	}

	err := waitForWorkerNodeHealthCheck(params)
	assert.Error(t, err)
}

func TestEnsureWorkerUpgrade_NeedExecute_WithUpgradeNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := workerUpgradeTestCluster("v1.29.0", "v1.28.0")
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(ctx *phaseframe.PhaseContext, cluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
		return bkev1beta1.BKENodes{}, true
	})

	patches.ApplyFunc(phaseutil.GetNeedUpgradeWorkerNodesWithBKENodes, func(cluster *bkev1beta1.BKECluster, nodes bkev1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{{IP: "192.168.1.1"}}
	})

	phase := NewEnsureWorkerUpgrade(ctx)
	e := phase.(*EnsureWorkerUpgrade)

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.True(t, result)
}

func TestEnsureWorkerUpgrade_NeedExecute_CPNotInitialized(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status:     confv1beta1.BKEClusterStatus{},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(ctx *phaseframe.PhaseContext, cluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
		return nil, false
	})

	phase := NewEnsureWorkerUpgrade(ctx).(*EnsureWorkerUpgrade)
	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func TestEnsureWorkerUpgrade_NeedExecute_ClusterUnknown(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterUnknown,
		},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	phase := NewEnsureWorkerUpgrade(ctx).(*EnsureWorkerUpgrade)
	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func TestEnsureWorkerUpgrade_NeedExecute_DefaultNeedExecuteFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status:     confv1beta1.BKEClusterStatus{},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, old, new *bkev1beta1.BKECluster) bool {
		return false
	})

	phase := NewEnsureWorkerUpgrade(ctx).(*EnsureWorkerUpgrade)
	result := phase.NeedExecute(nil, bkeCluster)
	assert.False(t, result)
}

func TestEnsureWorkerUpgrade_PrepareUpgradeNodes(t *testing.T) {
	t.Skip("Skipping - requires complex NodeFetcher mocking")
}

func TestEnsureWorkerUpgrade_Execute(t *testing.T) {
	t.Skip("Skipping - requires mocking private method reconcileWorkerUpgrade")
}

func TestEnsureWorkerUpgrade_ReconcileWorkerUpgrade(t *testing.T) {
	t.Skip("Skipping - private method cannot be mocked with gomonkey")
}

func TestEnsureWorkerUpgrade_RolloutUpgrade(t *testing.T) {
	t.Skip("Skipping - private method cannot be mocked with gomonkey")
}

func TestEnsureWorkerUpgrade_ProcessNodeUpgrade(t *testing.T) {
	t.Skip("Skipping - requires complex kubernetes client mocking")
}

func TestEnsureWorkerUpgrade_ExecuteNodeUpgrade(t *testing.T) {
	t.Skip("Skipping - requires complex command mocking")
}

func TestEnsureWorkerUpgrade_WaitForNodeHealth(t *testing.T) {
	t.Skip("Skipping - requires complex remote client mocking")
}

func TestEnsureWorkerUpgrade_UpgradeNode(t *testing.T) {
	t.Skip("Skipping - requires mocking private methods")
}

// newWorkerUpgradeCov builds an EnsureWorkerUpgrade wired to a fake
// controller-runtime client for the internal-access test package.
func newWorkerUpgradeCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureWorkerUpgrade {
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
	return &EnsureWorkerUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// workerUpgradeCovCluster builds a BKECluster with non-nil ClusterConfig so
// syncLegacyTargetKubernetesVersion can run its real happy path.
func workerUpgradeCovCluster(specK8s, statusK8s string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: specK8s},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			KubernetesVersion:  statusK8s,
			ClusterHealthState: bkev1beta1.Healthy,
		},
	}
}

func workerUpgradeNode(ip, host string) confv1beta1.Node {
	return confv1beta1.Node{IP: ip, Hostname: host, Role: []string{"worker"}}
}

// ---- Execute ----

func TestEnsureWorkerUpgradeExecuteCov(t *testing.T) {
	t.Run("annotation missing sync error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return assertErr("sync failed")
		})
		_, err := e.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync failed")
	})

	t.Run("annotation already set, reconcile error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		cluster := workerUpgradeCovCluster("v1.29.0", "v1.29.0")
		annotation.SetAnnotation(cluster, "deployAction", "k8s_upgrade")
		e := newWorkerUpgradeCov(t, cluster)
		// target == current -> reconcileWorkerUpgrade returns nil (no rollout)
		_, err := e.Execute()
		require.NoError(t, err)
	})

	t.Run("annotation missing sync success then reconcile", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		// target == current so reconcileWorkerUpgrade returns nil without rollout
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.29.0"))
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, bc *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			for _, p := range patchs {
				p(bc)
			}
			return nil
		})
		_, err := e.Execute()
		require.NoError(t, err)
		// deployAction annotation should be set on cluster now
		v, ok := annotation.HasAnnotation(e.Ctx.BKECluster, "deployAction")
		require.True(t, ok)
		assert.Equal(t, "k8s_upgrade", v)
	})
}

// ---- Version ----

func TestEnsureWorkerUpgradeVersionCov(t *testing.T) {
	t.Run("returns kubernetes version from status", func(t *testing.T) {
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		assert.Equal(t, "v1.28.0", e.Version())
	})
	t.Run("nil ctx returns empty", func(t *testing.T) {
		assert.Equal(t, "", (&EnsureWorkerUpgrade{}).Version())
	})
	t.Run("nil bkecluster returns empty", func(t *testing.T) {
		e := &EnsureWorkerUpgrade{BasePhase: phaseframe.BasePhase{Ctx: &phaseframe.PhaseContext{}}}
		assert.Equal(t, "", e.Version())
	})
}

// ---- prepareUpgradeNodes (real body via seams) ----

func TestEnsureWorkerUpgradePrepareUpgradeNodes(t *testing.T) {
	t.Run("success filters not-ready agents", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedUpgradeWorkerNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1"), workerUpgradeNode("10.0.0.2", "n2")}
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, ip string, flag int) (bool, error) {
			// n1 ready, n2 not ready
			return ip == "10.0.0.1", nil
		})
		params := PrepareUpgradeNodesParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Log:        e.Ctx.Log,
		}
		nodes, err := e.prepareUpgradeNodes(params)
		require.NoError(t, err)
		require.Len(t, nodes, 1)
		assert.Equal(t, "10.0.0.1", nodes[0].IP)
	})

	t.Run("GetBKENodesWrapper error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assertErr("fetch failed")
		})
		params := PrepareUpgradeNodesParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Log:        e.Ctx.Log,
		}
		_, err := e.prepareUpgradeNodes(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get BKENodes")
	})

	t.Run("all agents not ready returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedUpgradeWorkerNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")}
		})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlag, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _, _ string, _ int) (bool, error) {
			return false, nil // not ready
		})
		params := PrepareUpgradeNodesParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Log:        e.Ctx.Log,
		}
		_, err := e.prepareUpgradeNodes(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "BKEAgent is not ready")
	})
}

// ---- processNodeUpgrade (real body via seams) ----

func TestEnsureWorkerUpgradeProcessNodeUpgrade(t *testing.T) {
	t.Run("node already at target version skipped", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return &corev1.Node{Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.29.0"}}}, nil
		})
		params := ProcessNodeUpgradeParams{
			Ctx:              e.Ctx.Context,
			Client:           e.Ctx.Client,
			BKECluster:       e.Ctx.BKECluster,
			NeedUpgradeNodes: bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")},
			Log:              e.Ctx.Log,
		}
		_, failed, err := e.processNodeUpgrade(params)
		require.NoError(t, err)
		assert.Empty(t, failed)
	})

	t.Run("GetRemoteNodeByBKENode error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return nil, assertErr("remote node not found")
		})
		params := ProcessNodeUpgradeParams{
			Ctx:              e.Ctx.Context,
			Client:           e.Ctx.Client,
			BKECluster:       e.Ctx.BKECluster,
			NeedUpgradeNodes: bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")},
			Log:              e.Ctx.Log,
		}
		_, _, err := e.processNodeUpgrade(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "get remote cluster Node resource failed")
	})

	t.Run("upgrade node success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return &corev1.Node{Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.28.0"}}}, nil
		})
		patches.ApplyFunc(
			(*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster,
				_ string, _ confv1beta1.NodeState, _ string,
			) error {
				return nil
			})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "upgradeNode", func(_ *EnsureWorkerUpgrade, _ confv1beta1.Node, _ *corev1.Node, _ *kubedrain.Helper) error {
			return nil
		})
		params := ProcessNodeUpgradeParams{
			Ctx:              e.Ctx.Context,
			Client:           e.Ctx.Client,
			BKECluster:       e.Ctx.BKECluster,
			NeedUpgradeNodes: bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")},
			Log:              e.Ctx.Log,
		}
		_, failed, err := e.processNodeUpgrade(params)
		require.NoError(t, err)
		assert.Empty(t, failed)
	},
	)

	t.Run("upgrade node failure records failed node", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return &corev1.Node{Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.28.0"}}}, nil
		})
		patches.ApplyFunc(
			(*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster,
				_ string, _ confv1beta1.NodeState, _ string,
			) error {
				return nil
			})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return nil
		})
		patches.ApplyPrivateMethod(e, "upgradeNode", func(_ *EnsureWorkerUpgrade, _ confv1beta1.Node, _ *corev1.Node, _ *kubedrain.Helper) error {
			return assertErr("upgrade boom")
		})
		params := ProcessNodeUpgradeParams{
			Ctx:              e.Ctx.Context,
			Client:           e.Ctx.Client,
			BKECluster:       e.Ctx.BKECluster,
			NeedUpgradeNodes: bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")},
			Log:              e.Ctx.Log,
		}
		_, failed, err := e.processNodeUpgrade(params)
		require.NoError(t, err) // failure is recorded, not propagated
		assert.Len(t, failed, 1)
	},
	)

	t.Run("SyncStatusUntilComplete error after mark upgrading", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.GetRemoteNodeByBKENode, func(_ context.Context, _ *kubernetes.Clientset, _ confv1beta1.Node) (*corev1.Node, error) {
			return &corev1.Node{Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.28.0"}}}, nil
		})
		patches.ApplyFunc(
			(*nodeutil.NodeFetcher).SetNodeStateWithMessageForCluster,
			func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster,
				_ string, _ confv1beta1.NodeState, _ string,
			) error {
				return nil
			})
		callCount := 0
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			callCount++
			if callCount == 1 {
				return assertErr("sync boom")
			}
			return nil
		})
		params := ProcessNodeUpgradeParams{
			Ctx:              e.Ctx.Context,
			Client:           e.Ctx.Client,
			BKECluster:       e.Ctx.BKECluster,
			NeedUpgradeNodes: bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")},
			Log:              e.Ctx.Log,
		}
		_, _, err := e.processNodeUpgrade(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sync boom")
	},
	)
}

// ---- rolloutUpgrade (real body via seams) ----

func TestEnsureWorkerUpgradeRolloutUpgrade(t *testing.T) {
	t.Run("prepare error requeues", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "prepareUpgradeNodes", func(_ *EnsureWorkerUpgrade, _ PrepareUpgradeNodesParams) (bkenode.Nodes, error) {
			return nil, assertErr("prepare failed")
		})
		result, err := e.rolloutUpgrade()
		require.Error(t, err)
		assert.True(t, result.Requeue)
		assert.Contains(t, err.Error(), "prepare failed")
	})

	t.Run("happy path all success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "prepareUpgradeNodes", func(_ *EnsureWorkerUpgrade, _ PrepareUpgradeNodesParams) (bkenode.Nodes, error) {
			return bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")}, nil
		})
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.NewDrainer, func(_ context.Context, _ kubernetes.Interface, _ dynamic.Interface, _ bool, _ *bkev1beta1.BKELogger) *kubedrain.Helper {
			return &kubedrain.Helper{}
		})
		patches.ApplyPrivateMethod(e, "processNodeUpgrade", func(_ *EnsureWorkerUpgrade, _ ProcessNodeUpgradeParams) (ctrl.Result, []string, error) {
			return ctrl.Result{}, nil, nil
		})
		result, err := e.rolloutUpgrade()
		require.NoError(t, err)
		assert.False(t, result.Requeue)
	})

	t.Run("some failed nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "prepareUpgradeNodes", func(_ *EnsureWorkerUpgrade, _ PrepareUpgradeNodesParams) (bkenode.Nodes, error) {
			return bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")}, nil
		})
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.NewDrainer, func(_ context.Context, _ kubernetes.Interface, _ dynamic.Interface, _ bool, _ *bkev1beta1.BKELogger) *kubedrain.Helper {
			return &kubedrain.Helper{}
		})
		patches.ApplyPrivateMethod(e, "processNodeUpgrade", func(_ *EnsureWorkerUpgrade, _ ProcessNodeUpgradeParams) (ctrl.Result, []string, error) {
			return ctrl.Result{}, []string{"n1"}, nil
		})
		_, err := e.rolloutUpgrade()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "some nodes upgrade failed")
	})

	t.Run("processNodeUpgrade error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "prepareUpgradeNodes", func(_ *EnsureWorkerUpgrade, _ PrepareUpgradeNodesParams) (bkenode.Nodes, error) {
			return bkenode.Nodes{workerUpgradeNode("10.0.0.1", "n1")}, nil
		})
		patches.ApplyFunc(kube.GetTargetClusterClient, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (*kubernetes.Clientset, dynamic.Interface, error) {
			return &kubernetes.Clientset{}, nil, nil
		})
		patches.ApplyFunc(phaseutil.NewDrainer, func(_ context.Context, _ kubernetes.Interface, _ dynamic.Interface, _ bool, _ *bkev1beta1.BKELogger) *kubedrain.Helper {
			return &kubedrain.Helper{}
		})
		patches.ApplyPrivateMethod(e, "processNodeUpgrade", func(_ *EnsureWorkerUpgrade, _ ProcessNodeUpgradeParams) (ctrl.Result, []string, error) {
			return ctrl.Result{}, nil, assertErr("process boom")
		})
		_, err := e.rolloutUpgrade()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "process boom")
	})
}

// ---- executeNodeUpgrade (real body via command seams) ----

func TestEnsureWorkerUpgradeExecuteNodeUpgrade(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error { return nil })
		patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
			return nil, []string{"n1"}, nil
		})
		params := ExecuteNodeUpgradeParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Scheme:     e.Ctx.Scheme,
			Node:       workerUpgradeNode("10.0.0.1", "n1"),
			Log:        e.Ctx.Log,
		}
		require.NoError(t, e.executeNodeUpgrade(params))
	})

	t.Run("New error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error { return assertErr("create failed") })
		params := ExecuteNodeUpgradeParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Scheme:     e.Ctx.Scheme,
			Node:       workerUpgradeNode("10.0.0.1", "n1"),
			Log:        e.Ctx.Log,
		}
		err := e.executeNodeUpgrade(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "create upgrade command")
	})

	t.Run("Wait error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error { return nil })
		patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
			return assertErr("wait failed"), nil, nil
		})
		params := ExecuteNodeUpgradeParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Scheme:     e.Ctx.Scheme,
			Node:       workerUpgradeNode("10.0.0.1", "n1"),
			Log:        e.Ctx.Log,
		}
		err := e.executeNodeUpgrade(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "wait upgrade command complete failed")
	})
}

// ---- waitForNodeHealth (real body via seams) ----

func TestEnsureWorkerUpgradeWaitForNodeHealth(t *testing.T) {
	t.Run("remote client error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return nil, assertErr("dial failed")
		})
		params := WaitForNodeHealthParams{
			Ctx:        e.Ctx.Context,
			Client:     e.Ctx.Client,
			BKECluster: e.Ctx.BKECluster,
			Node:       workerUpgradeNode("10.0.0.1", "n1"),
			Log:        e.Ctx.Log,
		}
		err := e.waitForNodeHealth(params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "get remote client")
	})
}

// ---- upgradeNode (real body via seams) ----

func TestEnsureWorkerUpgradeUpgradeNode(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "executeNodeUpgrade", func(_ *EnsureWorkerUpgrade, _ ExecuteNodeUpgradeParams) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForNodeHealth", func(_ *EnsureWorkerUpgrade, _ WaitForNodeHealthParams) error { return nil })
		require.NoError(t, e.upgradeNode(workerUpgradeNode("10.0.0.1", "n1"), &corev1.Node{}, nil))
	})

	t.Run("execute error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "executeNodeUpgrade", func(_ *EnsureWorkerUpgrade, _ ExecuteNodeUpgradeParams) error {
			return assertErr("execute boom")
		})
		err := e.upgradeNode(workerUpgradeNode("10.0.0.1", "n1"), &corev1.Node{}, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "execute boom")
	})

	t.Run("wait health error propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newWorkerUpgradeCov(t, workerUpgradeCovCluster("v1.29.0", "v1.28.0"))
		patches.ApplyPrivateMethod(e, "executeNodeUpgrade", func(_ *EnsureWorkerUpgrade, _ ExecuteNodeUpgradeParams) error { return nil })
		patches.ApplyPrivateMethod(e, "waitForNodeHealth", func(_ *EnsureWorkerUpgrade, _ WaitForNodeHealthParams) error {
			return assertErr("health boom")
		})
		err := e.upgradeNode(workerUpgradeNode("10.0.0.1", "n1"), &corev1.Node{}, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "health boom")
	})
}

// ---- waitForWorkerNodeHealthCheck (additional branches) ----

func TestWaitForWorkerNodeHealthCheckCov(t *testing.T) {
	t.Run("node get error retries until timeout", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(time.Sleep, func(_ time.Duration) {})
		// canceled context -> wait.Poll returns immediately
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		params := WaitForWorkerNodeHealthCheckParams{
			Ctx:        ctx,
			ClientSet:  fake.NewSimpleClientset(),
			Node:       confv1beta1.Node{Hostname: "missing"},
			K8sVersion: "v1.29.0",
			Logger:     createTestLogger(),
		}
		err := waitForWorkerNodeHealthCheck(params)
		require.Error(t, err)
	})

	t.Run("healthy node returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		node := &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "n1"},
			Status: corev1.NodeStatus{
				NodeInfo: corev1.NodeSystemInfo{KubeletVersion: "v1.29.0"},
				Conditions: []corev1.NodeCondition{
					{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
				},
			},
		}
		cs := fake.NewSimpleClientset(node)
		// Patch RemoteClient.NodeHealthCheck to return nil (healthy).
		// waitForWorkerNodeHealthCheck takes a kube.RemoteKubeClient value (not pointer),
		// so we patch the concrete *kube.Client method instead.
		patches.ApplyMethod(&kube.Client{}, "NodeHealthCheck", func(_ *kube.Client, _ *corev1.Node, _ string, _ *bkelog.Logger) error {
			return nil
		})
		params := WaitForWorkerNodeHealthCheckParams{
			Ctx:          context.Background(),
			ClientSet:    cs,
			RemoteClient: &kube.Client{},
			Node:         confv1beta1.Node{Hostname: "n1"},
			K8sVersion:   "v1.29.0",
			Logger:       createTestLogger(),
		}
		require.NoError(t, waitForWorkerNodeHealthCheck(params))
	})
}
