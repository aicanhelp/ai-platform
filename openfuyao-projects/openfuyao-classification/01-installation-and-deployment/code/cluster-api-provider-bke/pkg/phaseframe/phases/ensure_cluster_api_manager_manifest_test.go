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
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func clusterAPIManagerTestCluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					ImageRepo: confv1beta1.Repo{Domain: "repo.example.com", Port: "443"},
				},
				Addons: []confv1beta1.Product{
					{Name: "cluster-api", Version: "v1.6.0"},
				},
			},
		},
	}
}

func newClusterAPIManagerPhase(t *testing.T, cluster *bkev1beta1.BKECluster) *EnsureClusterAPIManagerManifest {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cluster).Build()
	pc := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: cluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, cluster),
	}
	return NewEnsureClusterAPIManagerManifest(pc).(*EnsureClusterAPIManagerManifest)
}

func TestEnsureClusterAPIManagerManifestConstants(t *testing.T) {
	assert.Equal(t, "EnsureClusterAPIManagerManifest", string(EnsureClusterAPIManagerManifestName))
	found := false
	target := reflect.ValueOf(NewEnsureClusterAPIManagerManifest).Pointer()
	for _, regFn := range PostDeployPhases() {
		if reflect.ValueOf(regFn).Pointer() == target {
			found = true
			break
		}
	}
	assert.True(t, found, "PostDeployPhases should register NewEnsureClusterAPIManagerManifest")
}

func TestNewEnsureClusterAPIManagerManifest(t *testing.T) {
	cluster := clusterAPIManagerTestCluster()
	phase := newClusterAPIManagerPhase(t, cluster)
	assert.NotNil(t, phase)
	assert.Equal(t, EnsureClusterAPIManagerManifestName, phase.Name())
}

func TestFindAddonVersion(t *testing.T) {
	tests := []struct {
		name      string
		cluster   *bkev1beta1.BKECluster
		addonName string
		wantVer   string
		wantOK    bool
	}{
		{name: "nil cluster", cluster: nil, addonName: "cluster-api", wantOK: false},
		{
			name: "nil cluster config",
			cluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{ClusterConfig: nil},
			},
			addonName: "cluster-api",
			wantOK:    false,
		},
		{name: "found", cluster: clusterAPIManagerTestCluster(), addonName: "cluster-api", wantVer: "v1.6.0", wantOK: true},
		{name: "missing addon", cluster: clusterAPIManagerTestCluster(), addonName: "other", wantOK: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotVer, gotOK := findAddonVersion(tt.cluster, tt.addonName)
			assert.Equal(t, tt.wantOK, gotOK)
			if gotOK {
				assert.Equal(t, tt.wantVer, gotVer)
			}
		})
	}
}

func TestFindAddon(t *testing.T) {
	cluster := clusterAPIManagerTestCluster()
	addon, ok := findAddon(cluster, "cluster-api")
	require.True(t, ok)
	require.NotNil(t, addon)
	assert.Equal(t, "cluster-api", addon.Name)
	assert.Equal(t, "v1.6.0", addon.Version)

	_, ok = findAddon(cluster, "missing")
	assert.False(t, ok)
	_, ok = findAddon(nil, "cluster-api")
	assert.False(t, ok)
}

func TestEnsureClusterAPIManagerManifest_NeedExecute(t *testing.T) {
	cluster := clusterAPIManagerTestCluster()

	t.Run("default need execute false", func(t *testing.T) {
		phase := newClusterAPIManagerPhase(t, cluster)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultNeedExecute", func(_ *phaseframe.BasePhase, _, _ *bkev1beta1.BKECluster) bool {
			return false
		})
		assert.False(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("no cluster-api addon", func(t *testing.T) {
		noAddon := cluster.DeepCopy()
		noAddon.Spec.ClusterConfig.Addons = nil
		phase := newClusterAPIManagerPhase(t, noAddon)
		assert.False(t, phase.NeedExecute(nil, noAddon))
	})

	t.Run("empty addon version", func(t *testing.T) {
		emptyVer := cluster.DeepCopy()
		emptyVer.Spec.ClusterConfig.Addons[0].Version = "  "
		phase := newClusterAPIManagerPhase(t, emptyVer)
		assert.False(t, phase.NeedExecute(nil, emptyVer))
	})

	t.Run("already applied", func(t *testing.T) {
		applied := cluster.DeepCopy()
		annotation.SetAnnotation(applied, common.ClusterAPIManagerAppliedAnnotationKey, "true")
		phase := newClusterAPIManagerPhase(t, applied)
		assert.False(t, phase.NeedExecute(nil, applied))
	})

	t.Run("postprocess not ready waits", func(t *testing.T) {
		waiting := cluster.DeepCopy()
		phase := newClusterAPIManagerPhase(t, waiting)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(condition.HasConditionStatus, func(_ confv1beta1.ClusterConditionType, _ *bkev1beta1.BKECluster, status confv1beta1.ConditionStatus) bool {
			return status == confv1beta1.ConditionTrue
		})
		assert.True(t, phase.NeedExecute(nil, waiting))
		assert.Equal(t, bkev1beta1.PhaseWaiting, phase.GetStatus())
	})

	t.Run("postprocess ready waits for execute", func(t *testing.T) {
		ready := cluster.DeepCopy()
		phase := newClusterAPIManagerPhase(t, ready)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(condition.HasConditionStatus, func(cond confv1beta1.ClusterConditionType, _ *bkev1beta1.BKECluster, status confv1beta1.ConditionStatus) bool {
			return cond == bkev1beta1.NodesPostProcessCondition && status == confv1beta1.ConditionTrue
		})
		assert.True(t, phase.NeedExecute(nil, ready))
		assert.Equal(t, bkev1beta1.PhaseWaiting, phase.GetStatus())
	})
}

type manifestStubRemoteClient struct{}

func (manifestStubRemoteClient) InstallAddon(*bkev1beta1.BKECluster, *bkeaddon.AddonTransfer, *kube.AddonRecorder, client.Client, bkenode.Nodes) error {
	return nil
}
func (manifestStubRemoteClient) ApplyYaml(*kube.Task) error { return nil }
func (manifestStubRemoteClient) NewK8sToken() (string, error) {
	return "", nil
}
func (manifestStubRemoteClient) KubeClient() (*kubernetes.Clientset, dynamic.Interface) {
	return nil, nil
}
func (manifestStubRemoteClient) Collect() (*kube.CollectResult, []error, []error) {
	return nil, nil, nil
}
func (manifestStubRemoteClient) CheckClusterHealth(*bkev1beta1.BKECluster, string, bkev1beta1.BKENodes) error {
	return nil
}
func (manifestStubRemoteClient) NodeHealthCheck(*corev1.Node, string, *log.Logger) error {
	return nil
}
func (manifestStubRemoteClient) CheckComponentHealth(*corev1.Node) error { return nil }
func (manifestStubRemoteClient) ListNodes(*metav1.ListOptions) (*corev1.NodeList, error) {
	return nil, nil
}
func (manifestStubRemoteClient) GetPod(string, string) (*corev1.Pod, error) { return nil, nil }
func (manifestStubRemoteClient) SetLogger(*log.Logger)                      {}
func (manifestStubRemoteClient) SetBKELogger(*bkev1beta1.BKELogger)         {}

func TestEnsureClusterAPIManagerManifest_Execute(t *testing.T) {
	cluster := clusterAPIManagerTestCluster()

	t.Run("node fetch error", func(t *testing.T) {
		phase := newClusterAPIManagerPhase(t, cluster)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assert.AnError
		})
		_, err := phase.Execute()
		require.Error(t, err)
	})

	t.Run("postprocess not finished requeues", func(t *testing.T) {
		phase := newClusterAPIManagerPhase(t, cluster)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.1"}}}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		result, err := phase.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "postprocess not finished")
		assert.Equal(t, 10*time.Second, result.RequeueAfter)
	})

	t.Run("no addon version exits cleanly", func(t *testing.T) {
		noVer := cluster.DeepCopy()
		noVer.Spec.ClusterConfig.Addons = nil
		phase := newClusterAPIManagerPhase(t, noVer)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{}
		})
		result, err := phase.Execute()
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("remote client error", func(t *testing.T) {
		phase := newClusterAPIManagerPhase(t, cluster)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mockExecutePreconditions(t, patches)
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return nil, assert.AnError
		})
		_, err := phase.Execute()
		require.Error(t, err)
	})

	t.Run("unsupported remote client type", func(t *testing.T) {
		phase := newClusterAPIManagerPhase(t, cluster)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mockExecutePreconditions(t, patches)
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return manifestStubRemoteClient{}, nil
		})
		_, err := phase.Execute()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported remote client type")
	})

	t.Run("success applies manifest", func(t *testing.T) {
		phase := newClusterAPIManagerPhase(t, cluster)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		mockExecutePreconditions(t, patches)

		remoteClient := &kube.Client{}
		patches.ApplyFunc(kube.NewRemoteClientByBKECluster, func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster) (kube.RemoteKubeClient, error) {
			return remoteClient, nil
		})
		patches.ApplyMethod(remoteClient, "PrepareRenderParamForAddonFile", func(_ *kube.Client, _ *bkev1beta1.BKECluster, _ *confv1beta1.Product, _ string, _ string, _ bkenode.Nodes) (map[string]interface{}, error) {
			return map[string]interface{}{"key": "value"}, nil
		})
		patches.ApplyMethod(remoteClient, "ApplyYaml", func(_ *kube.Client, _ *kube.Task) error {
			return nil
		})
		patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return nil
		})

		result, err := phase.Execute()
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
		v, ok := annotation.HasAnnotation(cluster, common.ClusterAPIManagerAppliedAnnotationKey)
		require.True(t, ok)
		assert.Equal(t, "true", v)
	})
}

func mockExecutePreconditions(t *testing.T, patches *gomonkey.Patches) {
	t.Helper()
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
		return bkev1beta1.BKENodes{}, nil
	})
	patches.ApplyFunc(phaseutil.GetNeedPostProcessNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
		return bkenode.Nodes{}
	})
	patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
		return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "10.0.0.1"}}}, nil
	})
}
