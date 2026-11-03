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

package capbke

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/featuregate"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestShouldUseDeclarativeUpgrade(t *testing.T) {
	r := &BKEClusterReconciler{}

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				featuregate.UpgradeReadyAnnotationKey: "v2.5.0",
			},
		},
	}
	if !r.shouldUseDeclarativeUpgrade(bc) {
		t.Fatal("expected declarative path when upgrade-ready is set")
	}

	if r.shouldUseDeclarativeUpgrade(&bkev1beta1.BKECluster{}) {
		t.Fatal("expected no declarative path without upgrade-ready")
	}
}

func TestIsReleaseImageNotReady(t *testing.T) {
	if !isReleaseImageNotReady(&releaseImagePendingError{msg: "pending"}) {
		t.Fatal("expected pending detection")
	}
	if isReleaseImageNotReady(fmt.Errorf("other")) {
		t.Fatal("unexpected pending detection")
	}
}

func TestReleaseRefFromCRUsesVersionCacheKeyWhenSpecDigestEmpty(t *testing.T) {
	ri := &cvv1alpha1.ReleaseImage{
		Spec: cvv1alpha1.ReleaseImageSpec{
			Version: "openfuyao-v26.06",
		},
		Status: cvv1alpha1.ReleaseImageStatus{
			Digest: "sha256:d286a2c213244ef9b4581b2464c60a6198966c7a72764f1c9a1736f391394b8a",
		},
	}

	ref := releaseRefFromCR(ri)
	assert.Equal(t, "openfuyao-v26.06", ref.Version)
	assert.Empty(t, ref.Digest)
	assert.Equal(t, "openfuyao-v26.06", ref.CacheKey())
}

func TestReleaseRefFromCRUsesSpecDigestCacheKeyWhenSet(t *testing.T) {
	ri := &cvv1alpha1.ReleaseImage{
		Spec: cvv1alpha1.ReleaseImageSpec{
			Version: "v26.06",
			Digest:  "sha256:abc",
		},
	}

	ref := releaseRefFromCR(ri)
	assert.Equal(t, "sha256-abc", ref.CacheKey())
}

func TestResolveCurrentReleaseBundle_SameVersionStillResolvesBundle(t *testing.T) {
	scheme := declarativeUpgradeTestScheme()
	cluster := declarativeUpgradeTestCluster("c1", "ns")
	cluster.ObjectMeta.Annotations[featuregate.UpgradeReadyAnnotationKey] = "v26.06"
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.06",
			Phase:          cvv1alpha1.ClusterVersionPhaseReady,
		},
	}
	ri := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "openfuyao-v26.06", Namespace: "ns"},
		Spec: cvv1alpha1.ReleaseImageSpec{
			Version: "v26.06",
		},
		Status: cvv1alpha1.ReleaseImageStatus{
			Phase: cvv1alpha1.ReleaseImagePhaseValid,
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cluster, cv, ri).
		WithStatusSubresource(&bkev1beta1.BKECluster{}, &cvv1alpha1.ClusterVersion{}, &cvv1alpha1.ReleaseImage{}).
		Build()

	resolved := false
	store := releasemanifest.NewStore("", nil, nil)
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethodFunc(store, "ResolveRelease", func(_ context.Context, ref releasemanifest.ReleaseRef) (*releasemanifest.Bundle, error) {
		resolved = true
		assert.Equal(t, "v26.06", ref.Version)
		return &releasemanifest.Bundle{}, nil
	})

	r := &BKEClusterReconciler{
		Client:       fakeClient,
		ReleaseStore: store,
	}

	bundle, err := r.resolveCurrentReleaseBundle(context.Background(), cluster, "v26.06")
	require.NoError(t, err)
	require.NotNil(t, bundle)
	assert.True(t, resolved, "same-version retrigger should still resolve current bundle")
}

func TestDeclarativeUpgradePhaseName(t *testing.T) {
	node := &topology.ComponentNode{
		Name:   upgrade.ComponentEtcd,
		Inline: &topology.InlineRef{Handler: upgrade.InlineHandlerEtcdUpgrade},
	}
	if got := declarativeUpgradePhaseName(node); string(got) != upgrade.InlineHandlerEtcdUpgrade {
		t.Fatalf("got %s", got)
	}
}

func declarativeUpgradeTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

func declarativeUpgradeTestCluster(name, namespace string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Annotations: map[string]string{
				featuregate.UpgradeReadyAnnotationKey: "v26.06",
			},
		},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{},
		},
		Status: confv1beta1.BKEClusterStatus{},
	}
}

// mockDeclarativeUpgradeSyncStatus applies patch funcs in-memory, runs StatusManager,
// and writes status back to the fake client so later Patch calls do not wipe it.
func mockDeclarativeUpgradeSyncStatus(t *testing.T) *gomonkey.Patches {
	t.Helper()
	return gomonkey.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(c client.Client, bc *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			for _, patch := range patchs {
				if patch != nil {
					patch(bc)
				}
			}
			statusmanage.DefaultStatusManager().SetStatus(bc, nil)
			return refreshBKEClusterStatusOnClient(c, bc)
		})
}

func refreshBKEClusterStatusOnClient(c client.Client, bc *bkev1beta1.BKECluster) error {
	if c == nil || bc == nil {
		return nil
	}
	statusCopy := bc.DeepCopy()
	if err := c.Status().Update(context.Background(), statusCopy); err != nil {
		return err
	}
	fresh := &bkev1beta1.BKECluster{}
	if err := c.Get(context.Background(), client.ObjectKeyFromObject(bc), fresh); err != nil {
		return err
	}
	bc.ObjectMeta = fresh.ObjectMeta
	bc.Status = fresh.Status
	return nil
}

func TestHandleDeclarativeUpgradeDAGFailure_RetriesThenAborts(t *testing.T) {
	const allowed = 2
	origAllowed := statusmanage.ReconcileAllowedFailedCount
	origManager := statusmanage.DefaultStatusManager()
	t.Cleanup(func() {
		statusmanage.ReconcileAllowedFailedCount = origAllowed
		statusmanage.SetDefaultStatusManager(origManager)
	})

	statusmanage.ReconcileAllowedFailedCount = allowed
	statusmanage.SetDefaultStatusManager(statusmanage.NewStatusManager())

	scheme := declarativeUpgradeTestScheme()
	cluster := declarativeUpgradeTestCluster("c1", "ns")
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseUpgrading,
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cluster, cv).
		WithStatusSubresource(&bkev1beta1.BKECluster{}, &cvv1alpha1.ClusterVersion{}).
		Build()

	patches := mockDeclarativeUpgradeSyncStatus(t)
	defer patches.Reset()

	r := &BKEClusterReconciler{Client: fakeClient}
	ctx := context.Background()
	dagErr := errors.New("dag execution failed")

	require.NoError(t, r.patchClusterStatus(cluster, bkev1beta1.ClusterUpgrading))

	for i := 0; i < allowed-1; i++ {
		require.NoError(t, r.handleDeclarativeUpgradeDAGFailure(ctx, cluster, dagErr))
		_, ok := featuregate.UpgradeReady(cluster)
		assert.True(t, ok, "upgrade-ready should remain while retries are allowed")
		assert.True(t, statusmanage.DefaultStatusManager().GetCtrlResult(cluster).Requeue)
	}

	require.NoError(t, r.handleDeclarativeUpgradeDAGFailure(ctx, cluster, dagErr))
	_, ok := featuregate.UpgradeReady(cluster)
	assert.False(t, ok, "upgrade-ready should be cleared after retries are exhausted")

	gotCV := &cvv1alpha1.ClusterVersion{}
	require.NoError(t, fakeClient.Get(ctx, types.NamespacedName{Namespace: "ns", Name: "c1"}, gotCV))
	assert.Equal(t, cvv1alpha1.ClusterVersionPhaseFailed, gotCV.Status.Phase)
	assert.False(t, statusmanage.DefaultStatusManager().GetCtrlResult(cluster).Requeue)
}

func TestAbortDeclarativeUpgrade(t *testing.T) {
	origManager := statusmanage.DefaultStatusManager()
	t.Cleanup(func() {
		statusmanage.SetDefaultStatusManager(origManager)
	})
	statusmanage.SetDefaultStatusManager(statusmanage.NewStatusManager())

	scheme := declarativeUpgradeTestScheme()
	cluster := declarativeUpgradeTestCluster("c1", "ns")
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseUpgrading,
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cluster, cv).
		WithStatusSubresource(&bkev1beta1.BKECluster{}, &cvv1alpha1.ClusterVersion{}).
		Build()

	patches := mockDeclarativeUpgradeSyncStatus(t)
	defer patches.Reset()

	r := &BKEClusterReconciler{Client: fakeClient}
	require.NoError(t, r.abortDeclarativeUpgrade(context.Background(), cluster, fmt.Errorf("boom")))

	_, ok := featuregate.UpgradeReady(cluster)
	assert.False(t, ok)

	gotCV := &cvv1alpha1.ClusterVersion{}
	require.NoError(t, fakeClient.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, gotCV))
	assert.Equal(t, cvv1alpha1.ClusterVersionPhaseFailed, gotCV.Status.Phase)
	assert.Equal(t, bkev1beta1.UpgradeFailed, cluster.Status.ClusterHealthState)
}

func TestCompleteDeclarativeUpgrade_KeepsMemoryCompleted(t *testing.T) {
	scheme := declarativeUpgradeTestScheme()
	now := metav1.Now()
	cluster := declarativeUpgradeTestCluster("c1", "ns")
	cluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion = "v26.05"
	cluster.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.34.3"
	cluster.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{
		TargetVersion: "v26.06",
		StartedAt:     &now,
		Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
			{Name: "etcd", Version: "v3.6.7", CompletedAt: now},
			{Name: "kube-master", Version: "v1.34.3", CompletedAt: now},
			{Name: "calico", Version: "v3.28.0", CompletedAt: now},
			{Name: "coredns", Version: "v1.12.3", CompletedAt: now},
			{Name: "kubeproxy", Version: "v1.34.4", CompletedAt: now},
		},
	}

	// API object intentionally lags the last DAG batch (informer-stale Completed).
	apiCluster := cluster.DeepCopy()
	apiCluster.Status.DeclarativeUpgrade = &confv1beta1.DeclarativeUpgradeStatus{
		TargetVersion: "v26.06",
		StartedAt:     &now,
		Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{
			{Name: "etcd", Version: "v3.6.7", CompletedAt: now},
			{Name: "kube-master", Version: "v1.34.3", CompletedAt: now},
		},
	}

	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseUpgrading,
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(apiCluster, cv).
		WithStatusSubresource(&bkev1beta1.BKECluster{}, &cvv1alpha1.ClusterVersion{}).
		Build()

	r := &BKEClusterReconciler{Client: fakeClient}
	require.NoError(t, r.completeDeclarativeUpgrade(context.Background(), cluster))

	_, ok := featuregate.UpgradeReady(cluster)
	assert.False(t, ok)

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, fakeClient.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got))
	require.NotNil(t, got.Status.DeclarativeUpgrade)
	require.NotNil(t, got.Status.DeclarativeUpgrade.FinishedAt)
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("calico", "v3.28.0"))
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("coredns", "v1.12.3"))
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("kubeproxy", "v1.34.4"))
	assert.Len(t, got.Status.DeclarativeUpgrade.Completed, 5)
	assert.Equal(t, "v26.06", got.Status.OpenFuyaoVersion)
}
