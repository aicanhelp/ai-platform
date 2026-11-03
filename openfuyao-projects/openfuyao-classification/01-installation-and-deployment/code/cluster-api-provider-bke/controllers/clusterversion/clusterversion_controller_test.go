/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 ******************************************************************/

package clusterversion

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	cvensure "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/clusterversion"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func TestClusterVersionReconcilerMarksBKEClusterUpgradeReady(t *testing.T) {
	reconciler := newTestReconciler(t,
		testBKECluster("default", "cluster-a", "v2.5.0"),
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec: cvv1beta1.ClusterVersionSpec{
				DesiredVersion: "v2.6.0",
			},
			Status: cvv1beta1.ClusterVersionStatus{
				CurrentVersion: "v2.5.0",
			},
		},
		&cvv1beta1.ReleaseImage{
			ObjectMeta: objectMeta("default", "release-v2.6.0"),
			Spec: cvv1beta1.ReleaseImageSpec{
				Version: "v2.6.0",
			},
			Status: cvv1beta1.ReleaseImageStatus{
				Phase: cvv1beta1.ReleaseImagePhaseValid,
			},
		},
		&cvv1beta1.UpgradePath{
			ObjectMeta: objectMeta("", "openfuyao-upgrade-paths"),
			Spec: cvv1beta1.UpgradePathSpec{
				Paths: []cvv1beta1.UpgradePathRule{{From: "v2.5.0", To: "v2.6.0"}},
				Versions: []cvv1beta1.VersionEntry{{
					Version: "v2.6.0",
				}},
			},
			Status: cvv1beta1.UpgradePathStatus{Phase: cvv1beta1.UpgradePathPhaseActive},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})

	require.NoError(t, err)
	gotCluster := &bkev1beta1.BKECluster{}
	require.NoError(t, reconciler.Get(context.Background(),
		types.NamespacedName{Namespace: "default", Name: "cluster-a"}, gotCluster))
	assert.Equal(t, "v2.6.0", gotCluster.Annotations[annotation.CVOUpgradeReadyAnnotationKey])
	assert.Equal(t, "cluster-a", gotCluster.Annotations[annotation.CVOClusterVersionAnnotationKey])

	gotCV := &cvv1beta1.ClusterVersion{}
	require.NoError(t, reconciler.Get(context.Background(),
		types.NamespacedName{Namespace: "default", Name: "cluster-a"}, gotCV))
	assert.Equal(t, cvv1beta1.ClusterVersionPhasePreChecking, gotCV.Status.Phase)
	assert.Equal(t, "v2.5.0", gotCV.Status.CurrentVersion)
}

func TestClusterVersionReconcilerBlocksWhenOnlyPathIsBlocked(t *testing.T) {
	reconciler := newTestReconciler(t,
		testBKECluster("default", "cluster-a", "v2.5.0"),
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
			Status:     cvv1beta1.ClusterVersionStatus{CurrentVersion: "v2.5.0"},
		},
		&cvv1beta1.UpgradePath{
			ObjectMeta: objectMeta("", "openfuyao-upgrade-paths"),
			Spec: cvv1beta1.UpgradePathSpec{
				Paths: []cvv1beta1.UpgradePathRule{{
					From: "v2.5.0", To: "v2.6.0", Blocked: true, Notes: "manual gate",
				}},
			},
			Status: cvv1beta1.UpgradePathStatus{Phase: cvv1beta1.UpgradePathPhaseActive},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})

	require.NoError(t, err)
	gotCluster := &bkev1beta1.BKECluster{}
	require.NoError(t, reconciler.Get(context.Background(),
		types.NamespacedName{Namespace: "default", Name: "cluster-a"}, gotCluster))
	assert.NotContains(t, gotCluster.Annotations, annotation.CVOUpgradeReadyAnnotationKey)

	gotCV := &cvv1beta1.ClusterVersion{}
	require.NoError(t, reconciler.Get(context.Background(),
		types.NamespacedName{Namespace: "default", Name: "cluster-a"}, gotCV))
	assert.Equal(t, cvv1beta1.ClusterVersionPhasePreCheckFailed, gotCV.Status.Phase)
}

func TestIsInstallPhaseWhenCurrentEqualsDesiredWithoutReleaseImage(t *testing.T) {
	reconciler := newTestReconciler(t)
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Status: confv1beta1.BKEClusterStatus{
			OpenFuyaoVersion: "v2.5.0",
			ClusterStatus:    confv1beta1.ClusterStatus("Installing"),
		},
	}
	cv := &cvv1beta1.ClusterVersion{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.5.0"},
		Status:     cvv1beta1.ClusterVersionStatus{CurrentVersion: "v2.5.0"},
	}
	assert.True(t, reconciler.isInstallPhase(context.Background(), bc, cv))
}

func TestIsInstallPhaseWhenClusterReadyWithoutReleaseImage(t *testing.T) {
	reconciler := newTestReconciler(t)
	bc := testBKECluster("default", "cluster-a", "v26.05")
	cv := &cvv1beta1.ClusterVersion{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v26.05"},
		Status:     cvv1beta1.ClusterVersionStatus{CurrentVersion: "v26.05"},
	}
	assert.False(t, reconciler.isInstallPhase(context.Background(), bc, cv))
}

func TestClusterVersionReconcilerDoesNotRecreateReleaseImageAfterInstall(t *testing.T) {
	reconciler := newTestReconciler(t,
		testBKECluster("default", "cluster-a", "v26.05"),
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v26.05"},
			Status: cvv1beta1.ClusterVersionStatus{
				CurrentVersion: "v26.05",
				Phase:          cvv1beta1.ClusterVersionPhaseReady,
			},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})
	require.NoError(t, err)

	list := &cvv1beta1.ReleaseImageList{}
	require.NoError(t, reconciler.List(context.Background(), list, client.InNamespace("default")))
	assert.Empty(t, list.Items)
}

func TestIsInstallPhaseWhenCurrentEqualsDesiredWithReleaseImage(t *testing.T) {
	reconciler := newTestReconciler(t, &cvv1beta1.ReleaseImage{
		ObjectMeta: objectMeta("default", "release-v2.5.0"),
		Spec:       cvv1beta1.ReleaseImageSpec{Version: "v2.5.0"},
	})
	bc := testBKECluster("default", "cluster-a", "v2.5.0")
	cv := &cvv1beta1.ClusterVersion{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.5.0"},
		Status:     cvv1beta1.ClusterVersionStatus{CurrentVersion: "v2.5.0"},
	}
	assert.False(t, reconciler.isInstallPhase(context.Background(), bc, cv))
}

func TestIsInstallPhaseWhenClusterNotReadyAndCurrentEmpty(t *testing.T) {
	reconciler := newTestReconciler(t)
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Status:     confv1beta1.BKEClusterStatus{ClusterStatus: confv1beta1.ClusterStatus("Installing")},
	}
	cv := &cvv1beta1.ClusterVersion{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.5.0"},
	}
	assert.True(t, reconciler.isInstallPhase(context.Background(), bc, cv))
}

func TestSetStatusOnlyPatchesProvidedFields(t *testing.T) {
	cv := &cvv1beta1.ClusterVersion{
		ObjectMeta: objectMeta("default", "cluster-a"),
		Status: cvv1beta1.ClusterVersionStatus{
			CurrentVersion: "v2.5.0",
			Phase:          cvv1beta1.ClusterVersionPhaseReady,
		},
	}
	reconciler := newTestReconciler(t, cv)

	err := reconciler.setStatus(context.Background(), cv, statusPatch{
		Phase: phasePtr(cvv1beta1.ClusterVersionPhasePreChecking),
	})
	require.NoError(t, err)

	got := &cvv1beta1.ClusterVersion{}
	require.NoError(t, reconciler.Get(context.Background(),
		types.NamespacedName{Namespace: "default", Name: "cluster-a"}, got))
	assert.Equal(t, "v2.5.0", got.Status.CurrentVersion)
	assert.Equal(t, cvv1beta1.ClusterVersionPhasePreChecking, got.Status.Phase)
}

func TestClusterVersionReconcilerNotFoundReturnsNil(t *testing.T) {
	reconciler := newTestReconciler(t)
	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "missing"},
	})
	require.NoError(t, err)
}

func TestClusterVersionReconcilerClearsUpgradeReadyWhenDesiredEqualsCurrent(t *testing.T) {
	cluster := testBKECluster("default", "cluster-a", "v2.5.0")
	cluster.Annotations = map[string]string{
		annotation.CVOUpgradeReadyAnnotationKey: "v2.6.0",
		annotation.CVOUpgradePathAnnotationKey:  "v2.5.0->v2.6.0",
	}
	reconciler := newTestReconciler(t,
		cluster,
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.5.0"},
			Status:     cvv1beta1.ClusterVersionStatus{CurrentVersion: "v2.5.0"},
		},
		&cvv1beta1.ReleaseImage{
			ObjectMeta: objectMeta("default", "release-v2.5.0"),
			Spec:       cvv1beta1.ReleaseImageSpec{Version: "v2.5.0"},
			Status:     cvv1beta1.ReleaseImageStatus{Phase: cvv1beta1.ReleaseImagePhaseValid},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})
	require.NoError(t, err)

	gotCluster := &bkev1beta1.BKECluster{}
	require.NoError(t, reconciler.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "cluster-a"}, gotCluster))
	assert.NotContains(t, gotCluster.Annotations, annotation.CVOUpgradeReadyAnnotationKey)
	assert.NotContains(t, gotCluster.Annotations, annotation.CVOUpgradePathAnnotationKey)
}

func TestClusterVersionReconcilerSkipsUpgradeWhenPhaseFailed(t *testing.T) {
	reconciler := newTestReconciler(t,
		testBKECluster("default", "cluster-a", "v2.5.0"),
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
			Status: cvv1beta1.ClusterVersionStatus{
				CurrentVersion: "v2.5.0",
				Phase:          cvv1beta1.ClusterVersionPhaseFailed,
			},
		},
		&cvv1beta1.ReleaseImage{
			ObjectMeta: objectMeta("default", "release-v2.6.0"),
			Spec:       cvv1beta1.ReleaseImageSpec{Version: "v2.6.0"},
			Status:     cvv1beta1.ReleaseImageStatus{Phase: cvv1beta1.ReleaseImagePhaseValid},
		},
		&cvv1beta1.UpgradePath{
			ObjectMeta: objectMeta("", "openfuyao-upgrade-paths"),
			Spec: cvv1beta1.UpgradePathSpec{
				Paths:    []cvv1beta1.UpgradePathRule{{From: "v2.5.0", To: "v2.6.0"}},
				Versions: []cvv1beta1.VersionEntry{{Version: "v2.6.0"}},
			},
			Status: cvv1beta1.UpgradePathStatus{Phase: cvv1beta1.UpgradePathPhaseActive},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})
	require.NoError(t, err)

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, reconciler.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "cluster-a"}, got))
	assert.NotContains(t, got.Annotations, annotation.CVOUpgradeReadyAnnotationKey)
}

func TestClusterVersionReconcilerSkipsUpgradeWhenCurrentVersionMissing(t *testing.T) {
	reconciler := newTestReconciler(t,
		&bkev1beta1.BKECluster{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Status: confv1beta1.BKEClusterStatus{
				ClusterStatus: bkev1beta1.ClusterStatusReady,
			},
		},
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})
	require.NoError(t, err)

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, reconciler.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "cluster-a"}, got))
	assert.NotContains(t, got.Annotations, annotation.CVOUpgradeReadyAnnotationKey)
}

func TestClusterVersionReconcilerPrecheckFailsWhenReleaseImageInvalid(t *testing.T) {
	reconciler := newTestReconciler(t,
		testBKECluster("default", "cluster-a", "v2.5.0"),
		&cvv1beta1.ClusterVersion{
			ObjectMeta: objectMeta("default", "cluster-a"),
			Spec:       cvv1beta1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
			Status:     cvv1beta1.ClusterVersionStatus{CurrentVersion: "v2.5.0"},
		},
		&cvv1beta1.ReleaseImage{
			ObjectMeta: objectMeta("default", "release-v2.6.0"),
			Spec:       cvv1beta1.ReleaseImageSpec{Version: "v2.6.0"},
			Status: cvv1beta1.ReleaseImageStatus{
				Phase:   cvv1beta1.ReleaseImagePhaseInvalid,
				Message: "bad manifest",
			},
		},
		&cvv1beta1.UpgradePath{
			ObjectMeta: objectMeta("", "openfuyao-upgrade-paths"),
			Spec: cvv1beta1.UpgradePathSpec{
				Paths: []cvv1beta1.UpgradePathRule{{From: "v2.5.0", To: "v2.6.0"}},
				Versions: []cvv1beta1.VersionEntry{{
					Version: "v2.6.0",
				}},
			},
			Status: cvv1beta1.UpgradePathStatus{Phase: cvv1beta1.UpgradePathPhaseActive},
		},
	)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "cluster-a"},
	})
	require.Error(t, err)

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, reconciler.Get(context.Background(), types.NamespacedName{Namespace: "default", Name: "cluster-a"}, got))
	assert.NotContains(t, got.Annotations, annotation.CVOUpgradeReadyAnnotationKey)
}

func TestClusterVersionHelpers(t *testing.T) {
	assert.Equal(t, "release-v26.03", cvensure.ReleaseImageRefForVersion(" v26.03 "))
	assert.Equal(t, "a", firstNonEmpty(" ", "a", "b"))
	assert.Empty(t, firstNonEmpty(" ", ""))
	assert.Equal(t, "v2.5.0->v2.6.0", formatPath([]cvv1beta1.UpgradePathRule{{From: "v2.5.0", To: "v2.6.0"}}))
	assert.Equal(t, "spec", cvensure.OpenFuyaoVersionForBKECluster(&bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "spec"}}},
	}))
}

func newTestReconciler(t *testing.T, objs ...client.Object) *ClusterVersionReconciler {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	require.NoError(t, cvv1beta1.AddToScheme(scheme))

	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(&cvv1beta1.ClusterVersion{}, &cvv1beta1.ReleaseImage{}).
		Build()

	return &ClusterVersionReconciler{
		Client: c,
		Scheme: scheme,
	}
}

func testBKECluster(namespace, name, version string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: objectMeta(namespace, name),
		Status: confv1beta1.BKEClusterStatus{
			OpenFuyaoVersion: version,
			ClusterStatus:    bkev1beta1.ClusterStatusReady,
		},
	}
}

func objectMeta(namespace, name string) metav1.ObjectMeta {
	return metav1.ObjectMeta{Namespace: namespace, Name: name}
}
