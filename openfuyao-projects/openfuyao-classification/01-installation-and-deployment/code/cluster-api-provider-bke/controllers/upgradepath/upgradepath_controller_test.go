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

package upgradepath

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	pathstore "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgradepath"
)

func newUpgradePathTestReconciler(objs ...client.Object) (*UpgradePathReconciler, *pathstore.Service) {
	scheme := runtime.NewScheme()
	_ = upv1alpha1.AddToScheme(scheme)

	builder := fake.NewClientBuilder().WithScheme(scheme)
	if len(objs) > 0 {
		builder = builder.WithObjects(objs...).WithStatusSubresource(objs...)
	}

	service := pathstore.NewService()
	return &UpgradePathReconciler{
		Client:      builder.Build(),
		Scheme:      scheme,
		PathService: service,
	}, service
}

func testUpgradePathAnnotations(digest string) map[string]string {
	if digest == "" {
		return nil
	}
	return map[string]string{pathstore.OCIDigestAnnotation: digest}
}

func TestReconcileLoadsValidUpgradePath(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "paths",
			Annotations: testUpgradePathAnnotations("sha256:test"),
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
				{From: "v1.1.0", To: "v1.2.0"},
			},
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v1.0.0", Installable: true},
				{Version: "v1.1.0", Installable: true},
				{Version: "v1.2.0", Installable: true},
			},
		},
	}
	reconciler, service := newUpgradePathTestReconciler(up)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)

	path, err := service.FindPath("v1.0.0", "v1.2.0")
	require.NoError(t, err)
	assert.Len(t, path, 2)

	installable := service.GetInstallableVersions()
	assert.Len(t, installable, 3)

	current := &upv1alpha1.UpgradePath{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{Name: "paths"}, current))
	assert.Equal(t, upv1alpha1.UpgradePathPhaseActive, current.Status.Phase)
	assert.Equal(t, 2, current.Status.PathCount)
	assert.Equal(t, "sha256:test", current.Status.LastDigest)
	assert.NotNil(t, current.Status.LastCheckedAt)
	assert.Len(t, current.Status.Conditions, 1)
	assert.Equal(t, "Validated", current.Status.Conditions[0].Type)
	assert.Equal(t, metav1.ConditionTrue, current.Status.Conditions[0].Status)
}

func TestReconcileMarksInvalidUpgradePath(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "paths",
			Annotations: testUpgradePathAnnotations("sha256:test"),
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
				{From: "v1.1.0", To: "v1.0.0"},
			},
		},
	}
	reconciler, service := newUpgradePathTestReconciler(up)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)
	assert.Equal(t, 0, service.PathCount())

	current := &upv1alpha1.UpgradePath{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{Name: "paths"}, current))
	assert.Equal(t, upv1alpha1.UpgradePathPhaseInvalid, current.Status.Phase)
	assert.Equal(t, 2, current.Status.PathCount)
	assert.Len(t, current.Status.Conditions, 1)
	assert.Equal(t, "Validated", current.Status.Conditions[0].Type)
	assert.Equal(t, metav1.ConditionFalse, current.Status.Conditions[0].Status)
}

func TestReconcileAcceptsEmptyPaths(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "paths",
			Annotations: testUpgradePathAnnotations("sha256:empty"),
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v1.0.0", Installable: true},
			},
		},
	}
	reconciler, service := newUpgradePathTestReconciler(up)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)

	assert.Equal(t, 0, service.PathCount())
	assert.Equal(t, "sha256:empty", service.Digest())
	assert.Equal(t, []string{"v1.0.0"}, service.GetInstallableVersions())

	current := &upv1alpha1.UpgradePath{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{Name: "paths"}, current))
	assert.Equal(t, upv1alpha1.UpgradePathPhaseActive, current.Status.Phase)
	assert.Equal(t, 0, current.Status.PathCount)
	assert.Equal(t, "sha256:empty", current.Status.LastDigest)
	require.Len(t, current.Status.Conditions, 1)
	assert.Equal(t, metav1.ConditionTrue, current.Status.Conditions[0].Status)
}

func TestReconcileClearsGraphWhenUpgradePathDeleted(t *testing.T) {
	reconciler, service := newUpgradePathTestReconciler()
	require.NoError(t, service.Load([]upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
	}, nil, "digest-a"))

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)
	assert.Equal(t, 0, service.PathCount())
}

func TestPathServiceIsCreatedLazily(t *testing.T) {
	reconciler := &UpgradePathReconciler{}
	require.NotNil(t, reconciler.pathService())
	require.NotNil(t, reconciler.PathService)
}

func TestFindPathDelegatesToPathService(t *testing.T) {
	reconciler, service := newUpgradePathTestReconciler()
	require.NoError(t, service.Load([]upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
	}, nil, "digest-a"))

	path, err := reconciler.FindPath("v1.0.0", "v1.1.0")
	require.NoError(t, err)
	assert.Len(t, path, 1)
}

func TestReconcileWithVersions(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "paths",
			Annotations: testUpgradePathAnnotations("sha256:test"),
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v2.4.0", To: "v2.5.0"},
				{From: "v2.5.0", To: "v2.6.0"},
			},
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v2.4.0", Installable: false, Deprecated: true},
				{Version: "v2.5.0", Installable: true, Deprecated: false},
				{Version: "v2.6.0", Installable: true, Deprecated: false},
			},
		},
	}
	reconciler, service := newUpgradePathTestReconciler(up)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)

	installable := service.GetInstallableVersions()
	assert.Equal(t, []string{"v2.5.0", "v2.6.0"}, installable)

	upgradeable := service.GetUpgradeableVersions("v2.4.0")
	assert.Equal(t, []string{"v2.5.0", "v2.6.0"}, upgradeable)
}

func TestReconcileDoesNotDuplicateConditions(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "paths",
			Annotations: testUpgradePathAnnotations("sha256:test"),
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
				{From: "v1.1.0", To: "v1.2.0"},
			},
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v1.0.0", Installable: true},
				{Version: "v1.1.0", Installable: true},
				{Version: "v1.2.0", Installable: true},
			},
		},
	}
	reconciler, _ := newUpgradePathTestReconciler(up)

	for i := 0; i < 3; i++ {
		_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
			NamespacedName: client.ObjectKey{Name: "paths"},
		})
		require.NoError(t, err)
	}

	current := &upv1alpha1.UpgradePath{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{Name: "paths"}, current))
	assert.Len(t, current.Status.Conditions, 1)
	assert.Equal(t, "Validated", current.Status.Conditions[0].Type)
	assert.Equal(t, metav1.ConditionTrue, current.Status.Conditions[0].Status)
}

func TestReconcileSkipsPatchWhenDigestUnchanged(t *testing.T) {
	up := &upv1alpha1.UpgradePath{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "paths",
			Annotations: testUpgradePathAnnotations("sha256:test"),
		},
		Spec: upv1alpha1.UpgradePathSpec{
			Paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
			},
			Versions: []upv1alpha1.VersionEntry{
				{Version: "v1.0.0", Installable: true},
				{Version: "v1.1.0", Installable: true},
			},
		},
	}
	reconciler, _ := newUpgradePathTestReconciler(up)

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)

	current := &upv1alpha1.UpgradePath{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{Name: "paths"}, current))
	firstCheckedAt := current.Status.LastCheckedAt

	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKey{Name: "paths"},
	})
	require.NoError(t, err)

	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKey{Name: "paths"}, current))
	assert.Equal(t, firstCheckedAt, current.Status.LastCheckedAt)
}
