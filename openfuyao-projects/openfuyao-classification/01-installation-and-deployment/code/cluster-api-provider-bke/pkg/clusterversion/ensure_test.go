/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 ******************************************************************/

package clusterversion

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestReleaseImageRefForVersion(t *testing.T) {
	if got := ReleaseImageRefForVersion(" v26.03 "); got != "release-v26.03" {
		t.Fatalf("got %q", got)
	}
	if got := ReleaseImageRefForVersion(" V26/03@RC:1 "); got != "release-v26-03-rc-1" {
		t.Fatalf("got sanitized ref %q", got)
	}
}

func TestOpenFuyaoVersionForBKECluster(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "spec-ver"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "status-ver"},
	}
	if got := OpenFuyaoVersionForBKECluster(bc); got != "status-ver" {
		t.Fatalf("got %q", got)
	}

	bc.Status.OpenFuyaoVersion = ""
	if got := OpenFuyaoVersionForBKECluster(bc); got != "spec-ver" {
		t.Fatalf("got spec fallback %q", got)
	}
	if got := OpenFuyaoVersionForBKECluster(nil); got != "" {
		t.Fatalf("nil cluster got %q", got)
	}
	if got := OpenFuyaoVersionForBKECluster(&bkev1beta1.BKECluster{}); got != "" {
		t.Fatalf("empty cluster got %q", got)
	}
}

func TestNewClusterVersionFromBKEClusterErrors(t *testing.T) {
	if _, err := NewClusterVersionFromBKECluster(nil, InstallProvision()); err == nil {
		t.Fatalf("expected nil cluster error")
	}

	bc := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	if _, err := NewClusterVersionFromBKECluster(bc, InstallProvision()); err == nil {
		t.Fatalf("expected empty version error")
	}
}

func TestNewClusterVersionFromBKECluster(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: " v2.6.0 "},
			},
		},
	}

	cv, err := NewClusterVersionFromBKECluster(bc, InstallProvision())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cv.Name != "c1" || cv.Namespace != "ns" || cv.Spec.DesiredVersion != "v2.6.0" {
		t.Fatalf("unexpected ClusterVersion: %#v", cv)
	}
}

func TestEnsureClusterVersionForBKECluster_CreatesOnInstall(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", UID: "uid-1"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v2.6.0"},
			},
		},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).Build()

	created, err := EnsureClusterVersionForBKECluster(context.Background(), c, scheme, bc)
	if err != nil || !created {
		t.Fatalf("created=%v err=%v", created, err)
	}

	cv := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, cv); err != nil {
		t.Fatal(err)
	}
	if cv.Spec.DesiredVersion != "v2.6.0" {
		t.Fatalf("desired %q", cv.Spec.DesiredVersion)
	}
	if cv.Status.Phase != "" {
		t.Fatalf("status phase should be empty on create, got %q", cv.Status.Phase)
	}
	if len(cv.OwnerReferences) != 1 || cv.OwnerReferences[0].Kind != "BKECluster" {
		t.Fatalf("owner refs %+v", cv.OwnerReferences)
	}

	created, err = EnsureClusterVersionForBKECluster(context.Background(), c, scheme, bc)
	if err != nil || created {
		t.Fatalf("second ensure created=%v err=%v", created, err)
	}
}

func TestEnsureClusterVersionForBKECluster_NoCreateCases(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	if created, err := EnsureClusterVersionForBKECluster(context.Background(), nil, scheme, nil); err == nil || created {
		t.Fatalf("nil cluster created=%v err=%v", created, err)
	}

	deleteTime := metav1.Now()
	deletingBC := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "deleting",
			Namespace:         "ns",
			DeletionTimestamp: &deleteTime,
			Finalizers:        []string{"test"},
		},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(deletingBC).Build()
	created, err := EnsureClusterVersionForBKECluster(context.Background(), c, scheme, deletingBC)
	if err != nil || created {
		t.Fatalf("deleting cluster created=%v err=%v", created, err)
	}

	existingBC := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "existing", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v2.6.0"},
			},
		},
	}
	existingCV := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "existing", Namespace: "ns"},
	}
	c = fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingBC, existingCV).Build()
	created, err = EnsureClusterVersionForBKECluster(context.Background(), c, scheme, existingBC)
	if err != nil || created {
		t.Fatalf("existing ClusterVersion created=%v err=%v", created, err)
	}
}
