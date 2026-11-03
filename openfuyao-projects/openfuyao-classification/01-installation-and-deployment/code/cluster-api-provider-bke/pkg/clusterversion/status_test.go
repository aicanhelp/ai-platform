/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 ******************************************************************/

package clusterversion

import (
	"context"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

func TestSyncClusterVersionInstallStatus(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			OpenFuyaoVersion: "v26.05",
			ClusterStatus:    bkev1beta1.ClusterStatusReady,
		},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.05"},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bc, cv).
		WithStatusSubresource(&cvv1alpha1.ClusterVersion{}).
		Build()

	if err := SyncClusterVersionInstallStatus(context.Background(), c, bc); err != nil {
		t.Fatal(err)
	}
	got := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got); err != nil {
		t.Fatal(err)
	}
	if got.Status.CurrentVersion != "v26.05" {
		t.Fatalf("currentVersion %q", got.Status.CurrentVersion)
	}
	if got.Status.Phase != cvv1alpha1.ClusterVersionPhaseReady {
		t.Fatalf("phase %q", got.Status.Phase)
	}
	if len(got.Status.Conditions) != 1 || got.Status.Conditions[0].Reason != "InstallComplete" {
		t.Fatalf("conditions %+v", got.Status.Conditions)
	}
}

func TestSyncClusterVersionInstallStatus_SkipsWhenUpgradePending(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v26.06"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			OpenFuyaoVersion: "v26.05",
			ClusterStatus:    bkev1beta1.ClusterStatusReady,
		},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseReady,
		},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bc, cv).
		WithStatusSubresource(&cvv1alpha1.ClusterVersion{}).
		Build()

	ok, err := ShouldSyncClusterVersionInstallStatus(context.Background(), c, bc)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("expected no install sync while upgrade hop is pending")
	}
	if err := SyncClusterVersionInstallStatus(context.Background(), c, bc); err != nil {
		t.Fatal(err)
	}
	got := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got); err != nil {
		t.Fatal(err)
	}
	if got.Status.CurrentVersion != "v26.05" {
		t.Fatalf("currentVersion should stay v26.05, got %q", got.Status.CurrentVersion)
	}
}

func TestShouldSyncClusterVersionInstallStatus_SkipsWhenUpgradeReady(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "c1",
			Namespace: "ns",
			Annotations: map[string]string{
				annotation.CVOUpgradeReadyAnnotationKey: "v26.06",
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			OpenFuyaoVersion: "v26.05",
			ClusterStatus:    bkev1beta1.ClusterStatusReady,
		},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseReady,
		},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc, cv).Build()

	ok, err := ShouldSyncClusterVersionInstallStatus(context.Background(), c, bc)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("expected no install sync when upgrade-ready is set")
	}
}

func TestCompleteUpgradeHop(t *testing.T) {
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseUpgrading,
		},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cv).
		WithStatusSubresource(&cvv1alpha1.ClusterVersion{}).
		Build()

	if err := CompleteUpgradeHop(context.Background(), c, cv, "v26.06"); err != nil {
		t.Fatal(err)
	}
	got := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got); err != nil {
		t.Fatal(err)
	}
	if got.Status.CurrentVersion != "v26.06" {
		t.Fatalf("currentVersion %q", got.Status.CurrentVersion)
	}
	if got.Status.Phase != cvv1alpha1.ClusterVersionPhaseReady {
		t.Fatalf("phase %q", got.Status.Phase)
	}
	if len(got.Status.UpgradeHistory) != 1 {
		t.Fatalf("history %+v", got.Status.UpgradeHistory)
	}
	if got.Status.UpgradeHistory[0].From != "v26.05" || got.Status.UpgradeHistory[0].To != "v26.06" {
		t.Fatalf("history %+v", got.Status.UpgradeHistory[0])
	}
}

func TestHasUpgradeRecord(t *testing.T) {
	records := []cvv1alpha1.ClusterUpgradeRecord{{From: "v26.05", To: "v26.06"}}
	if !HasUpgradeRecord(records, "v26.05", "v26.06") {
		t.Fatal("expected record")
	}
	if HasUpgradeRecord(records, "v26.04", "v26.06") {
		t.Fatal("unexpected record")
	}
}

func TestFailUpgradeHop(t *testing.T) {
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.06"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v26.05",
			Phase:          cvv1alpha1.ClusterVersionPhaseUpgrading,
		},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cv).
		WithStatusSubresource(&cvv1alpha1.ClusterVersion{}).
		Build()

	if err := FailUpgradeHop(context.Background(), c, cv, "v26.06", fmt.Errorf("etcd upgrade failed")); err != nil {
		t.Fatal(err)
	}
	got := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got); err != nil {
		t.Fatal(err)
	}
	if got.Status.Phase != cvv1alpha1.ClusterVersionPhaseFailed {
		t.Fatalf("phase %q", got.Status.Phase)
	}
	if got.Status.CurrentVersion != "v26.05" {
		t.Fatalf("currentVersion should remain v26.05, got %q", got.Status.CurrentVersion)
	}
	if len(got.Status.UpgradeHistory) != 1 {
		t.Fatalf("history %+v", got.Status.UpgradeHistory)
	}
	if got.Status.UpgradeHistory[0].Status != cvv1alpha1.ClusterUpgradeRecordStatusFailed {
		t.Fatalf("history status %q", got.Status.UpgradeHistory[0].Status)
	}
	if len(got.Status.Conditions) != 1 || got.Status.Conditions[0].Reason != "UpgradeHopFailed" {
		t.Fatalf("conditions %+v", got.Status.Conditions)
	}
}
