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

package clusterversion

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestGetClusterVersionForBKECluster_OwnerReference(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", UID: "bc-uid"},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cv-other",
			Namespace: "ns",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: bkev1beta1.GroupVersion.String(),
				Kind:       "BKECluster",
				Name:       "c1",
				UID:        "bc-uid",
			}},
		},
		Spec: cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v1"},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cv, bc).Build()

	got, err := GetClusterVersionForBKECluster(context.Background(), c, bc)
	if err != nil || got.Name != "cv-other" {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestGetClusterVersionForBKECluster_SameName(t *testing.T) {
	bc := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cv, bc).Build()

	got, err := GetClusterVersionForBKECluster(context.Background(), c, bc)
	if err != nil || got.Name != "c1" {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestGetClusterVersionForBKECluster_ErrorCases(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	if _, err := GetClusterVersionForBKECluster(context.Background(), c, nil); err == nil {
		t.Fatalf("expected nil cluster error")
	}
	if _, err := GetClusterVersionForBKECluster(context.Background(), nil, &bkev1beta1.BKECluster{}); err == nil {
		t.Fatalf("expected nil client error")
	}
	if _, err := GetClusterVersionForBKECluster(context.Background(), c, &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "ns"},
	}); err == nil {
		t.Fatalf("expected not found error")
	}
}

func TestGetBKEClusterForClusterVersion(t *testing.T) {
	bc := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cv1",
			Namespace: "ns",
			OwnerReferences: []metav1.OwnerReference{{
				Kind: "BKECluster",
				Name: "c1",
			}},
		},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc, cv).Build()

	got, err := GetBKEClusterForClusterVersion(context.Background(), c, cv)
	if err != nil || got.Name != "c1" {
		t.Fatalf("owner lookup got %v err %v", got, err)
	}

	sameName := &cvv1alpha1.ClusterVersion{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
	got, err = GetBKEClusterForClusterVersion(context.Background(), c, sameName)
	if err != nil || got.Name != "c1" {
		t.Fatalf("same-name lookup got %v err %v", got, err)
	}
}

func TestGetBKEClusterForClusterVersion_ErrorCases(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()

	if _, err := GetBKEClusterForClusterVersion(context.Background(), c, nil); err == nil {
		t.Fatalf("expected nil ClusterVersion error")
	}
	if _, err := GetBKEClusterForClusterVersion(context.Background(), c, &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "missing", Namespace: "ns"},
	}); err == nil {
		t.Fatalf("expected missing BKECluster error")
	}
}

func TestClusterVersionOwnsBKECluster(t *testing.T) {
	bc := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", UID: "uid-1"}}
	if clusterVersionOwnsBKECluster(nil, bc) {
		t.Fatalf("nil ClusterVersion should not own")
	}
	if clusterVersionOwnsBKECluster(&cvv1alpha1.ClusterVersion{}, nil) {
		t.Fatalf("nil BKECluster should not match")
	}
	cv := &cvv1alpha1.ClusterVersion{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{
		Kind: "Other",
		Name: "c1",
	}}}}
	if clusterVersionOwnsBKECluster(cv, bc) {
		t.Fatalf("wrong kind should not match")
	}
	cv.OwnerReferences = []metav1.OwnerReference{{Kind: "BKECluster", Name: "c1", UID: "other"}}
	if clusterVersionOwnsBKECluster(cv, bc) {
		t.Fatalf("wrong uid should not match")
	}
	cv.OwnerReferences = []metav1.OwnerReference{{Kind: "BKECluster", Name: "c1"}}
	if !clusterVersionOwnsBKECluster(cv, bc) {
		t.Fatalf("empty owner uid should match by name")
	}
}
