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

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestResolveReleaseImageForDesiredVersion_ConventionalName(t *testing.T) {
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "cv1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v26.03"},
	}
	ri := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release-v26.03", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v26.03"},
		Status:     cvv1alpha1.ReleaseImageStatus{Phase: cvv1alpha1.ReleaseImagePhaseValid},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ri).Build()

	got, err := ResolveReleaseImageForDesiredVersion(context.Background(), c, cv)
	if err != nil || got.Name != "release-v26.03" {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestResolveReleaseImageForVersion_ConventionalName(t *testing.T) {
	ri := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release-v2.5.0", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v2.5.0"},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ri).Build()

	got, err := ResolveReleaseImageForVersion(context.Background(), c, "ns", "v2.5.0")
	if err != nil || got.Name != "release-v2.5.0" {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestResolveReleaseImageForDesiredVersion_ListByVersion(t *testing.T) {
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "cv1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
	}
	ri := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "custom-release-name", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v2.6.0"},
		Status:     cvv1alpha1.ReleaseImageStatus{Phase: cvv1alpha1.ReleaseImagePhaseValid},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ri).Build()

	got, err := ResolveReleaseImageForDesiredVersion(context.Background(), c, cv)
	if err != nil || got.Name != "custom-release-name" {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestResolveReleaseImageForDesiredVersion_PrefersValidPhase(t *testing.T) {
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "cv1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
	}
	invalid := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release-a", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v2.6.0"},
		Status:     cvv1alpha1.ReleaseImageStatus{Phase: cvv1alpha1.ReleaseImagePhaseInvalid},
	}
	valid := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release-b", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v2.6.0"},
		Status:     cvv1alpha1.ReleaseImageStatus{Phase: cvv1alpha1.ReleaseImagePhaseValid},
	}
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(invalid, valid).Build()

	got, err := ResolveReleaseImageForDesiredVersion(context.Background(), c, cv)
	if err != nil || got.Name != "release-b" {
		t.Fatalf("got %v err %v", got, err)
	}
}
