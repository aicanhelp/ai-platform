/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 ******************************************************************/

package clusterversion

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/imageref"
)

func TestReleaseImageCRFromParsedUsesMetadataName(t *testing.T) {
	parsed := &cvv1alpha1.ReleaseImage{
		Spec: cvv1alpha1.ReleaseImageSpec{Version: "v26.05"},
	}
	parsed.Name = "release-v26.05"
	ri := releaseImageCRFromParsed(parsed, "ns", "cluster-a")
	if ri.Name != "release-v26.05" {
		t.Fatalf("name %q", ri.Name)
	}
	if ri.Annotations[imageref.AnnotationBKEClusterName] != "cluster-a" {
		t.Fatalf("annotation %v", ri.Annotations)
	}
}

func TestReleaseImageCRFromParsedFallbackName(t *testing.T) {
	parsed := &cvv1alpha1.ReleaseImage{
		Spec: cvv1alpha1.ReleaseImageSpec{Version: "v26.05"},
	}
	ri := releaseImageCRFromParsed(parsed, "ns", "cluster-a")
	if ri.Name != "release-v26.05" {
		t.Fatalf("name %q", ri.Name)
	}
}

func TestReleaseImageEnsurerEnsureExistingReleaseImage(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	ri := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release-v26.05", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v26.05"},
		Status:     cvv1alpha1.ReleaseImageStatus{Phase: cvv1alpha1.ReleaseImagePhaseValid},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ri).Build()
	ensurer := &ReleaseImageEnsurer{Client: c, Scheme: scheme}
	cv := &cvv1alpha1.ClusterVersion{ObjectMeta: metav1.ObjectMeta{Name: "cluster-a", Namespace: "ns"}}

	got, result, err := ensurer.Ensure(context.Background(), &bkev1beta1.BKECluster{}, cv, " v26.05 ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "release-v26.05" || result != (ctrl.Result{}) {
		t.Fatalf("got %s result %#v", got.Name, result)
	}
}

func TestReleaseImageEnsurerEnsureExistingReleaseImagePhases(t *testing.T) {
	tests := []struct {
		name      string
		phase     cvv1alpha1.ReleaseImagePhase
		wantError bool
		wantQueue bool
	}{
		{name: "invalid", phase: cvv1alpha1.ReleaseImagePhaseInvalid, wantError: true},
		{name: "manifest missing", phase: cvv1alpha1.ReleaseImagePhaseManifestMissing, wantError: true},
		{name: "compatibility failed", phase: cvv1alpha1.ReleaseImagePhaseCompatibilityFailed, wantError: true},
		{name: "pending", phase: "", wantQueue: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			_ = cvv1alpha1.AddToScheme(scheme)
			ri := &cvv1alpha1.ReleaseImage{
				ObjectMeta: metav1.ObjectMeta{Name: "release", Namespace: "ns"},
				Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v26.05"},
				Status:     cvv1alpha1.ReleaseImageStatus{Phase: tt.phase, Message: "bad"},
			}
			c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ri).Build()
			ensurer := &ReleaseImageEnsurer{Client: c, Scheme: scheme}
			cv := &cvv1alpha1.ClusterVersion{ObjectMeta: metav1.ObjectMeta{Name: "cluster-a", Namespace: "ns"}}

			_, result, err := ensurer.Ensure(context.Background(), nil, cv, "v26.05")
			if tt.wantError && err == nil {
				t.Fatalf("expected error")
			}
			if !tt.wantError && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.wantQueue && result.RequeueAfter != releaseImageRequeueInterval {
				t.Fatalf("expected requeue, got %#v", result)
			}
		})
	}
}

func TestReleaseImageEnsurerEnsureErrors(t *testing.T) {
	if _, _, err := (*ReleaseImageEnsurer)(nil).Ensure(context.Background(), nil, nil, "v1"); err == nil {
		t.Fatalf("expected nil ensurer error")
	}
	ensurer := &ReleaseImageEnsurer{}
	if _, _, err := ensurer.Ensure(context.Background(), nil, nil, "v1"); err == nil {
		t.Fatalf("expected unconfigured client error")
	}

	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	c := fake.NewClientBuilder().WithScheme(scheme).Build()
	ensurer = &ReleaseImageEnsurer{Client: c}
	if _, _, err := ensurer.Ensure(context.Background(), nil, &cvv1alpha1.ClusterVersion{}, " "); err == nil {
		t.Fatalf("expected empty version error")
	}
}

func TestFindReleaseImageByVersion(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	ri := &cvv1alpha1.ReleaseImage{
		ObjectMeta: metav1.ObjectMeta{Name: "release", Namespace: "ns"},
		Spec:       cvv1alpha1.ReleaseImageSpec{Version: "v26.05"},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ri).Build()
	ensurer := &ReleaseImageEnsurer{Client: c}

	got, err := ensurer.findReleaseImageByVersion(context.Background(), "ns", "v26.05")
	if err != nil || got.Name != "release" {
		t.Fatalf("got %v err %v", got, err)
	}
	got, err = ensurer.findReleaseImageByVersion(context.Background(), "ns", "missing")
	if err != nil || got != nil {
		t.Fatalf("expected nil missing result, got %v err %v", got, err)
	}
}
