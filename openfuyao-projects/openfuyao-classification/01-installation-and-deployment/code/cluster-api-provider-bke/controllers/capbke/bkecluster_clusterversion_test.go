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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/featuregate"
)

func TestCompleteClusterVersionInstall(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", UID: "uid-1"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v2.6.0"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus:    bkev1beta1.ClusterStatusReady,
			OpenFuyaoVersion: "v2.6.0",
		},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "c1",
			Namespace: "ns",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: bkev1beta1.GroupVersion.String(),
				Kind:       "BKECluster",
				Name:       "c1",
				UID:        "uid-1",
				Controller: ptr(true),
			}},
		},
		Spec:   cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
		Status: cvv1alpha1.ClusterVersionStatus{Phase: cvv1alpha1.ClusterVersionPhaseInstalling},
	}

	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc, cv).WithStatusSubresource(cv).Build()
	recorder := record.NewFakeRecorder(8)
	r := &BKEClusterReconciler{Client: c, Scheme: scheme, Recorder: recorder}
	bkeLogger := bkev1beta1.NewBKELogger(nil, recorder, bc)

	if err := r.completeClusterVersionInstall(context.Background(), bc, bkeLogger); err != nil {
		t.Fatal(err)
	}

	got := &cvv1alpha1.ClusterVersion{}
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got); err != nil {
		t.Fatal(err)
	}
	if got.Status.CurrentVersion != "v2.6.0" {
		t.Fatalf("currentVersion %q", got.Status.CurrentVersion)
	}
	if got.Status.Phase != cvv1alpha1.ClusterVersionPhaseReady {
		t.Fatalf("phase %q", got.Status.Phase)
	}
}

func TestCompleteClusterVersionInstall_SkipsWhenUpgrading(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = cvv1alpha1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)

	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "c1",
			Namespace: "ns",
			Annotations: map[string]string{
				featuregate.UpgradeReadyAnnotationKey: "v2.7.0",
			},
		},
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: bkev1beta1.ClusterStatusReady},
	}
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.7.0"},
		Status: cvv1alpha1.ClusterVersionStatus{
			CurrentVersion: "v2.6.0",
			Phase:          cvv1alpha1.ClusterVersionPhaseInstalling,
		},
	}

	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc, cv).WithStatusSubresource(cv).Build()
	recorder := record.NewFakeRecorder(8)
	r := &BKEClusterReconciler{Client: c, Scheme: scheme, Recorder: recorder}
	if err := r.completeClusterVersionInstall(context.Background(), bc, bkev1beta1.NewBKELogger(nil, recorder, bc)); err != nil {
		t.Fatal(err)
	}

	got := &cvv1alpha1.ClusterVersion{}
	_ = c.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c1"}, got)
	if got.Status.Phase != cvv1alpha1.ClusterVersionPhaseInstalling {
		t.Fatalf("phase should remain Installing, got %q", got.Status.Phase)
	}
}

func ptr(b bool) *bool { return &b }
