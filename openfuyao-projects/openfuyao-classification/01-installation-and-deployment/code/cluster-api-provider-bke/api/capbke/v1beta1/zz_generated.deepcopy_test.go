/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package v1beta1

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

func TestBKEClusterDeepCopy(t *testing.T) {
	t.Run("BKEClusterNilCases", func(t *testing.T) {
		var nilCluster *BKECluster
		if nilCluster.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilCluster.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEClusterWithSpecAndStatus", func(t *testing.T) {
		original := &BKECluster{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKECluster",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:            "test-cluster",
				Namespace:       "default",
				ResourceVersion: "12345",
			},
			Spec: confv1beta1.BKEClusterSpec{
				ControlPlaneEndpoint: confv1beta1.APIEndpoint{
					Host: "api.example.com",
					Port: 6443,
				},
			},
			Status: confv1beta1.BKEClusterStatus{
				Ready: true,
				Phase: "Running",
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
		}

		if copied.Spec.ControlPlaneEndpoint.Host != original.Spec.ControlPlaneEndpoint.Host {
			t.Errorf("Expected Host %s, got %s", original.Spec.ControlPlaneEndpoint.Host, copied.Spec.ControlPlaneEndpoint.Host)
		}

		if !copied.Status.Ready {
			t.Error("Status Ready should be true")
		}

		copied.Status.Ready = false
		if original.Status.Ready == copied.Status.Ready {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEClusterDeepCopyInto", func(t *testing.T) {
		original := &BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-cluster",
			},
			Spec:   confv1beta1.BKEClusterSpec{},
			Status: confv1beta1.BKEClusterStatus{Ready: true},
		}

		out := &BKECluster{}
		original.DeepCopyInto(out)

		if out.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, out.Name)
		}

		if !out.Status.Ready {
			t.Error("Status Ready should be true")
		}
	})

	t.Run("BKEClusterDeepCopyObject", func(t *testing.T) {
		original := &BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-cluster",
			},
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}

		clusterObj, ok := obj.(*BKECluster)
		if !ok {
			t.Error("DeepCopyObject should return BKECluster")
		}

		if clusterObj.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, clusterObj.Name)
		}

		if clusterObj == original {
			t.Error("DeepCopyObject should return different instance")
		}
	})
}

func TestBKEClusterListDeepCopy(t *testing.T) {
	t.Run("BKEClusterListNilCases", func(t *testing.T) {
		var nilList *BKEClusterList
		if nilList.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilList.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEClusterListWithItems", func(t *testing.T) {
		original := &BKEClusterList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEClusterList",
			},
			ListMeta: metav1.ListMeta{
				ResourceVersion: "1000",
			},
			Items: []BKECluster{
				{ObjectMeta: metav1.ObjectMeta{Name: "cluster-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "cluster-2"}},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Items) != 2 {
			t.Errorf("Expected 2 items, got %d", len(copied.Items))
		}

		if copied.Items[0].Name != original.Items[0].Name {
			t.Error("First item should match")
		}

		copied.Items[0].Name = "modified"
		if original.Items[0].Name == copied.Items[0].Name {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEClusterListDeepCopyInto", func(t *testing.T) {
		original := &BKEClusterList{
			Items: []BKECluster{
				{ObjectMeta: metav1.ObjectMeta{Name: "cluster-1"}},
			},
		}

		out := &BKEClusterList{}
		original.DeepCopyInto(out)

		if len(out.Items) != 1 {
			t.Errorf("Expected 1 item, got %d", len(out.Items))
		}
	})

	t.Run("BKEClusterListDeepCopyObject", func(t *testing.T) {
		original := &BKEClusterList{
			Items: []BKECluster{{ObjectMeta: metav1.ObjectMeta{Name: "test"}}},
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}

		listObj, ok := obj.(*BKEClusterList)
		if !ok {
			t.Error("DeepCopyObject should return BKEClusterList")
		}

		if listObj == original {
			t.Error("DeepCopyObject should return different instance")
		}
	})
}

func TestBKEClusterTemplateDeepCopy(t *testing.T) {
	t.Run("BKEClusterTemplateNilCases", func(t *testing.T) {
		var nilTemplate *BKEClusterTemplate
		if nilTemplate.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilTemplate.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEClusterTemplateBasic", func(t *testing.T) {
		original := &BKEClusterTemplate{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEClusterTemplate",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-template",
			},
			Spec: BKEClusterTemplateSpec{
				Template: BKEClusterTemplateResource{},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
		}
	})

	t.Run("BKEClusterTemplateDeepCopyInto", func(t *testing.T) {
		original := &BKEClusterTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-template",
			},
		}

		out := &BKEClusterTemplate{}
		original.DeepCopyInto(out)

		if out.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, out.Name)
		}
	})

	t.Run("BKEClusterTemplateDeepCopyObject", func(t *testing.T) {
		original := &BKEClusterTemplate{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-template",
			},
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}

		templateObj, ok := obj.(*BKEClusterTemplate)
		if !ok {
			t.Error("DeepCopyObject should return BKEClusterTemplate")
		}

		if templateObj.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, templateObj.Name)
		}
	})
}

func TestBKEClusterTemplateListDeepCopy(t *testing.T) {
	t.Run("BKEClusterTemplateListNilCases", func(t *testing.T) {
		var nilList *BKEClusterTemplateList
		if nilList.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilList.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEClusterTemplateListWithItems", func(t *testing.T) {
		original := &BKEClusterTemplateList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEClusterTemplateList",
			},
			Items: []BKEClusterTemplate{
				{ObjectMeta: metav1.ObjectMeta{Name: "template-1"}},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Items) != 1 {
			t.Errorf("Expected 1 item, got %d", len(copied.Items))
		}
	})

	t.Run("BKEClusterTemplateListDeepCopyInto", func(t *testing.T) {
		original := &BKEClusterTemplateList{
			Items: []BKEClusterTemplate{
				{ObjectMeta: metav1.ObjectMeta{Name: "template-1"}},
			},
		}

		out := &BKEClusterTemplateList{}
		original.DeepCopyInto(out)

		if len(out.Items) != 1 {
			t.Errorf("Expected 1 item, got %d", len(out.Items))
		}
	})

	t.Run("BKEClusterTemplateListDeepCopyObject", func(t *testing.T) {
		original := &BKEClusterTemplateList{}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}
	})
}

func TestBKEClusterTemplateResourceDeepCopy(t *testing.T) {
	t.Run("BKEClusterTemplateResourceBasic", func(t *testing.T) {
		original := &BKEClusterTemplateResource{
			ObjectMeta: clusterv1.ObjectMeta{
				Labels: map[string]string{"key": "value"},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.ObjectMeta.Labels["key"] != "value" {
			t.Error("Labels should be copied")
		}

		copied.ObjectMeta.Labels["new-key"] = "new-value"
		if _, exists := original.ObjectMeta.Labels["new-key"]; exists {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEClusterTemplateResourceDeepCopyInto", func(t *testing.T) {
		original := &BKEClusterTemplateResource{}

		out := &BKEClusterTemplateResource{}
		original.DeepCopyInto(out)
	})
}

func TestBKEClusterTemplateSpecDeepCopy(t *testing.T) {
	t.Run("BKEClusterTemplateSpecBasic", func(t *testing.T) {
		original := &BKEClusterTemplateSpec{
			Template: BKEClusterTemplateResource{
				ObjectMeta: clusterv1.ObjectMeta{
					Labels: map[string]string{"env": "prod"},
				},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Template.ObjectMeta.Labels["env"] != "prod" {
			t.Error("Labels should be copied")
		}

		copied.Template.ObjectMeta.Labels["new-label"] = "new-value"
		if original.Template.ObjectMeta.Labels["new-label"] == "new-value" {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEClusterTemplateSpecDeepCopyInto", func(t *testing.T) {
		original := &BKEClusterTemplateSpec{}

		out := &BKEClusterTemplateSpec{}
		original.DeepCopyInto(out)
	})
}
