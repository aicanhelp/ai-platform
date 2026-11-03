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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
)

const (
	numOne  = 1
	numTwo  = 2
	numZero = 0
)

func TestBKEMachineTypes(t *testing.T) {
	t.Run("BKEMachineNilCases", func(t *testing.T) {
		var nilMachine *BKEMachine
		if nilMachine.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilMachine.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEMachineWithProviderID", func(t *testing.T) {
		providerID := "test-provider-id"
		original := &BKEMachine{
			Spec: BKEMachineSpec{
				ProviderID: &providerID,
				Pause:      true,
				DryRun:     false,
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Spec.ProviderID == nil {
			t.Error("ProviderID should not be nil")
		}
		if *copied.Spec.ProviderID != *original.Spec.ProviderID {
			t.Errorf("Expected ProviderID %s, got %s", *original.Spec.ProviderID, *copied.Spec.ProviderID)
		}

		*copied.Spec.ProviderID = "modified-id"
		if *original.Spec.ProviderID == *copied.Spec.ProviderID {
			t.Error("Modifying copy affected original")
		}

		if copied.Spec.Pause != original.Spec.Pause {
			t.Errorf("Expected Pause %v, got %v", original.Spec.Pause, copied.Spec.Pause)
		}
	})

	t.Run("BKEMachineWithoutProviderID", func(t *testing.T) {
		original := &BKEMachine{
			Spec: BKEMachineSpec{
				ProviderID: nil,
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Spec.ProviderID != nil {
			t.Error("ProviderID should be nil")
		}
	})

	t.Run("BKEMachineStatusWithAddresses", func(t *testing.T) {
		original := &BKEMachineStatus{
			Ready:        true,
			Bootstrapped: true,
			Addresses: []MachineAddress{
				{Type: MachineHostName, Address: "node-1"},
				{Type: MachineInternalIP, Address: "192.168.1.1"},
				{Type: MachineExternalIP, Address: "10.0.0.1"},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Addresses) != len(original.Addresses) {
			t.Errorf("Expected %d addresses, got %d", len(original.Addresses), len(copied.Addresses))
		}

		copied.Addresses[numZero].Address = "modified-node"
		if original.Addresses[numZero].Address == copied.Addresses[numZero].Address {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEMachineStatusWithConditions", func(t *testing.T) {
		original := &BKEMachineStatus{
			Ready:        true,
			Bootstrapped: true,
			Conditions: clusterv1.Conditions{
				{
					Type:   clusterv1.ReadyCondition,
					Status: corev1.ConditionTrue,
				},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Conditions) != len(original.Conditions) {
			t.Errorf("Expected %d conditions, got %d", len(original.Conditions), len(copied.Conditions))
		}

		copied.Conditions[numZero].Status = corev1.ConditionFalse
		if original.Conditions[numZero].Status == copied.Conditions[numZero].Status {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEMachineStatusNilFields", func(t *testing.T) {
		original := &BKEMachineStatus{
			Ready:        true,
			Bootstrapped: false,
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Addresses != nil {
			t.Error("Addresses should be nil")
		}
		if copied.Conditions != nil {
			t.Error("Conditions should be nil")
		}
		if copied.Node != nil {
			t.Error("Node should be nil")
		}
	})

	t.Run("BKEMachineListWithItems", func(t *testing.T) {
		original := &BKEMachineList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEMachineList",
			},
			ListMeta: metav1.ListMeta{
				ResourceVersion: "1000",
			},
			Items: []BKEMachine{
				{Spec: BKEMachineSpec{Pause: true}},
				{Spec: BKEMachineSpec{Pause: false}},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Items) != len(original.Items) {
			t.Errorf("Expected %d items, got %d", len(original.Items), len(copied.Items))
		}

		copied.Items[numZero].Spec.Pause = false
		if original.Items[numZero].Spec.Pause == copied.Items[numZero].Spec.Pause {
			t.Error("Modifying copy affected original")
		}
	})

	t.Run("BKEMachineListNilItems", func(t *testing.T) {
		original := &BKEMachineList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEMachineList",
			},
			Items: nil,
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Items != nil {
			t.Error("Items should be nil")
		}

		var nilList *BKEMachineList
		if nilList.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilList.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEMachineListDeepCopyObject", func(t *testing.T) {
		original := &BKEMachineList{
			Items: []BKEMachine{{Spec: BKEMachineSpec{}}},
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}

		listObj, ok := obj.(*BKEMachineList)
		if !ok {
			t.Error("DeepCopyObject should return BKEMachineList")
		}

		if listObj == original {
			t.Error("DeepCopyObject should return different instance")
		}
	})

	t.Run("BKEMachineDeepCopyObject", func(t *testing.T) {
		original := &BKEMachine{
			Spec: BKEMachineSpec{
				Pause: true,
			},
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}

		machineObj, ok := obj.(*BKEMachine)
		if !ok {
			t.Error("DeepCopyObject should return BKEMachine")
		}

		if machineObj == original {
			t.Error("DeepCopyObject should return different instance")
		}
	})

	t.Run("MachineAddressTypes", func(t *testing.T) {
		tests := []struct {
			name     string
			addrType MachineAddressType
			address  string
		}{
			{"Hostname", MachineHostName, "node-1"},
			{"InternalIP", MachineInternalIP, "192.168.1.1"},
			{"ExternalIP", MachineExternalIP, "10.0.0.1"},
			{"InternalDNS", MachineInternalDNS, "node-1.internal.local"},
			{"ExternalDNS", MachineExternalDNS, "node-1.example.com"},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				original := &MachineAddress{
					Type:    tt.addrType,
					Address: tt.address,
				}

				copied := original.DeepCopy()
				if copied == nil {
					t.Fatal("DeepCopy returned nil")
				}

				if copied.Type != original.Type {
					t.Errorf("Expected Type %s, got %s", original.Type, copied.Type)
				}
				if copied.Address != original.Address {
					t.Errorf("Expected Address %s, got %s", original.Address, copied.Address)
				}

				copied.Address = "modified"
				if original.Address == copied.Address {
					t.Error("Modifying copy affected original")
				}

				var nilAddr *MachineAddress
				if nilAddr.DeepCopy() != nil {
					t.Error("DeepCopy of nil should return nil")
				}
			})
		}
	})

	t.Run("BKEMachineSpecDeepCopy", func(t *testing.T) {
		original := &BKEMachineSpec{
			ProviderID: nil,
			Pause:      true,
			DryRun:     true,
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Pause != original.Pause {
			t.Errorf("Expected Pause %v, got %v", original.Pause, copied.Pause)
		}
		if copied.DryRun != original.DryRun {
			t.Errorf("Expected DryRun %v, got %v", original.DryRun, copied.DryRun)
		}

		var nilSpec *BKEMachineSpec
		if nilSpec.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
	})

	t.Run("BKEMachineWithFullStatus", func(t *testing.T) {
		providerID := "test-id"
		original := &BKEMachine{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEMachine",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:            "test-machine",
				Namespace:       "default",
				ResourceVersion: "12345",
			},
			Spec: BKEMachineSpec{
				ProviderID: &providerID,
				Pause:      false,
				DryRun:     false,
			},
			Status: BKEMachineStatus{
				Ready:        true,
				Bootstrapped: true,
				Addresses: []MachineAddress{
					{Type: MachineHostName, Address: "test-node"},
					{Type: MachineInternalIP, Address: "192.168.1.100"},
				},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
		}
		if copied.Namespace != original.Namespace {
			t.Errorf("Expected Namespace %s, got %s", original.Namespace, copied.Namespace)
		}
		if copied.Spec.Pause != original.Spec.Pause {
			t.Error("Spec should match")
		}
		if !copied.Status.Ready {
			t.Error("Status Ready should be true")
		}
		if !copied.Status.Bootstrapped {
			t.Error("Status Bootstrapped should be true")
		}

		if len(copied.Status.Addresses) != len(original.Status.Addresses) {
			t.Error("Addresses should be copied")
		}
	})
}

func TestGetConditionsAndSetConditions(t *testing.T) {
	t.Run("GetConditions", func(t *testing.T) {
		machine := &BKEMachine{
			Status: BKEMachineStatus{
				Conditions: clusterv1.Conditions{
					{
						Type:   clusterv1.ReadyCondition,
						Status: corev1.ConditionTrue,
					},
				},
			},
		}

		conditions := machine.GetConditions()
		if len(conditions) != numOne {
			t.Errorf("Expected %d conditions, got %d", numOne, len(conditions))
		}
	})

	t.Run("SetConditions", func(t *testing.T) {
		machine := &BKEMachine{}

		newConditions := clusterv1.Conditions{
			{
				Type:   clusterv1.ReadyCondition,
				Status: corev1.ConditionTrue,
			},
			{
				Type:   clusterv1.ConditionType("UpToDate"),
				Status: corev1.ConditionFalse,
			},
		}

		machine.SetConditions(newConditions)

		if len(machine.Status.Conditions) != numTwo {
			t.Errorf("Expected %d conditions, got %d", numTwo, len(machine.Status.Conditions))
		}

		if machine.Status.Conditions[numZero].Type != clusterv1.ReadyCondition {
			t.Error("First condition type should be Ready")
		}
	})

	t.Run("GetConditionsEmpty", func(t *testing.T) {
		machine := &BKEMachine{}
		conditions := machine.GetConditions()
		if conditions != nil {
			t.Error("Expected nil conditions for empty machine")
		}
	})
}

func TestContainerdConfigDeepCopy(t *testing.T) {
	t.Run("ContainerdConfigNilCases", func(t *testing.T) {
		var nilConfig *ContainerdConfig
		if nilConfig.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilConfig.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("ContainerdConfigBasic", func(t *testing.T) {
		original := &ContainerdConfig{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "ContainerdConfig",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-config",
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}
	})

	t.Run("ContainerdConfigListNilCases", func(t *testing.T) {
		var nilList *ContainerdConfigList
		if nilList.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilList.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("ContainerdConfigListWithItems", func(t *testing.T) {
		original := &ContainerdConfigList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "ContainerdConfigList",
			},
			Items: []ContainerdConfig{
				{ObjectMeta: metav1.ObjectMeta{Name: "config-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "config-2"}},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Items) != numTwo {
			t.Errorf("Expected %d items, got %d", numTwo, len(copied.Items))
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}
	})
}

func TestBKEMachineTemplateDeepCopy(t *testing.T) {
	t.Run("BKEMachineTemplateNilCases", func(t *testing.T) {
		var nilTemplate *BKEMachineTemplate
		if nilTemplate.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilTemplate.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEMachineTemplateBasic", func(t *testing.T) {
		original := &BKEMachineTemplate{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEMachineTemplate",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-template",
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if copied.Name != original.Name {
			t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
		}

		obj := original.DeepCopyObject()
		if obj == nil {
			t.Error("DeepCopyObject should not return nil")
		}
	})

	t.Run("BKEMachineTemplateListNilCases", func(t *testing.T) {
		var nilList *BKEMachineTemplateList
		if nilList.DeepCopy() != nil {
			t.Error("DeepCopy of nil should return nil")
		}
		if nilList.DeepCopyObject() != nil {
			t.Error("DeepCopyObject of nil should return nil")
		}
	})

	t.Run("BKEMachineTemplateListWithItems", func(t *testing.T) {
		original := &BKEMachineTemplateList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "infrastructure.cluster.x-k8s.io/v1beta1",
				Kind:       "BKEMachineTemplateList",
			},
			Items: []BKEMachineTemplate{
				{ObjectMeta: metav1.ObjectMeta{Name: "template-1"}},
			},
		}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}

		if len(copied.Items) != numOne {
			t.Errorf("Expected %d item, got %d", numOne, len(copied.Items))
		}
	})

	t.Run("BKEMachineTemplateResource", func(t *testing.T) {
		original := &BKEMachineTemplateResource{
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
	})

	t.Run("BKEMachineTemplateSpec", func(t *testing.T) {
		original := &BKEMachineTemplateSpec{}

		copied := original.DeepCopy()
		if copied == nil {
			t.Fatal("DeepCopy returned nil")
		}
	})
}

func TestBKENodesDeepCopy(t *testing.T) {
	t.Run("BKENodesBasic", func(t *testing.T) {
		original := BKENodes{}
		out := &BKENodes{}
		original.DeepCopyInto(out)
	})
}
