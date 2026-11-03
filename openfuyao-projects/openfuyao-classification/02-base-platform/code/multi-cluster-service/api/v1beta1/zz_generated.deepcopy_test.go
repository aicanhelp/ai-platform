/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1beta1

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestClusterSpecDeepCopy(t *testing.T) {
	// Create a test cluster spec
	original := &ClusterSpec{
		UID:            "test-uid",
		APIEndpoint:    "https://test-api-endpoint",
		Kubeconfigpath: "/path/to/kubeconfig",
		KubeConfig: runtime.RawExtension{
			Raw: []byte(`{"test": "kubeconfig"}`),
		},
		OutClusterConfig: runtime.RawExtension{
			Raw: []byte(`{"test": "outclusterconfig"}`),
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()
	assert.NotNil(t, copied)
	assert.Equal(t, original.UID, copied.UID)
	assert.Equal(t, original.APIEndpoint, copied.APIEndpoint)
	assert.Equal(t, original.Kubeconfigpath, copied.Kubeconfigpath)
	assert.Equal(t, original.KubeConfig.Raw, copied.KubeConfig.Raw)
	assert.Equal(t, original.OutClusterConfig.Raw, copied.OutClusterConfig.Raw)

	// Verify deep copy by modifying original
	original.UID = "new-uid"
	assert.Equal(t, "test-uid", copied.UID)
	assert.Equal(t, "new-uid", original.UID)
}

func TestClusterStatusDeepCopy(t *testing.T) {
	// Create a test cluster status
	original := &ClusterStatus{
		Condition: []v1.Condition{
			{
				Type:   "Ready",
				Status: v1.ConditionTrue,
				Reason: "ClusterReady",
			},
		},
		ResourceList: ResourcesCondition{
			AllocatableCPU:    "4",
			AllocatableMemory: "8Gi",
			AllocatedCPU:      "2",
			AllocatedMemory:   "4Gi",
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()
	assert.NotNil(t, copied)
	assert.Equal(t, len(original.Condition), len(copied.Condition))
	assert.Equal(t, original.Condition[0].Type, copied.Condition[0].Type)
	assert.Equal(t, original.Condition[0].Status, copied.Condition[0].Status)
	assert.Equal(t, original.ResourceList, copied.ResourceList)

	// Verify deep copy by modifying original
	original.ResourceList.AllocatableCPU = "8"
	assert.Equal(t, "4", copied.ResourceList.AllocatableCPU)
	assert.Equal(t, "8", original.ResourceList.AllocatableCPU)
}

func TestResourcesConditionDeepCopy(t *testing.T) {
	// Create a test resources condition
	original := &ResourcesCondition{
		AllocatableCPU:    "4",
		AllocatableMemory: "8Gi",
		AllocatedCPU:      "2",
		AllocatedMemory:   "4Gi",
	}

	// Test DeepCopy
	copied := original
	assert.NotNil(t, copied)
	assert.Equal(t, original.AllocatableCPU, copied.AllocatableCPU)
	assert.Equal(t, original.AllocatableMemory, copied.AllocatableMemory)
	assert.Equal(t, original.AllocatedCPU, copied.AllocatedCPU)
	assert.Equal(t, original.AllocatedMemory, copied.AllocatedMemory)
}

func TestResourcesConditionDeepCopyNil(t *testing.T) {
	// Test DeepCopy with nil
	var original *ResourcesCondition
	copied := original
	assert.Nil(t, copied)
}

func TestClusterDeepCopy(t *testing.T) {
	// Create a test cluster
	original := &Cluster{
		TypeMeta: v1.TypeMeta{
			Kind:       "Cluster",
			APIVersion: "openfuyao.io/v1beta1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
			Labels: map[string]string{
				"environment": "test",
			},
		},
		Spec: ClusterSpec{
			UID:            "test-uid",
			APIEndpoint:    "https://test-api-endpoint",
			Kubeconfigpath: "/path/to/kubeconfig",
		},
		Status: ClusterStatus{
			Condition: []v1.Condition{
				{
					Type:   "Ready",
					Status: v1.ConditionTrue,
				},
			},
			ResourceList: ResourcesCondition{
				AllocatableCPU:    "4",
				AllocatableMemory: "8Gi",
			},
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()
	assert.NotNil(t, copied)
	assert.Equal(t, original.Name, copied.Name)
	assert.Equal(t, original.Namespace, copied.Namespace)
	assert.Equal(t, original.Kind, copied.Kind)
	assert.Equal(t, original.APIVersion, copied.APIVersion)
	assert.Equal(t, original.Spec.UID, copied.Spec.UID)
	assert.Equal(t, original.Spec.APIEndpoint, copied.Spec.APIEndpoint)
	assert.Equal(t, original.Spec.Kubeconfigpath, copied.Spec.Kubeconfigpath)
	assert.Equal(t, len(original.Status.Condition), len(copied.Status.Condition))
	assert.Equal(t, original.Status.ResourceList, copied.Status.ResourceList)

	// Verify it's a deep copy by modifying the original
	original.ObjectMeta.Labels["environment"] = "production"
	assert.Equal(t, "test", copied.ObjectMeta.Labels["environment"])
	assert.Equal(t, "production", original.ObjectMeta.Labels["environment"])
}

func TestClusterDeepCopyInto(t *testing.T) {
	// Create a test cluster
	original := &Cluster{
		TypeMeta: v1.TypeMeta{
			APIVersion: "openfuyao.io/v1beta1",
			Kind:       "Cluster",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: ClusterSpec{
			APIEndpoint:    "https://test-api-endpoint",
			UID:            "test-uid",
			Kubeconfigpath: "/path/to/kubeconfig",
		},
		Status: ClusterStatus{
			ResourceList: ResourcesCondition{
				AllocatableMemory: "8Gi",
				AllocatableCPU:    "4",
			},
		},
	}

	// Test DeepCopyInto
	copied := &Cluster{}
	original.DeepCopyInto(copied)

	assert.Equal(t, original.Name, copied.Name)
	assert.Equal(t, original.Namespace, copied.Namespace)
	assert.Equal(t, original.Kind, copied.Kind)
	assert.Equal(t, original.APIVersion, copied.APIVersion)
	assert.Equal(t, original.Spec.UID, copied.Spec.UID)
	assert.Equal(t, original.Spec.APIEndpoint, copied.Spec.APIEndpoint)
	assert.Equal(t, original.Spec.Kubeconfigpath, copied.Spec.Kubeconfigpath)
	assert.Equal(t, original.Status.ResourceList, copied.Status.ResourceList)
}

func TestClusterDeepCopyObject(t *testing.T) {
	// Create a test cluster
	original := &Cluster{
		TypeMeta: v1.TypeMeta{
			Kind:       "Cluster",
			APIVersion: "openfuyao.io/v1beta1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: ClusterSpec{
			UID:            "test-uid",
			APIEndpoint:    "https://test-api-endpoint",
			Kubeconfigpath: "/path/to/kubeconfig",
		},
		Status: ClusterStatus{
			ResourceList: ResourcesCondition{
				AllocatableCPU:    "4",
				AllocatableMemory: "8Gi",
			},
		},
	}

	// Test DeepCopyObject
	obj := original.DeepCopyObject()
	assert.NotNil(t, obj)

	// Cast back to Cluster
	copied, ok := obj.(*Cluster)
	assert.True(t, ok)
	assert.Equal(t, original.Name, copied.Name)
	assert.Equal(t, original.Namespace, copied.Namespace)
	assert.Equal(t, original.Spec.UID, copied.Spec.UID)
	assert.Equal(t, original.Spec.APIEndpoint, copied.Spec.APIEndpoint)
	assert.Equal(t, original.Status.ResourceList, copied.Status.ResourceList)
}

func TestClusterListDeepCopy(t *testing.T) {
	// Create a test cluster list
	original := &ClusterList{
		TypeMeta: v1.TypeMeta{
			APIVersion: "openfuyao.io/v1beta1",
			Kind:       "ClusterList",
		},
		ListMeta: v1.ListMeta{
			ResourceVersion: "1",
		},
		Items: []Cluster{
			{
				Spec: ClusterSpec{
					UID: "uid-1",
				},
				ObjectMeta: v1.ObjectMeta{
					Name: "cluster-1",
				},
				Status: ClusterStatus{
					ResourceList: ResourcesCondition{
						AllocatableCPU: "4",
					},
				},
			},
			{
				ObjectMeta: v1.ObjectMeta{
					Name: "cluster-2",
				},
				Spec: ClusterSpec{
					UID: "uid-2",
				},
				Status: ClusterStatus{
					ResourceList: ResourcesCondition{
						AllocatableMemory: "8Gi",
					},
				},
			},
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()
	assert.NotNil(t, copied)
	assert.Equal(t, original.Kind, copied.Kind)
	assert.Equal(t, original.APIVersion, copied.APIVersion)
	assert.Equal(t, len(original.Items), len(copied.Items))
	assert.Equal(t, original.Items[0].Name, copied.Items[0].Name)
	assert.Equal(t, original.Items[1].Name, copied.Items[1].Name)
	assert.Equal(t, original.Items[0].Spec.UID, copied.Items[0].Spec.UID)
	assert.Equal(t, original.Items[1].Spec.UID, copied.Items[1].Spec.UID)

	// Verify deep copy by modifying original
	original.Items[0].Spec.UID = "new-uid"
	assert.Equal(t, "uid-1", copied.Items[0].Spec.UID)
	assert.Equal(t, "new-uid", original.Items[0].Spec.UID)
}

func TestClusterListDeepCopyInto(t *testing.T) {
	// Create a test cluster list
	original := &ClusterList{
		TypeMeta: v1.TypeMeta{
			APIVersion: "openfuyao.io/v1beta1",
			Kind:       "ClusterList",
		},
		ListMeta: v1.ListMeta{
			ResourceVersion: "2",
		},
		Items: []Cluster{
			{
				ObjectMeta: v1.ObjectMeta{
					Name: "cluster-1",
				},
				Status: ClusterStatus{
					ResourceList: ResourcesCondition{
						AllocatableCPU: "4",
					},
				},
				Spec: ClusterSpec{
					UID: "uid-1",
				},
			},
		},
	}

	// Test DeepCopyInto
	copied := &ClusterList{}
	original.DeepCopyInto(copied)

	assert.Equal(t, original.Kind, copied.Kind)
	assert.Equal(t, original.APIVersion, copied.APIVersion)
	assert.Equal(t, len(original.Items), len(copied.Items))
	assert.Equal(t, original.Items[0].Name, copied.Items[0].Name)
	assert.Equal(t, original.Items[0].Spec.UID, copied.Items[0].Spec.UID)
	assert.Equal(t, original.Items[0].Status.ResourceList.AllocatableCPU,
		copied.Items[0].Status.ResourceList.AllocatableCPU)
}

func TestClusterListDeepCopyObject(t *testing.T) {
	// Create a test cluster list
	original := &ClusterList{
		TypeMeta: v1.TypeMeta{
			Kind:       "ClusterList",
			APIVersion: "openfuyao.io/v1beta1",
		},
		ListMeta: v1.ListMeta{
			ResourceVersion: "1",
		},
		Items: []Cluster{
			{
				ObjectMeta: v1.ObjectMeta{
					Name: "cluster-1",
				},
				Spec: ClusterSpec{
					UID: "uid-1",
				},
				Status: ClusterStatus{
					ResourceList: ResourcesCondition{
						AllocatableCPU: "4",
					},
				},
			},
		},
	}

	// Test DeepCopyObject
	obj := original.DeepCopyObject()
	assert.NotNil(t, obj)

	// Cast back to ClusterList
	copied, ok := obj.(*ClusterList)
	assert.True(t, ok)
	assert.Equal(t, original.Kind, copied.Kind)
	assert.Equal(t, original.APIVersion, copied.APIVersion)
	assert.Equal(t, len(original.Items), len(copied.Items))
	assert.Equal(t, original.Items[0].Name, copied.Items[0].Name)
	assert.Equal(t, original.Items[0].Spec.UID, copied.Items[0].Spec.UID)
}
