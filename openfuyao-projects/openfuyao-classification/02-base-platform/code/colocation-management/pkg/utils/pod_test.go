/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-08-18
 */

package utils

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestOfflinePod(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected bool
	}{
		{
			name: "annotation true",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "label true",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "both annotation and label true",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
					Labels: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "annotation false, label true",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.PriorityAnnotationKey: "false",
					},
					Labels: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
				},
			},
			expected: true,
		},
		{
			name:     "nil pod",
			pod:      nil,
			expected: false,
		},
		{
			name: "no annotations or labels",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := OfflinePod(tt.pod)
			if result != tt.expected {
				t.Errorf("OfflinePod() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestOfflinePodMeta(t *testing.T) {
	tests := []struct {
		name     string
		podMeta  *metav1.PartialObjectMetadata
		expected bool
	}{
		{
			name: "annotation true",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "label true",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						common.PriorityAnnotationKey: "true",
					},
				},
			},
			expected: true,
		},
		{
			name: "no annotations or labels",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected: false,
		},
		{
			name:     "nil podMeta",
			podMeta:  nil,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := OfflinePodMeta(tt.podMeta)
			if result != tt.expected {
				t.Errorf("OfflinePodMeta() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestContributivePod(t *testing.T) {
	tests := []struct {
		name     string
		phase    corev1.PodPhase
		expected bool
	}{
		{"Running", corev1.PodRunning, true},
		{"Pending", corev1.PodPending, true},
		{"Succeeded", corev1.PodSucceeded, false},
		{"Failed", corev1.PodFailed, false},
		{"Unknown", corev1.PodUnknown, true},
		{"Empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContributivePod(tt.phase)
			if result != tt.expected {
				t.Errorf("ContributivePod(%s) = %v, want %v", tt.phase, result, tt.expected)
			}
		})
	}
}

func TestGetPodQosLevelByName(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		expected    common.QosLevel
	}{
		{
			name:        "QosBe",
			annotations: map[string]string{common.QosClassAnnotationKey: "BE"},
			expected:    common.QosBe,
		},
		{
			name:        "QosLs",
			annotations: map[string]string{common.QosClassAnnotationKey: "LS"},
			expected:    common.QosLs,
		},
		{
			name:        "QosHLS",
			annotations: map[string]string{common.QosClassAnnotationKey: "HLS"},
			expected:    common.QosHLS,
		},
		{
			name:        "invalid Qos",
			annotations: map[string]string{common.QosClassAnnotationKey: "Invalid"},
			expected:    common.QosNone,
		},
		{
			name:        "no annotation",
			annotations: map[string]string{},
			expected:    common.QosNone,
		},
		{
			name:        "nil annotations",
			annotations: nil,
			expected:    common.QosNone,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetPodQosLevelByName(tt.annotations)
			if result != tt.expected {
				t.Errorf("GetPodQosLevelByName() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetPodQosLevelByKubeQos(t *testing.T) {
	tests := []struct {
		name     string
		kubeQos  corev1.PodQOSClass
		expected common.QosLevel
	}{
		{"Guaranteed", corev1.PodQOSGuaranteed, common.QosNone},
		{"Burstable", corev1.PodQOSBurstable, common.QosLs},
		{"BestEffort", corev1.PodQOSBestEffort, common.QosBe},
		{"Empty", "", common.QosNone},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetPodQosLevelByKubeQos(tt.kubeQos)
			if result != tt.expected {
				t.Errorf("GetPodQosLevelByKubeQos(%s) = %v, want %v", tt.kubeQos, result, tt.expected)
			}
		})
	}
}

func TestGetQosLevelByPod(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected common.QosLevel
	}{
		{
			name: "with Qos annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "BE",
					},
				},
			},
			expected: common.QosBe,
		},
		{
			name: "without Qos annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected: common.QosNone,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetQosLevelByPod(tt.pod)
			if result != tt.expected {
				t.Errorf("GetQosLevelByPod() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetKubePodQosClass(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected corev1.PodQOSClass
	}{
		{
			name: "already has QOSClass",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					QOSClass: corev1.PodQOSGuaranteed,
				},
			},
			expected: corev1.PodQOSGuaranteed,
		},
		{
			name: "BestEffort - no resources",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{},
						},
					},
				},
			},
			expected: corev1.PodQOSBestEffort,
		},
		{
			name: "Guaranteed - equal requests and limits",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("100Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
			expected: corev1.PodQOSGuaranteed,
		},
		{
			name: "BestEffort - unsupported requests and limits",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									common.ExtenderResourceCPU:    resource.MustParse("100m"),
									common.ExtenderResourceMemory: resource.MustParse("100Mi"),
								},
								Limits: corev1.ResourceList{
									common.ExtenderResourceCPU:    resource.MustParse("100m"),
									common.ExtenderResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
			expected: corev1.PodQOSBestEffort,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetKubePodQosClass(tt.pod)
			if result != tt.expected {
				t.Errorf("GetKubePodQosClass() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetAllContainers(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "container1"},
				{Name: "container2"},
			},
			InitContainers: []corev1.Container{
				{Name: "init1"},
			},
		},
	}

	containers := getAllContainers(pod)
	if len(containers) != 3 {
		t.Errorf("getAllContainers() returned %d containers, want 3", len(containers))
	}
}

func TestUsesUnsupportedQoSResources(t *testing.T) {
	tests := []struct {
		name       string
		containers []corev1.Container
		expected   bool
	}{
		{
			name: "supported resources only",
			containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "unsupported resource",
			containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							common.ExtenderResourceMemory: resource.MustParse("1Gi"),
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := usesUnsupportedQoSResources(tt.containers)
			if result != tt.expected {
				t.Errorf("usesUnsupportedQoSResources() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateTotalRequestsAndLimits(t *testing.T) {
	containers := []corev1.Container{
		{
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("100m"),
					corev1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("200m"),
					corev1.ResourceMemory: resource.MustParse("200Mi"),
				},
			},
		},
		{
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU: resource.MustParse("50m"),
				},
			},
		},
	}

	requests := calculateTotalRequests(containers)
	if requests.Cpu().MilliValue() != 150 {
		t.Errorf("calculateTotalRequests() CPU = %dm, want 150m", requests.Cpu().MilliValue())
	}

	limits := calculateTotalLimits(containers)
	if limits.Cpu().MilliValue() != 200 {
		t.Errorf("calculateTotalLimits() CPU = %dm, want 200m", limits.Cpu().MilliValue())
	}
}

func TestIsSupportedQoSComputeResource(t *testing.T) {
	tests := []struct {
		name     string
		resource corev1.ResourceName
		expected bool
	}{
		{"CPU", corev1.ResourceCPU, true},
		{"Memory", corev1.ResourceMemory, true},
		{"be-cpu", common.ExtenderResourceCPU, false},
		{"be-memory", common.ExtenderResourceMemory, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isSupportedQoSComputeResource(tt.resource)
			if result != tt.expected {
				t.Errorf("isSupportedQoSComputeResource(%s) = %v, want %v", tt.resource, result, tt.expected)
			}
		})
	}
}

func TestHLSPod(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected bool
	}{
		{
			name: "HLS pod",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "HLS",
					},
				},
			},
			expected: true,
		},
		{
			name: "not HLS pod",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "BE",
					},
				},
			},
			expected: false,
		},
		{
			name:     "nil pod",
			pod:      nil,
			expected: false,
		},
		{
			name: "nil annotations",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected: false,
		},
		{
			name: "annotations without qos key",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{"other": "value"},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HLSPod(tt.pod)
			if result != tt.expected {
				t.Errorf("HLSPod() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestHLSPodMeta(t *testing.T) {
	tests := []struct {
		name     string
		podMeta  *metav1.PartialObjectMetadata
		expected bool
	}{
		{
			name: "HLS pod meta",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "HLS",
					},
				},
			},
			expected: true,
		},
		{
			name: "nil annotations",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected: false,
		},
		{
			name: "annotation present but not HLS",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						common.QosClassAnnotationKey: "BE",
					},
				},
			},
			expected: false,
		},
		{
			name: "annotations present without qos key",
			podMeta: &metav1.PartialObjectMetadata{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"some-other-key": "value",
					},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HLSPodMeta(tt.podMeta)
			if result != tt.expected {
				t.Errorf("HLSPodMeta() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestMergeResourceLists(t *testing.T) {
	list1 := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("100m"),
		corev1.ResourceMemory: resource.MustParse("100Mi"),
	}
	list2 := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("50m"),
		corev1.ResourceMemory: resource.MustParse("200Mi"),
	}

	result := mergeResourceLists(list1, list2)
	if result.Cpu().MilliValue() != 150 {
		t.Errorf("mergeResourceLists() CPU = %dm, want 150m", result.Cpu().MilliValue())
	}
	if result.Memory().Value() != 300*1024*1024 {
		t.Errorf("mergeResourceLists() Memory = %d, want 300Mi", result.Memory().Value()/(1024*1024))
	}
}

func TestHasNoResourceRequirements(t *testing.T) {
	tests := []struct {
		name     string
		requests corev1.ResourceList
		limits   corev1.ResourceList
		expected bool
	}{
		{
			name:     "both empty",
			requests: corev1.ResourceList{},
			limits:   corev1.ResourceList{},
			expected: true,
		},
		{
			name: "requests not empty",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("100m"),
			},
			limits:   corev1.ResourceList{},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasNoResourceRequirements(tt.requests, tt.limits)
			if result != tt.expected {
				t.Errorf("hasNoResourceRequirements() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsGuaranteedQoS(t *testing.T) {
	tests := []struct {
		name     string
		requests corev1.ResourceList
		limits   corev1.ResourceList
		expected bool
	}{
		{
			name: "guaranteed - equal",
			requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("100Mi"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("100Mi"),
			},
			expected: true,
		},
		{
			name: "not guaranteed - different values",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("100m"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("200m"),
			},
			expected: false,
		},
		{
			name: "not guaranteed - len mismatch (request missing)",
			requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("100Mi"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("100m"),
			},
			expected: false,
		},
		{
			name: "not guaranteed - limit missing for requested resource",
			requests: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("100m"),
			},
			limits: corev1.ResourceList{
				corev1.ResourceMemory: resource.MustParse("100Mi"),
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isGuaranteedQoS(tt.requests, tt.limits)
			if result != tt.expected {
				t.Errorf("isGuaranteedQoS() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestGetKubePodQosClassBurstable covers the Burstable return path of
// GetKubePodQosClass (requests present, limits present but not equal/matching).
func TestGetKubePodQosClassBurstable(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected corev1.PodQOSClass
	}{
		{
			name: "Burstable - request without matching limit",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("100m"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
			expected: corev1.PodQOSBurstable,
		},
		{
			name: "Burstable - only requests, no limits",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
				},
			},
			expected: corev1.PodQOSBurstable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetKubePodQosClass(tt.pod)
			if result != tt.expected {
				t.Errorf("GetKubePodQosClass() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestUsesUnsupportedQoSResourcesLimits covers the Limits-loop branch of
// usesUnsupportedQoSResources (the Requests branch is already covered).
func TestUsesUnsupportedQoSResourcesLimits(t *testing.T) {
	tests := []struct {
		name       string
		containers []corev1.Container
		expected   bool
	}{
		{
			name: "unsupported resource in limits only",
			containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							common.ExtenderResourceCPU: resource.MustParse("1"),
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "supported resources in both requests and limits",
			containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("100m"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
			expected: false,
		},
		{
			name:       "empty containers slice",
			containers: []corev1.Container{},
			expected:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := usesUnsupportedQoSResources(tt.containers)
			if result != tt.expected {
				t.Errorf("usesUnsupportedQoSResources() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// TestMergeResourceListsNilFirstList covers the branch of mergeResourceLists
// where list1 (the DeepCopy result) is nil.
func TestMergeResourceListsNilFirstList(t *testing.T) {
	var nilList corev1.ResourceList = nil
	list2 := corev1.ResourceList{
		corev1.ResourceCPU:    resource.MustParse("100m"),
		corev1.ResourceMemory: resource.MustParse("200Mi"),
	}

	result := mergeResourceLists(nilList, list2)
	if result.Cpu().MilliValue() != 100 {
		t.Errorf("mergeResourceLists(nil, list2) CPU = %dm, want 100m", result.Cpu().MilliValue())
	}
	if result.Memory().Value() != 200*1024*1024 {
		t.Errorf("mergeResourceLists(nil, list2) Memory = %d, want 200Mi", result.Memory().Value())
	}

	// both nil -> result is a non-nil empty ResourceList, no panic.
	resultBoth := mergeResourceLists(corev1.ResourceList(nil), corev1.ResourceList(nil))
	if resultBoth == nil {
		t.Error("Expected non-nil result when both inputs are nil")
	}
	if len(resultBoth) != 0 {
		t.Errorf("Expected empty result, got %d entries", len(resultBoth))
	}
}
