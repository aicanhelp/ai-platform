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
* Date: 2024-11-25
 */

package pod

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"gomodules.xyz/jsonpatch/v2"
	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-manager/webhook/helper"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/common"
)

// JsonPatch Patch内容 用于AdmissionResponse
type JsonPatch struct {
	OP    string      `json:"op"`
	Path  string      `json:"path"`
	Value interface{} `json:"value"`
}

// PodMutator PodMutator interface
type PodMutator interface {
	MutatePod(pod *v1.Pod) (allowed bool, err error)
}

// PodMutators PodMutators
type PodMutators struct {
	Mutators []PodMutator
}

// NewBEPodMutators 离线pod需要额外设置NRI需要使用的拓展资源信息注解
func NewBEPodMutators() (*PodMutators, error) {
	var mutators []PodMutator
	mutators = append(mutators, NewBEPodQoSMutator(), NewNodeAffinityMutator(),
		NewSchedulerMutator(), NewBEPodResourceMutator())
	return &PodMutators{
		Mutators: mutators,
	}, nil
}

// NewHLSPodMutators 配置HLS pod的mutator
func NewHLSPodMutators() (*PodMutators, error) {
	var mutators []PodMutator
	mutators = append(mutators, NewHLSPodQoSMutator(), NewNodeAffinityMutator(), NewSchedulerMutator())
	return &PodMutators{
		Mutators: mutators,
	}, nil
}

// NewLSPodMutators 配置LS pod的mutator
func NewLSPodMutators() (*PodMutators, error) {
	var mutators []PodMutator
	mutators = append(mutators, NewLSPodQoSMutator(), NewNodeAffinityMutator(), NewSchedulerMutator())
	return &PodMutators{
		Mutators: mutators,
	}, nil
}

// MutatingPodsJson MutatingPodsJson
func MutatingPodsJson(obj []byte, mutators *PodMutators) ([]byte, error) {
	oc := helper.NewDynamicObjectCreator()
	object, err := oc.NewObject(obj)
	if err != nil {
		return nil, err
	}
	mutatingObj, ok := object.(metav1.Object)
	if !ok {
		return nil, err
	}
	pod, ok := mutatingObj.(*v1.Pod)
	if !ok {
		return nil, err
	}

	klog.InfoS("start to mutate hls pod")
	for _, k := range mutators.Mutators {
		_, err := k.MutatePod(pod)
		if err != nil {
			return nil, err
		}
	}
	klog.Info("end to mutate hls pod")
	mutatedJSON, err := json.Marshal(mutatingObj)
	if err != nil {
		return nil, err
	}
	return mutatedJSON, nil
}

// MutatingAdmissionResponse 过渡解析
func MutatingAdmissionResponse(ar *admissionv1.AdmissionReview,
	mutatedJSON []byte) (*admissionv1.AdmissionResponse, error) {
	patches, err := jsonpatch.CreatePatch(ar.Request.Object.Raw, mutatedJSON)
	if err != nil {
		return nil, err
	}
	jsonPatch, err := json.Marshal(patches)
	if err != nil {
		return nil, err
	}
	patchType := admissionv1.PatchTypeJSONPatch

	return &admissionv1.AdmissionResponse{
		UID:       ar.Request.UID,
		Allowed:   true,
		Patch:     jsonPatch,
		PatchType: &patchType,
	}, nil
}

// ==== For test only= ====

// CreateTestPod is a helper that create test pods for testing.
func CreateTestPod() *v1.Pod {
	container := &v1.Container{
		Name: "container1",
		Resources: v1.ResourceRequirements{
			Requests: map[v1.ResourceName]resource.Quantity{
				common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				common.ExtenderResourceCPU:    resource.MustParse("2"),
			},
		},
	}

	initContainer := &v1.Container{
		Name: "containerInit",
		Resources: v1.ResourceRequirements{
			Requests: map[v1.ResourceName]resource.Quantity{
				common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				common.ExtenderResourceCPU:    resource.MustParse("2"),
			},
			Limits: map[v1.ResourceName]resource.Quantity{
				common.ExtenderResourceMemory: resource.MustParse("2Gi"),
				common.ExtenderResourceCPU:    resource.MustParse("2"),
			},
		},
	}

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "pod",
			Annotations: map[string]string{
				common.PriorityAnnotationKey: "true",
				common.QosClassAnnotationKey: "BE",
			},
		},
		Spec: v1.PodSpec{
			Containers:     []v1.Container{*container},
			InitContainers: []v1.Container{*initContainer},
		},
		Status: v1.PodStatus{QOSClass: v1.PodQOSBestEffort},
	}

	return pod
}

// CreateConfiguration creates test configuration for testing.
func CreateConfiguration(t *testing.T) []JsonPatch {
	configuration := createExtendedResourceConfig()
	marshal, err := json.Marshal(configuration)
	assert.NoError(t, err)

	expectJSONPatches := []JsonPatch{
		createAnnotationPatch(marshal),
		createAffinityPatch(),
		createSchedulerPatch(),
		createPriorityClassPatch(),
	}

	return expectJSONPatches
}

// createExtendedResourceConfig creates the extended resource configuration
func createExtendedResourceConfig() core.ExtendedResourceConfiguration {
	cpu := resource.MustParse("2")
	memory := resource.MustParse("2Gi")

	return core.ExtendedResourceConfiguration{
		InitContainers: createInitContainerConfig(cpu, memory),
		Containers:     createContainerConfig(cpu, memory),
	}
}

// createInitContainerConfig creates init container configuration
func createInitContainerConfig(cpu, memory resource.Quantity) map[string]core.ExtendedResourceItem {
	requests := map[v1.ResourceName]resource.Quantity{
		common.ExtenderResourceMemory: memory,
		common.ExtenderResourceCPU:    cpu,
	}
	limits := map[v1.ResourceName]resource.Quantity{
		common.ExtenderResourceMemory: memory,
		common.ExtenderResourceCPU:    cpu,
	}

	return map[string]core.ExtendedResourceItem{
		"containerInit": {
			Requests: requests,
			Limits:   limits,
		},
	}
}

// createContainerConfig creates regular container configuration
func createContainerConfig(cpu, memory resource.Quantity) map[string]core.ExtendedResourceItem {
	requests := map[v1.ResourceName]resource.Quantity{
		common.ExtenderResourceCPU:    cpu,
		common.ExtenderResourceMemory: memory,
	}

	return map[string]core.ExtendedResourceItem{
		"container1": {
			Requests: requests,
		},
	}
}

// createAnnotationPatch creates the annotation JSON patch
func createAnnotationPatch(marshal []byte) JsonPatch {
	return JsonPatch{
		OP:    "add",
		Path:  "/metadata/annotations/openfuyao.com~1extender-resource-cfg",
		Value: string(marshal),
	}
}

// createAffinityPatch creates the affinity JSON patch
func createAffinityPatch() JsonPatch {
	nodeAffinity := map[string]interface{}{
		"nodeAffinity": map[string]interface{}{
			"requiredDuringSchedulingIgnoredDuringExecution": map[string]interface{}{
				"nodeSelectorTerms": []map[string]interface{}{
					{
						"matchExpressions": []map[string]interface{}{
							{
								"key":      "node.openfuyao.com/colocation",
								"operator": "Exists",
							},
						},
					},
				},
			},
		},
	}

	return JsonPatch{
		OP:    "add",
		Path:  "/spec/affinity",
		Value: nodeAffinity,
	}
}

// createSchedulerPatch creates the scheduler name JSON patch
func createSchedulerPatch() JsonPatch {
	return JsonPatch{
		OP:    "add",
		Path:  "/spec/schedulerName",
		Value: "volcano",
	}
}

// createPriorityClassPatch creates the priority class JSON patch
func createPriorityClassPatch() JsonPatch {
	return JsonPatch{
		OP:    "add",
		Path:  "/spec/priorityClassName",
		Value: "priority-be",
	}
}
