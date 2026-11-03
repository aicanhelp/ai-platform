/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package kubelet

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestFyResizeLimit(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	pcm := kl.containerManager.NewPodContainerManager()

	probe := &v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			HTTPGet: &v1.HTTPGetAction{
				Path:   "/",
				Scheme: v1.URISchemeHTTP,
			},
		},
		TimeoutSeconds:   1,
		PeriodSeconds:    10,
		SuccessThreshold: 1,
		FailureThreshold: 3,
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				ResizeLimitOnStarting: `[{"name":"container1","cpu":"500m"}]`,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "container1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					ReadinessProbe: probe,
				},
				{
					Name: "container2",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					ReadinessProbe: probe,
				},
			},
		},
	}
	var err error
	// Testing ineffective situations
	if pod, err = fyResizeLimit(pod, kl, pcm); err != nil {
		t.Errorf("Testing ineffective unexpected error: %v", err)
	}
	if cpuLimit := pod.Spec.Containers[0].Resources.Limits.Cpu().MilliValue(); cpuLimit == 500 {
		t.Errorf("Expected an unchanged value, but it took effect")
	}

	// Testing empty configuration situations
	pod.Annotations[ResizeLimitOnStarting] = ""
	if pod, err = fyResizeLimit(pod, kl, pcm); err != nil {
		t.Errorf("Testing empty configuration unexpected error: %v", err)
	}
	if cpuLimit := pod.Spec.Containers[0].Resources.Limits.Cpu().MilliValue(); cpuLimit != 1000 {
		t.Errorf("Expected an unchanged value with empy configuration")
	}

	// Testing incorrect configuration situations
	pod.Annotations[ResizeLimitOnStarting] = `[{"name":"container1","memory":"2000m"}]`
	if pod, err = fyResizeLimit(pod, kl, pcm); err != nil {
		t.Errorf("Testing incorrect configuration unexpected error: %v", err)
	}
	if cpuLimit := pod.Spec.Containers[0].Resources.Limits.Cpu().MilliValue(); cpuLimit != 1000 {
		t.Errorf("Expected an unchanged value with incorrect configuration")
	}

	// Testing no-existent container situations
	pod.Annotations[ResizeLimitOnStarting] = `[{"name":"no_container","cpu":"2000m"}]`
	if pod, err = fyResizeLimit(pod, kl, pcm); err != nil {
		t.Errorf("Testing no-existent container unexpected error: %v", err)
	}
	for _, rContainer := range pod.Spec.Containers {
		if cpuLimit := rContainer.Resources.Limits.Cpu().MilliValue(); cpuLimit != 1000 {
			t.Errorf("Expected an unchanged value with no-existent container")
		}
	}

	// Testing single legal configuration situations
	pod.Annotations[ResizeLimitOnStarting] = `[{"name":"container1","cpu":"2000m"}]`
	if pod, err = fyResizeLimit(pod, kl, pcm); err != nil {
		t.Errorf("Testing legal configuration unexpected error: %v", err)
	}
	if cpuLimit := pod.Spec.Containers[0].Resources.Limits.Cpu().MilliValue(); cpuLimit != 2000 {
		t.Errorf("Expected container1`s cpu is 2000, but got %v", cpuLimit)
	}
	if cpuLimit := pod.Spec.Containers[1].Resources.Limits.Cpu().MilliValue(); cpuLimit != 1000 {
		t.Errorf("Expected container2`s cpu is 1000, but got %v", cpuLimit)
	}

	// Testing all legal configuration situations
	pod.Annotations[ResizeLimitOnStarting] = `[{"name":"container1","cpu":"2000m"},{"name":"container2","cpu":"1500m"}]`
	if pod, err = fyResizeLimit(pod, kl, pcm); err != nil {
		t.Errorf("Testing legal configuration unexpected error: %v", err)
	}
	if cpuLimit := pod.Spec.Containers[0].Resources.Limits.Cpu().MilliValue(); cpuLimit != 2000 {
		t.Errorf("Expected container1`s cpu is 2000, but got %v", cpuLimit)
	}
	if cpuLimit := pod.Spec.Containers[1].Resources.Limits.Cpu().MilliValue(); cpuLimit != 1500 {
		t.Errorf("Expected container2`s cpu is 1500, but got %v", cpuLimit)
	}
}

func TestRecoverCpuLimit(t *testing.T) {
	pod := buildPod()
	limitMap := map[string]resource.Quantity{
		"container1": *resource.NewQuantity(2, resource.DecimalSI),
	}
	recoverCpuLimit(pod, limitMap)
	quantity := resource.NewQuantity(2, resource.DecimalSI)
	r := pod.Spec.Containers[0].Resources.Limits[v1.ResourceCPU]
	if r.String() != quantity.String() {
		t.Errorf("Expected the cpu limit to be %s, but got %s", quantity.String(), r.String())
	}
}

func buildPod() *v1.Pod {
	probe := &v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			HTTPGet: &v1.HTTPGetAction{
				Path:   "/",
				Scheme: v1.URISchemeHTTP,
			},
		},
		TimeoutSeconds:   1,
		PeriodSeconds:    10,
		SuccessThreshold: 1,
		FailureThreshold: 3,
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				ResizeLimitOnStarting: `[{"name":"container1","cpu":"500m"}]`,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "container1",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					ReadinessProbe: probe,
				},
				{
					Name: "container2",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
					ReadinessProbe: probe,
				},
			},
		},
	}
	return pod
}
