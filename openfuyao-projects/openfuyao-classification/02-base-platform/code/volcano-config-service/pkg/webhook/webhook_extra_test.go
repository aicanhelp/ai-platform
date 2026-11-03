/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package webhook

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestMutatePodWithAnnotation_EmptyScheduler(t *testing.T) {
	server := &WhServer{}
	pod := createPodWithAnnotation("single-numa-node")
	ar := createAdmissionReviewWithObject("Pod", pod)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
	if result.Response.Patch != nil {
		var patches []patchOperation
		err := json.Unmarshal(result.Response.Patch, &patches)
		assert.NoError(t, err)
		assert.True(t, len(patches) > 0)
	}
}

func TestMutatePodWithAnnotation_DefaultScheduler(t *testing.T) {
	server := &WhServer{}
	pod := createPodWithAnnotation("single-numa-node")
	pod.Spec.SchedulerName = "default-scheduler"
	ar := createAdmissionReviewWithObject("Pod", pod)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
}

func TestMutatePodWithAnnotation_VolcanoScheduler(t *testing.T) {
	server := &WhServer{}
	pod := createPodWithAnnotation("single-numa-node")
	pod.Spec.SchedulerName = "volcano"
	ar := createAdmissionReviewWithObject("Pod", pod)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
}

func TestMutatePodWithAnnotation_CustomScheduler(t *testing.T) {
	server := &WhServer{}
	pod := createPodWithAnnotation("single-numa-node")
	pod.Spec.SchedulerName = "custom-scheduler"
	ar := createAdmissionReviewWithObject("Pod", pod)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
}

func TestMutateDeploymentWithAnnotation(t *testing.T) {
	server := &WhServer{}
	deployment := createDeploymentWithAnnotation("single-numa-node")
	deployment.Spec.Template.Spec.SchedulerName = "default-scheduler"
	ar := createAdmissionReviewWithObject("Deployment", deployment)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
}

func TestMutateStatefulSetWithAnnotation(t *testing.T) {
	server := &WhServer{}
	statefulSet := createStatefulSetWithAnnotation("single-numa-node")
	statefulSet.Spec.Template.Spec.SchedulerName = ""
	ar := createAdmissionReviewWithObject("StatefulSet", statefulSet)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
}

func TestMutateDaemonSetWithAnnotation(t *testing.T) {
	server := &WhServer{}
	daemonSet := createDaemonSetWithAnnotation("single-numa-node")
	daemonSet.Spec.Template.Spec.SchedulerName = "default-scheduler"
	ar := createAdmissionReviewWithObject("DaemonSet", daemonSet)
	result := server.mutate(ar)
	assert.True(t, result.Response.Allowed)
}

func TestGeneratePatches_InvalidObject(t *testing.T) {
	server := &WhServer{}
	req := &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Kind: "Pod"},
		Object: runtime.RawExtension{
			Raw: []byte("invalid json"),
		},
	}
	_, err := server.generatePatches(req)
	assert.Error(t, err)
}

func TestGeneratePatches_InvalidDeployment(t *testing.T) {
	server := &WhServer{}
	req := &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Kind: "Deployment"},
		Object: runtime.RawExtension{
			Raw: []byte("invalid json"),
		},
	}
	_, err := server.generatePatches(req)
	assert.Error(t, err)
}

func TestGeneratePatches_InvalidStatefulSet(t *testing.T) {
	server := &WhServer{}
	req := &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Kind: "StatefulSet"},
		Object: runtime.RawExtension{
			Raw: []byte("invalid json"),
		},
	}
	_, err := server.generatePatches(req)
	assert.Error(t, err)
}

func TestGeneratePatches_InvalidDaemonSet(t *testing.T) {
	server := &WhServer{}
	req := &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Kind: "DaemonSet"},
		Object: runtime.RawExtension{
			Raw: []byte("invalid json"),
		},
	}
	_, err := server.generatePatches(req)
	assert.Error(t, err)
}

func TestGeneratePatches_InvalidJob(t *testing.T) {
	server := &WhServer{}
	req := &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Kind: "Job"},
		Object: runtime.RawExtension{
			Raw: []byte("invalid json"),
		},
	}
	_, err := server.generatePatches(req)
	assert.Error(t, err)
}

func TestGeneratePatches_InvalidCronJob(t *testing.T) {
	server := &WhServer{}
	req := &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{Kind: "CronJob"},
		Object: runtime.RawExtension{
			Raw: []byte("invalid json"),
		},
	}
	_, err := server.generatePatches(req)
	assert.Error(t, err)
}
