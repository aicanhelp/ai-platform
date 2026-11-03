/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
 */

// Package webhook defined mutate and validate rules
package webhook

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

func TestWhServerServe(t *testing.T) {
	server := &WhServer{}

	tests := []struct {
		name           string
		path           string
		contentType    string
		body           interface{}
		expectedStatus int
		checkResponse  bool
	}{
		{
			name:           "valid mutate request",
			path:           "/mutate",
			contentType:    "application/json",
			body:           createAdmissionReviewWithObject("Pod", createTestPod()),
			expectedStatus: http.StatusOK,
			checkResponse:  true,
		},
		{
			name:           "invalid content type",
			path:           "/mutate",
			contentType:    "text/plain",
			body:           createAdmissionReviewWithObject("Pod", createTestPod()),
			expectedStatus: http.StatusUnsupportedMediaType,
			checkResponse:  false,
		},
		{
			name:           "invalid JSON body",
			path:           "/mutate",
			contentType:    "application/json",
			body:           "invalid-json",
			expectedStatus: http.StatusOK,
			checkResponse:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req *http.Request
			if tt.body == "invalid-json" {
				req = httptest.NewRequest("POST", tt.path, bytes.NewBufferString("invalid-json"))
			} else {
				bodyBytes, err := json.Marshal(tt.body)
				if err != nil {
					t.Errorf("failed to unmarshal bodyBytes: %v", err)
				}
				req = httptest.NewRequest("POST", tt.path, bytes.NewBuffer(bodyBytes))
			}

			if req != nil {
				req.Header.Set("Content-Type", tt.contentType)
			} else {
				// 处理 req 为 nil 的情况
				fmt.Println("req 是 nil，无法设置 header")
			}

			rr := httptest.NewRecorder()
			server.Serve(rr, req)

			if tt.expectedStatus == http.StatusUnsupportedMediaType {
				if rr.Code != tt.expectedStatus {
					t.Errorf("expected status %d, got %d", tt.expectedStatus, rr.Code)
				}
			} else if tt.checkResponse {
				// For valid requests, check if response is valid AdmissionReview
				var response admissionv1.AdmissionReview
				if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
					t.Errorf("failed to unmarshal response: %v", err)
				}
				if response.Response == nil {
					t.Error("response should not be nil")
				}
			} else {
				t.Logf("response ok")
			}
		})
	}
}

func TestWhServerMutate(t *testing.T) {
	server := &WhServer{}

	tests := []struct {
		name            string
		kind            string
		object          runtime.Object
		expectedAllowed bool
	}{
		{
			name:            "mutate pod",
			kind:            "Pod",
			object:          createTestPod(),
			expectedAllowed: true,
		},
		{
			name:            "mutate statefulset",
			kind:            "StatefulSet",
			object:          createTestStatefulSet(),
			expectedAllowed: true,
		},
		{
			name:            "mutate deployment",
			kind:            "Deployment",
			object:          createTestDeployment(),
			expectedAllowed: true,
		},
		{
			name:            "mutate daemonset",
			kind:            "DaemonSet",
			object:          createTestDaemonSet(),
			expectedAllowed: true,
		},
		{
			name:            "unsupported kind",
			kind:            "ConfigMap",
			object:          &corev1.ConfigMap{},
			expectedAllowed: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ar := createAdmissionReviewWithObject(tt.kind, tt.object)
			result := server.mutate(ar)

			if result.Response.Allowed != tt.expectedAllowed {
				t.Errorf("expected allowed=%v, got %v", tt.expectedAllowed, result.Response.Allowed)
			}
		})
	}
}

func TestBuildAdmissionResponse(t *testing.T) {
	server := &WhServer{}
	ar := createTestAdmissionReview("Pod")

	tests := []struct {
		name    string
		patches []patchOperation
	}{
		{
			name:    "no patches",
			patches: nil,
		},
		{
			name: "with patches",
			patches: []patchOperation{
				{Op: "add", Path: "/metadata/annotations", Value: map[string]string{}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := server.buildAdmissionResponse(ar, tt.patches)

			if !result.Response.Allowed {
				t.Errorf("expected allowed=true, got %v", result.Response.Allowed)
			}

			hasPatches := result.Response.Patch != nil
			expectPatches := tt.patches != nil && len(tt.patches) > 0
			if hasPatches != expectPatches {
				t.Errorf("expected patches=%v, got %v", expectPatches, hasPatches)
			}
		})
	}
}

func createTestAdmissionReview(kind string) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			UID: types.UID("test-uid"),
			Kind: metav1.GroupVersionKind{
				Kind: kind,
			},
			Name:      "test-object",
			Namespace: "default",
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
	}
}

func createAdmissionReviewWithObject(kind string, obj runtime.Object) *admissionv1.AdmissionReview {
	objBytes, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}

	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Request: &admissionv1.AdmissionRequest{
			UID: types.UID("test-uid"),
			Kind: metav1.GroupVersionKind{
				Kind: kind,
			},
			Name:      "test-object",
			Namespace: "default",
			Object: runtime.RawExtension{
				Raw: objBytes,
			},
		},
	}
}

func createTestPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "test-container",
					Image: "test-image",
				},
			},
		},
	}
}

func createPodWithAnnotation(numaTopologyPolicyAnnotation string) *corev1.Pod {
	pod := createTestPod()
	pod.Annotations = map[string]string{
		NumaTopologyPolicyAnnotation: numaTopologyPolicyAnnotation,
	}
	return pod
}

func createTestDeployment() *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "default",
		},
		Spec: appsv1.DeploymentSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "test-container-deployment",
							Image: "test-image",
						},
					},
				},
			},
		},
	}
}

func createDeploymentWithAnnotation(numaTopologyPolicyAnnotation string) *appsv1.Deployment {
	deployment := createTestDeployment()
	deployment.Spec.Template.Annotations = map[string]string{
		NumaTopologyPolicyAnnotation: numaTopologyPolicyAnnotation,
	}
	return deployment
}

func createTestStatefulSet() *appsv1.StatefulSet {
	return &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-statefulset",
			Namespace: "default",
		},
		Spec: appsv1.StatefulSetSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "test-container-statefulset",
							Image: "test-image",
						},
					},
				},
			},
		},
	}
}

func createStatefulSetWithAnnotation(numaTopologyPolicyAnnotation string) *appsv1.StatefulSet {
	statefulSet := createTestStatefulSet()
	statefulSet.Spec.Template.Annotations = map[string]string{
		NumaTopologyPolicyAnnotation: numaTopologyPolicyAnnotation,
	}
	return statefulSet
}

func createTestDaemonSet() *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-daemonset",
			Namespace: "default",
		},
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "test-container-daemonset",
							Image: "test-image",
						},
					},
				},
			},
		},
	}
}

func createDaemonSetWithAnnotation(numaTopologyPolicyAnnotation string) *appsv1.DaemonSet {
	daemonSet := createTestDaemonSet()
	daemonSet.Spec.Template.Annotations = map[string]string{
		NumaTopologyPolicyAnnotation: numaTopologyPolicyAnnotation,
	}
	return daemonSet
}

func getResultMessage(result *metav1.Status) string {
	if result == nil {
		return ""
	}
	return result.Message
}

func TestMutateJob(t *testing.T) {
	server := &WhServer{}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-job",
			Namespace: "default",
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "test-container", Image: "test-image"},
					},
				},
			},
		},
	}

	ar := createAdmissionReviewWithObject("Job", job)
	result := server.mutate(ar)

	if result.Response == nil {
		t.Fatal("response should not be nil")
	}
	if !result.Response.Allowed {
		t.Errorf("expected allowed=true, got %v", result.Response.Allowed)
	}
}

func TestMutateCronJob(t *testing.T) {
	server := &WhServer{}
	cronJob := &batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cronjob",
			Namespace: "default",
		},
		Spec: batchv1.CronJobSpec{
			Schedule: "*/5 * * * *",
			JobTemplate: batchv1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Name: "test-container", Image: "test-image"},
							},
						},
					},
				},
			},
		},
	}

	ar := createAdmissionReviewWithObject("CronJob", cronJob)
	result := server.mutate(ar)

	if result.Response == nil {
		t.Fatal("response should not be nil")
	}
	if !result.Response.Allowed {
		t.Errorf("expected allowed=true, got %v", result.Response.Allowed)
	}
}

func TestCreateAdmissionReview(t *testing.T) {
	server := &WhServer{}
	uid := types.UID("test-uid-123")
	result := server.createAdmissionReview(uid)

	if result == nil {
		t.Fatal("result should not be nil")
	}
	if result.Response == nil {
		t.Fatal("response should not be nil")
	}
	if result.Response.UID != uid {
		t.Errorf("expected UID %s, got %s", uid, result.Response.UID)
	}
	if !result.Response.Allowed {
		t.Errorf("expected allowed=true, got %v", result.Response.Allowed)
	}
}

func TestToAdmissionErrorResponse(t *testing.T) {
	ar := createTestAdmissionReview("Pod")
	testErr := fmt.Errorf("test error")

	resp := ToAdmissionErrorResponse(ar, testErr)
	if resp == nil {
		t.Fatal("response should not be nil")
	}
	if resp.UID != ar.Request.UID {
		t.Errorf("expected UID %s, got %s", ar.Request.UID, resp.UID)
	}
	if resp.Result == nil {
		t.Fatal("result should not be nil")
	}
	if resp.Result.Message != "test error" {
		t.Errorf("expected message 'test error', got '%s'", resp.Result.Message)
	}
}

func TestToAdmissionReview(t *testing.T) {
	resp := &admissionv1.AdmissionResponse{
		UID:     types.UID("test-uid"),
		Allowed: true,
	}

	review := ToAdmissionReview(resp)
	if review == nil {
		t.Fatal("review should not be nil")
	}
	if review.Response != resp {
		t.Error("review response should match input response")
	}
	if review.Kind != "AdmissionReview" {
		t.Errorf("expected kind 'AdmissionReview', got '%s'", review.Kind)
	}
}
