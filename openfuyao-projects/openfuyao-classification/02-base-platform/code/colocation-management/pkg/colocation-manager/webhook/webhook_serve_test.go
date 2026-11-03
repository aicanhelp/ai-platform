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
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"

	"openfuyao.com/colocation-management/pkg/common"
)

func rawPod(t *testing.T, qos string) []byte {
	t.Helper()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "p",
			Annotations: map[string]string{common.QosClassAnnotationKey: qos},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "c",
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("1"),
					corev1.ResourceMemory: resource.MustParse("1Gi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("1"),
					corev1.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
		}}},
	}
	b, err := json.Marshal(pod)
	if err != nil {
		t.Fatalf("marshal pod: %v", err)
	}
	return b
}

func reviewFor(kind, qos string, op admissionv1.Operation, raw []byte) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			UID:       types.UID("uid-1"),
			Kind:      metav1.GroupVersionKind{Kind: kind},
			Operation: op,
			Object:    runtime.RawExtension{Raw: raw},
		},
	}
}

// ---- mutate dispatch ----

func TestMutateDispatch(t *testing.T) {
	w := &WebhookServer{}

	t.Run("delete operation", func(t *testing.T) {
		ar := reviewFor("Pod", "", admissionv1.Delete, nil)
		assert.NotNil(t, w.mutate(ar))
	})
	t.Run("kind node", func(t *testing.T) {
		ar := reviewFor("Node", "", admissionv1.Create, nil)
		assert.NotNil(t, w.mutate(ar))
	})
	t.Run("kind other", func(t *testing.T) {
		ar := reviewFor("Service", "", admissionv1.Create, nil)
		assert.NotNil(t, w.mutate(ar))
	})
}

// ---- handlePodMutation qos dispatch + error/default paths ----

func TestHandlePodMutation(t *testing.T) {
	w := &WebhookServer{}

	t.Run("malformed raw returns base response", func(t *testing.T) {
		ar := reviewFor("Pod", "", admissionv1.Create, []byte("not-json"))
		resp := w.handlePodMutation(ar)
		assert.NotNil(t, resp)
		assert.True(t, resp.Response.Allowed)
	})
	t.Run("no qos annotation -> default allowed", func(t *testing.T) {
		ar := reviewFor("Pod", "", admissionv1.Create, rawPod(t, ""))
		resp := w.handlePodMutation(ar)
		assert.NotNil(t, resp)
		assert.True(t, resp.Response.Allowed) // default path admits the pod
	})
	// With this test pod (1 CPU / 1Gi, limits==requests, no extender requests,
	// and a raw Object missing TypeMeta) the handlers reject every QoS class:
	//   HLS -> mutation fails (raw pod has no Kind), LS -> "cannot satisfy all
	//   three conditions", BE -> "must use extend resources".
	t.Run("HLS pod -> rejected", func(t *testing.T) {
		ar := reviewFor("Pod", "HLS", admissionv1.Create, rawPod(t, "HLS"))
		resp := w.handlePodMutation(ar)
		assert.NotNil(t, resp)
		assert.False(t, resp.Response.Allowed)
	})
	t.Run("LS pod -> rejected", func(t *testing.T) {
		ar := reviewFor("Pod", "LS", admissionv1.Create, rawPod(t, "LS"))
		resp := w.handlePodMutation(ar)
		assert.False(t, resp.Response.Allowed)
	})
	t.Run("BE pod -> rejected", func(t *testing.T) {
		ar := reviewFor("Pod", "BE", admissionv1.Create, rawPod(t, "BE"))
		resp := w.handlePodMutation(ar)
		assert.False(t, resp.Response.Allowed)
	})
}

// ---- validate dispatch ----

func TestValidate(t *testing.T) {
	w := &WebhookServer{}
	for _, kind := range []string{"Pod", "Node", "Other"} {
		ar := reviewFor(kind, "", admissionv1.Create, nil)
		assert.NotNil(t, w.validate(ar))
	}
}

// ---- Serve via httptest ----

func TestServe(t *testing.T) {
	w := &WebhookServer{}

	t.Run("valid mutate request", func(t *testing.T) {
		ar := reviewFor("Pod", "HLS", admissionv1.Create, rawPod(t, "HLS"))
		body, _ := json.Marshal(ar)
		req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		w.Serve(rec, req)
		assert.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("valid validate request", func(t *testing.T) {
		ar := reviewFor("Pod", "", admissionv1.Create, rawPod(t, ""))
		body, _ := json.Marshal(ar)
		req := httptest.NewRequest(http.MethodPost, "/validate", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		w.Serve(rec, req)
		assert.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("wrong content type -> 415", func(t *testing.T) {
		ar := reviewFor("Pod", "", admissionv1.Create, rawPod(t, ""))
		body, _ := json.Marshal(ar)
		req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader(body))
		req.Header.Set("Content-Type", "text/plain")
		rec := httptest.NewRecorder()
		w.Serve(rec, req)
		assert.Equal(t, http.StatusUnsupportedMediaType, rec.Code)
	})

	t.Run("malformed body -> returns without writing", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader([]byte("not-json")))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		w.Serve(rec, req)
		assert.Equal(t, http.StatusOK, rec.Code) // nothing written; default 200
		assert.Empty(t, rec.Body.String())
	})
}
