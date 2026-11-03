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

package webhook

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"openfuyao.com/colocation-management/pkg/colocation-manager/webhook/pod"
	"openfuyao.com/colocation-management/pkg/common"
)

func TestVerifyExtendResource(t *testing.T) {

	testPod := pod.CreateTestPod()
	rawPod, err := ToRawPod(testPod)
	if err != nil {
		t.Errorf("ToRawPod error")
	}

	case1 := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Request: &admissionv1.AdmissionRequest{
			UID: "111",
			Kind: metav1.GroupVersionKind{
				Kind: "Pod",
			},
			Object: runtime.RawExtension{
				Raw: rawPod,
			},
		},
	}
	whsvr := &WebhookServer{
		Server: &http.Server{
			Addr:      fmt.Sprintf(":%v", "111"),
			TLSConfig: &tls.Config{Certificates: []tls.Certificate{}},
		},
	}

	mutate := whsvr.mutate(case1)

	if !reflect.DeepEqual(mutate.Response.Allowed, true) {
		t.Errorf("Expected  to be %v, got %v", true, mutate.Response.Allowed)
	}

}

var whsvr = &WebhookServer{
	Server: &http.Server{
		Addr:      fmt.Sprintf(":%v", "111"),
		TLSConfig: &tls.Config{Certificates: []tls.Certificate{}},
	},
}

func getBaseAdmissionReview() *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
	}
}

func TestToAdmissionErrorResponse1(t *testing.T) {
	// 测试删除pod
	arcase1 := getBaseAdmissionReview()
	arcase1.Request = &admissionv1.AdmissionRequest{
		UID:       "111",
		Operation: admissionv1.Delete,
	}

	primt := getBaseAdmissionReview()
	primt = &admissionv1.AdmissionReview{
		Response: &admissionv1.AdmissionResponse{
			UID:     "111",
			Allowed: true,
		},
	}
	mutate := whsvr.mutate(arcase1)
	expectedJSON, err := json.Marshal(primt.Response)
	if err != nil {
		t.Fatalf("Failed to marshal expected object: %v", err)
	}
	actualJSON, err := json.Marshal(mutate.Response)
	if err != nil {
		t.Fatalf("Failed to marshal actual object: %v", err)
	}

	if string(expectedJSON) != string(actualJSON) {
		t.Errorf("Expected: %s, Got: %s", string(expectedJSON), string(actualJSON))
	}
}

func TestToAdmissionErrorResponse2(t *testing.T) {
	testPod := pod.CreateTestPod()
	testPod.Annotations = map[string]string{
		common.PriorityAnnotationKey: "true",
		common.QosClassAnnotationKey: "BE",
	}

	rawPod, err := ToRawPod(testPod)
	if err != nil {
		t.Errorf("ToRawPod error")
	}
	arcase2 := getBaseAdmissionReview()
	arcase2.Request = &admissionv1.AdmissionRequest{
		UID: "111",
		Kind: metav1.GroupVersionKind{
			Kind: "Pod",
		},
		Object: runtime.RawExtension{
			Raw: rawPod,
		},
	}

	mutate2 := whsvr.mutate(arcase2)
	var warning []string
	expectJsonPatches := pod.CreateConfiguration(t)
	expectJSONPatchesByte, err := json.Marshal(expectJsonPatches)
	assert.NoError(t, err)
	patchType := admissionv1.PatchTypeJSONPatch

	expect2 := getBaseAdmissionReview()
	expect2.Response = &admissionv1.AdmissionResponse{
		UID:              "111",
		Allowed:          true,
		Result:           nil,
		Patch:            expectJSONPatchesByte,
		PatchType:        &patchType,
		AuditAnnotations: map[string]string{},
		Warnings:         warning,
	}

	expectedJSON, err := json.Marshal(expect2.Response.Allowed)
	if err != nil {
		t.Fatalf("Failed to marshal expected object: %v", err)
	}
	actualJSON, err := json.Marshal(mutate2.Response.Allowed)
	if err != nil {
		t.Fatalf("Failed to marshal actual object: %v", err)
	}

	if string(expectedJSON) != string(actualJSON) {
		t.Errorf("Expected: %s, Got: %s", string(expectedJSON), string(actualJSON))
	}

}

func ToRawPod(pod *v1.Pod) ([]byte, error) {
	RawPodJson, err := json.Marshal(pod)
	if err != nil {
		return nil, err
	}
	return RawPodJson, err
}

// JsonPatch Patch内容 用于AdmissionResponse
type JsonPatch struct {
	OP    string      `json:"op"`
	Path  string      `json:"path"`
	Value interface{} `json:"value"`
}

// ToAdmissionErrorResponse 测试
func TestToAdmissionErrorResponse(t *testing.T) {
	review := getBaseAdmissionReview()
	review.Request = &admissionv1.AdmissionRequest{
		UID: "111",
	}
	resp := ToAdmissionErrorResponse(review, fmt.Errorf("test"))
	assert.Equal(t, false, resp.Allowed)
}

// ToAdmissionReview 测试
func TestToAdmissionReview(t *testing.T) {
	resp := &admissionv1.AdmissionResponse{
		UID: "111",
	}
	review := ToAdmissionReview(resp)

	expect := getBaseAdmissionReview()
	expect.Response = &admissionv1.AdmissionResponse{
		UID: "111",
	}
	assert.Equal(t, expect, review)
}

// NewAdmissionReview 测试
func TestNewAdmissionReview(t *testing.T) {
	review := newAdmissionReview("111")
	expect := admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: &admissionv1.AdmissionResponse{
			UID:     "111",
			Allowed: true,
		},
	}
	assert.Equal(t, expect, review)
}
