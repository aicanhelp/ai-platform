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

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"openfuyao.com/colocation-management/pkg/common"
)

var expectedBEMutatorLength = 4
var expectedMutatorLength = 3

func TestNewBEPodMutators(t *testing.T) {
	mutators, _ := NewBEPodMutators()
	if len(mutators.Mutators) != expectedBEMutatorLength {
		t.Errorf("Expected be pod mutators to be %d, got %d", expectedBEMutatorLength, len(mutators.Mutators))
	}
}

func TestNewHLSPodMutators(t *testing.T) {
	mutators, _ := NewHLSPodMutators()
	if len(mutators.Mutators) != expectedMutatorLength {
		t.Errorf("Expected hls pod mutators to be %d, got %d", expectedMutatorLength, len(mutators.Mutators))
	}
}

func TestNewLSPodMutators(t *testing.T) {
	mutators, _ := NewLSPodMutators()
	if len(mutators.Mutators) != expectedMutatorLength {
		t.Errorf("Expected ls pod mutators to be %d, got %d", expectedMutatorLength, len(mutators.Mutators))
	}
}

func TestMutatePod(t *testing.T) {
	pod := CreateTestPod()
	pod.ObjectMeta.Annotations = map[string]string{
		common.PriorityAnnotationKey: "true",
		common.QosClassAnnotationKey: "BE",
	}
	rawPod, err := ToRawPod(pod)
	if err != nil {
		t.Errorf("Failed to marshal pod: %v", err)
	}
	ar := &admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			UID: "111",
			Kind: metav1.GroupVersionKind{
				Kind: "pod",
			},
			Object: runtime.RawExtension{
				Raw: rawPod,
			},
		},
	}
	mutators, _ := NewBEPodMutators()
	mutatedJSON, _ := MutatingPodsJson(rawPod, mutators)
	response, _ := MutatingAdmissionResponse(ar, mutatedJSON)

	expectJSONPatches := CreateConfiguration(t)
	if !patchesEqual(expectJSONPatches, ToPatchJson(response.Patch)) {
		t.Errorf("Expected patch to be %v, got %v", expectJSONPatches, ToPatchJson(response.Patch))
	}
}

func ToRawPod(pod *v1.Pod) ([]byte, error) {
	RawPodJson, err := json.Marshal(pod)
	return RawPodJson, err
}
func ToPatchJson(patch []byte) []JsonPatch {
	var jsonPatchs []JsonPatch
	err := json.Unmarshal(patch, &jsonPatchs)
	if err != nil {
		return nil
	}
	return jsonPatchs
}

func patchesEqual(expected, actual []JsonPatch) bool {
	if len(expected) != len(actual) {
		return false
	}
	expectedSet := make(map[string]bool)
	for _, patch := range expected {
		jsonBytes, err := json.Marshal(patch)
		if err != nil {
			return false
		}
		expectedSet[string(jsonBytes)] = true
	}
	for _, patch := range actual {
		jsonBytes, err := json.Marshal(patch)
		if err != nil {
			return false
		}
		if !expectedSet[string(jsonBytes)] {
			return false
		}
	}
	return true
}
