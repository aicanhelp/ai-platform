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


package controller

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func TestInfernexManagedInferencePoolMatchLabels_DirectPDUsesOpenFuyaoGroupID(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "qwen3-8b-pd"},
		Spec:       infernexv1alpha1.InferNexServiceSpec{},
	}
	spec := infernexv1alpha1.InferNexServiceSpec{
		Engine: testPDEngineSpec(),
	}
	got := infernexManagedInferencePoolMatchLabels(owner, spec)
	if len(got) != 2 {
		t.Fatalf("want two match labels, got %#v", got)
	}
	if v := got[igwapiv1.LabelKey(labelOpenFuyaoPDGroup)]; string(v) != "qwen3-8b-pd" {
		t.Fatalf("openfuyao.com/pdGroupID: got %q", v)
	}
	if v := got[igwapiv1.LabelKey(labelInfernexOwner)]; string(v) != "qwen3-8b-pd" {
		t.Fatalf("infernex.io/owner: got %q", v)
	}
}

func TestInfernexManagedInferencePoolMatchLabels_LinkedPDUsesAppKubernetesIOName(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "infsvc-wrap", Namespace: "ns-a"},
		Spec: infernexv1alpha1.InferNexServiceSpec{
			SourceRef: &infernexv1alpha1.SourceRef{
				Kind: "LLMInferenceService",
				Name: "my-llm",
			},
		},
	}
	spec := infernexv1alpha1.InferNexServiceSpec{
		Engine: testPDEnginePrefillOnly(),
	}
	got := infernexManagedInferencePoolMatchLabels(owner, spec)
	if len(got) != 2 {
		t.Fatalf("want two match labels, got %#v", got)
	}
	if v := got[igwapiv1.LabelKey(labelAppKubernetesIOName)]; string(v) != "my-llm" {
		t.Fatalf("app.kubernetes.io/name: got %q", v)
	}
	if v := got[igwapiv1.LabelKey(labelAppKubernetesIOPartOf)]; string(v) != valueKServeAppPartOf {
		t.Fatalf("app.kubernetes.io/part-of: got %q want %q", v, valueKServeAppPartOf)
	}
}

func TestInfernexManagedInferencePoolMatchLabels_AggregateUsesInfernexOwnerComponent(t *testing.T) {
	t.Parallel()
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "agg"},
		Spec:       infernexv1alpha1.InferNexServiceSpec{},
	}
	spec := infernexv1alpha1.InferNexServiceSpec{
		Engine: &infernexv1alpha1.InferenceEngineSpec{},
	}
	got := infernexManagedInferencePoolMatchLabels(owner, spec)
	if got[igwapiv1.LabelKey(labelInfernexOwner)] != "agg" {
		t.Fatalf("owner: got %#v", got)
	}
	if got[igwapiv1.LabelKey(labelInfernexComponent)] != valueInfernexEngineAggregate {
		t.Fatalf("component: got %#v", got)
	}
}
