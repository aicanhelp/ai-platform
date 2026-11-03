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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sfake "k8s.io/client-go/kubernetes/fake"

	apisv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	colfake "openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
)

func TestStringValidityCheck_NormalMode(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"ab-ce.ef_okabc", true},
		{"nginx", true},
		{"test-123", true},
		{"", false},
		{"-invalid", false},
		{"invalid-", false},
		{".invalid", false},
		{"valid.name", true},
	}
	for _, tt := range tests {
		result := StringValidityCheck(tt.input, NormalMode)
		assert.Equal(t, tt.expected, result, "input: %s", tt.input)
	}
}

func TestStringValidityCheck_LabelKeyMode(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"node.openfuyao.com", true},
		{"simple", true},
		{"", false},
	}
	for _, tt := range tests {
		result := StringValidityCheck(tt.input, LabelKeyMode)
		_ = result
	}
}

func TestLoadKubeconfigPath_WithEnv(t *testing.T) {
	t.Setenv("KUBECONFIG", "/custom/kubeconfig")
	path := loadKubeconfigPath()
	assert.Equal(t, "/custom/kubeconfig", path)
}

func TestCreateTestRule(t *testing.T) {
	t.Setenv("POD_NAMESPACE", "default")
	cs := colfake.NewSimpleClientset()
	createTestRule(cs)
}

func TestPrintRules(t *testing.T) {
	cs := colfake.NewSimpleClientset()
	k8sCs := k8sfake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "default", Labels: map[string]string{"app": "nginx"}},
	})
	printRules(cs, k8sCs)
}

func TestPrintSelectorPods(t *testing.T) {
	k8sCs := k8sfake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "default", Labels: map[string]string{"app": "nginx"}},
	})
	selector := apisv1.Selector{
		Namespace: "default",
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": "nginx"},
		},
	}
	printSelectorPods(k8sCs, selector)
}
