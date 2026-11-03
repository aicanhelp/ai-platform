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

package executor

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestGetAllPodsOnNode_FiltersTerminatedPods(t *testing.T) {
	runningPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "running-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}
	succeededPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "succeeded-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
		Status:     corev1.PodStatus{Phase: corev1.PodSucceeded},
	}
	failedPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "failed-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
		Status:     corev1.PodStatus{Phase: corev1.PodFailed},
	}
	pendingPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pending-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	clientset := fake.NewSimpleClientset(runningPod, succeededPod, failedPod, pendingPod)
	client := NewK8sClientForTest(clientset)

	pods, err := client.GetAllPodsOnNode(context.Background(), "node-1")
	if err != nil {
		t.Fatalf("GetAllPodsOnNode failed: %v", err)
	}

	// Should only return running and pending, exclude succeeded/failed
	if len(pods) != 2 {
		t.Errorf("expected 2 pods, got %d", len(pods))
	}

	names := map[string]bool{}
	for _, p := range pods {
		names[p.Name] = true
	}

	if !names["running-pod"] {
		t.Error("expected running-pod to be included")
	}
	if !names["pending-pod"] {
		t.Error("expected pending-pod to be included")
	}
	if names["succeeded-pod"] {
		t.Error("succeeded-pod should be filtered out")
	}
	if names["failed-pod"] {
		t.Error("failed-pod should be filtered out")
	}
}

func TestGetPodsInNamespace(t *testing.T) {
	pod1 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "app-pod",
			Namespace: "default",
			Labels:    map[string]string{"app": "test"},
		},
	}
	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "other-pod",
			Namespace: "default",
			Labels:    map[string]string{"app": "other"},
		},
	}

	clientset := fake.NewSimpleClientset(pod1, pod2)
	client := NewK8sClientForTest(clientset)

	pods, err := client.GetPodsInNamespace(context.Background(), "default", "app=test")
	if err != nil {
		t.Fatalf("GetPodsInNamespace failed: %v", err)
	}

	if len(pods) != 1 {
		t.Errorf("expected 1 pod, got %d", len(pods))
	}
	if len(pods) > 0 && pods[0].Name != "app-pod" {
		t.Errorf("expected app-pod, got %s", pods[0].Name)
	}
}

func TestGetPodsInAllNamespaces(t *testing.T) {
	pod1 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-1",
			Namespace: "ns1",
			Labels:    map[string]string{"k8s-app": "kube-dns"},
		},
	}
	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-2",
			Namespace: "ns2",
			Labels:    map[string]string{"k8s-app": "kube-dns"},
		},
	}

	clientset := fake.NewSimpleClientset(pod1, pod2)
	client := NewK8sClientForTest(clientset)

	pods, err := client.GetPodsInAllNamespaces(context.Background(), "k8s-app=kube-dns")
	if err != nil {
		t.Fatalf("GetPodsInAllNamespaces failed: %v", err)
	}

	if len(pods) != 2 {
		t.Errorf("expected 2 pods, got %d", len(pods))
	}
}

func TestGetNode(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
	}

	clientset := fake.NewSimpleClientset(node)
	client := NewK8sClientForTest(clientset)

	result, err := client.GetNode(context.Background(), "test-node")
	if err != nil {
		t.Fatalf("GetNode failed: %v", err)
	}
	if result.Name != "test-node" {
		t.Errorf("expected test-node, got %s", result.Name)
	}

	// Test node not found
	_, err = client.GetNode(context.Background(), "non-existent")
	if err == nil {
		t.Error("expected error for non-existent node")
	}
}

func TestGetNodes(t *testing.T) {
	node1 := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}
	node2 := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-2"}}

	clientset := fake.NewSimpleClientset(node1, node2)
	client := NewK8sClientForTest(clientset)

	nodes, err := client.GetNodes(context.Background())
	if err != nil {
		t.Fatalf("GetNodes failed: %v", err)
	}
	if len(nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(nodes))
	}
}

func TestCreateAndDeletePod(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	client := NewK8sClientForTest(clientset)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
	}

	created, err := client.CreatePod(context.Background(), pod)
	if err != nil {
		t.Fatalf("CreatePod failed: %v", err)
	}
	if created.Name != "test-pod" {
		t.Errorf("expected test-pod, got %s", created.Name)
	}

	err = client.DeletePod(context.Background(), "default", "test-pod")
	if err != nil {
		t.Fatalf("DeletePod failed: %v", err)
	}
}

func TestGetPod(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
	}

	clientset := fake.NewSimpleClientset(pod)
	client := NewK8sClientForTest(clientset)

	result, err := client.GetPod(context.Background(), "default", "test-pod")
	if err != nil {
		t.Fatalf("GetPod failed: %v", err)
	}
	if result.Name != "test-pod" {
		t.Errorf("expected test-pod, got %s", result.Name)
	}
}

func TestClientset(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	client := NewK8sClientForTest(clientset)

	if client.Clientset() != clientset {
		t.Error("Clientset() should return the wrapped clientset")
	}
}
