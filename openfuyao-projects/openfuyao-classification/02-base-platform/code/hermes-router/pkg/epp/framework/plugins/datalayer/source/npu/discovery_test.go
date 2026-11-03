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

package npu

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolveTargetURLReturnsSameNodeExporterURL(t *testing.T) {
	resolver, err := newKubeTargetResolver(newFakePodClient(t,
		newWorkloadPod("kserve", "model-a", "n1"),
		newExporterPod("npu-exporter", "exporter-a", "n0", "10.0.0.1", corev1.PodRunning),
		newExporterPod("npu-exporter", "exporter-b", "n1", "10.0.0.2", corev1.PodRunning),
	), validParameters())
	if err != nil {
		t.Fatalf("newKubeTargetResolver: %v", err)
	}

	targetURL, err := resolver.resolveTargetURL(context.Background(), "kserve", "model-a")
	if err != nil {
		t.Fatalf("resolveTargetURL: %v", err)
	}
	if want := "http://10.0.0.2:8082/metrics"; targetURL != want {
		t.Fatalf("targetURL = %q, want %q", targetURL, want)
	}
}

func TestResolveTargetURLReturnsErrorWhenWorkloadNodeMissing(t *testing.T) {
	resolver, err := newKubeTargetResolver(newFakePodClient(t,
		newWorkloadPod("kserve", "model-a", ""),
	), validParameters())
	if err != nil {
		t.Fatalf("newKubeTargetResolver: %v", err)
	}

	if _, err := resolver.resolveTargetURL(context.Background(), "kserve", "model-a"); err == nil {
		t.Fatal("expected resolveTargetURL error")
	}
}

func TestResolveTargetURLReturnsErrorWhenExporterPodIPMissing(t *testing.T) {
	resolver, err := newKubeTargetResolver(newFakePodClient(t,
		newWorkloadPod("kserve", "model-a", "n1"),
		newExporterPod("npu-exporter", "exporter-b", "n1", "", corev1.PodRunning),
	), validParameters())
	if err != nil {
		t.Fatalf("newKubeTargetResolver: %v", err)
	}

	if _, err := resolver.resolveTargetURL(context.Background(), "kserve", "model-a"); err == nil {
		t.Fatal("expected resolveTargetURL error")
	}
}

func newFakePodClient(t *testing.T, pods ...ctrlclient.Object) ctrlclient.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(pods...).Build()
}

func newWorkloadPod(namespace, name, nodeName string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Spec:       corev1.PodSpec{NodeName: nodeName},
	}
}

func newExporterPod(namespace, name, nodeName, podIP string, phase corev1.PodPhase) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			Labels:    map[string]string{"app": "npu-exporter"},
		},
		Spec: corev1.PodSpec{NodeName: nodeName},
		Status: corev1.PodStatus{
			Phase: phase,
			PodIP: podIP,
		},
	}
}
