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

package k8sutil

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestGetNodeList_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
	})
	list, err := GetNodeList(clientset)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(list.Items) != 1 {
		t.Errorf("Expected 1 node, got %d", len(list.Items))
	}
}

func TestGetNodeListByName_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}},
		&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node2"}},
	)
	list, err := GetNodeListByName(clientset, "node1")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(list.Items) != 1 {
		t.Errorf("Expected 1 node, got %d", len(list.Items))
	}
}

func TestGetScheduler_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "volcano-scheduler-abc", Namespace: "default"},
		Spec:       v1.PodSpec{SchedulerName: "volcano"},
	})
	schedulers, err := GetScheduler(clientset)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(schedulers) != 1 {
		t.Errorf("Expected 1 scheduler, got %d", len(schedulers))
	}
}

func TestGetPodNameByPrefix_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "volcano-scheduler-abc", Namespace: "default"},
	})
	name, err := GetPodNameByPrefix(clientset, "default", "volcano-scheduler")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if name != "volcano-scheduler-abc" {
		t.Errorf("Expected volcano-scheduler-abc, got %s", name)
	}
}

func TestGetPodNameByPrefix_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := GetPodNameByPrefix(clientset, "default", "nonexistent")
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestDeleteVolcanoPod_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "volcano-scheduler-abc", Namespace: "volcano-system"},
	})
	err := DeleteVolcanoPod(clientset, "volcano-system")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
}

func TestPodUuidToName_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			UID:       "test-uid",
		},
	})
	name, err := PodUuidToName(clientset, "test-uid")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if name != "test-pod" {
		t.Errorf("Expected test-pod, got %s", name)
	}
}

func TestPodUuidToName_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := PodUuidToName(clientset, "nonexistent-uid")
	if err != nil {
		t.Errorf("Expected no error for not found, got %v", err)
	}
}

func TestExtractSchedulerName_NoMatch(t *testing.T) {
	result := extractSchedulerName("some-pod-name")
	if result != "" {
		t.Errorf("Expected empty string, got %s", result)
	}
}

func TestGetCR(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_ = clientset
}

func TestCreateSecret_Error(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "existing", Namespace: "default"},
	})
	_, err := CreateSecret(clientset, &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "existing", Namespace: "default"},
	})
	_ = err
}

func TestGetConfigMap_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := GetConfigMap(clientset, "nonexistent", "default")
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestGetSecret_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := GetSecret(clientset, "nonexistent", "default")
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestDeleteSecret_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	err := DeleteSecret(clientset, "nonexistent", "default")
	_ = err
}

func TestDeleteConfigMap_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	err := DeleteConfigMap(clientset, "nonexistent", "default")
	_ = err
}

func TestUpdateConfigMap_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := UpdateConfigMap(clientset, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "nonexistent", Namespace: "default"},
	})
	_ = err
}

func TestUpdateSecret_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := UpdateSecret(clientset, &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "nonexistent", Namespace: "default"},
	})
	_ = err
}

func TestGetNodeListByName_NoMatch(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
	})
	list, err := GetNodeListByName(clientset, "nonexistent")
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(list.Items) != 0 {
		t.Errorf("Expected 0 nodes, got %d", len(list.Items))
	}
}

func TestGetScheduler_NoSchedulers(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "regular-pod", Namespace: "default"},
	})
	schedulers, err := GetScheduler(clientset)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if len(schedulers) != 0 {
		t.Errorf("Expected 0 schedulers, got %d", len(schedulers))
	}
}

func TestDeleteVolcanoPod_NoPod(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	err := DeleteVolcanoPod(clientset, "volcano-system")
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func ensureNoErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestCreateConfigMap_Success(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	_, err := CreateConfigMap(clientset, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "default"},
	})
	ensureNoErr(t, err)
}

func TestGetScheduler_EmptySchedulerName(t *testing.T) {
	clientset := fake.NewSimpleClientset(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "volcano-scheduler-test", Namespace: "default"},
	})
	schedulers, _ := GetScheduler(clientset)
	if len(schedulers) != 0 {
		t.Errorf("Expected 0 schedulers for empty scheduler name, got %d", len(schedulers))
	}
}
