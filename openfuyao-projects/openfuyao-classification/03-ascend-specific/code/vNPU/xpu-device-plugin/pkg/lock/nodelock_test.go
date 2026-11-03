/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package lock

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

func newMockClient() {
	client := fake.NewSimpleClientset()
	kubeClient = client
}

func newMockNode(name string) *corev1.Node {
	return &corev1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
	}
}

func TestGetClient(t *testing.T) {
	newMockClient()
	kc := GetClient()
	if kc == nil {
		t.Errorf("kubeclient is nil")
	}
}

func TestSetNodelock(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("valid_test_node")
	nodeMock.ObjectMeta.Annotations["lock_key"] = time.Now().Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	b := make([]byte, 1000)
	for i := range b {
		b[i] = 'z'
	}
	bStr := string(b)
	errTestCases := []struct {
		nodeName  string
		lockName  string
		namespace string
		podName   string
	}{
		{
			nodeName: "",
			lockName: "",
		},
		{
			nodeName: bStr,
			lockName: bStr,
		},
		{
			nodeName: "valid_test_node",
			lockName: "lock_key",
		},
		{
			nodeName: "valid_test_node",
			lockName: "",
		},
		{
			nodeName: "",
			lockName: "new_lock_key",
		},
		{
			nodeName: "not_empty_not_valid",
			lockName: "new_lock_key",
		},
	}

	for i := range errTestCases {
		tc := errTestCases[i]
		err := setNodeLock(tc.nodeName, tc.lockName, tc.namespace, tc.podName)
		if err == nil {
			t.Errorf("no error on invalid input parameters: %d", i)
		}
	}
}

func TestValidSetNodelock(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("valid_test_node1")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})
	err := setNodeLock("valid_test_node1", "new_lock_key", "default", "pod1")
	if err != nil {
		fmt.Println(err)
		t.Errorf("error while setting node lock for valid input")
	}
	err = setNodeLock("valid_test_node1", "new_lock_key1", "default", "pod1")
	if err != nil {
		fmt.Println(err)
		t.Errorf("error while setting node lock for valid input")
	}
}

func TestReleaseNodeLock(t *testing.T) {
	b := make([]byte, 1000)
	for i := range b {
		b[i] = 'z'
	}
	bStr := string(b)
	errTestCases := []struct {
		nodeName string
		lockName string
	}{
		{
			nodeName: "",
			lockName: "",
		},
		{
			nodeName: bStr,
			lockName: bStr,
		},
		{
			nodeName: "",
			lockName: "valid_lock_key",
		},
		{
			nodeName: "not_empty_not_valid",
			lockName: "valid_lock_key",
		},
	}
	newMockClient()
	nodeMock := newMockNode("valid_test_node2")
	nodeMock.ObjectMeta.Annotations["valid_lock_key"] = time.Now().Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})
	for i := range errTestCases {
		tc := errTestCases[i]
		err := ReleaseNodeLock(tc.nodeName, tc.lockName)
		if err == nil {
			t.Errorf("no error on invalid input parameters: %d", i)
		}
	}
	nodeMock = newMockNode("valid_test_node3")
	nodeMock.ObjectMeta.Annotations["valid_lock_key"] = time.Now().Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})
	err := ReleaseNodeLock("valid_test_node3", "valid_lock_key")
	if err != nil {
		fmt.Println(err)
		t.Errorf("error while releasing node lock for valid input")
	}
}

func TestReleaseNonExistentLockName(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("valid_test_node2")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	errTestCases := []struct {
		nodeName string
		lockName string
	}{
		{
			nodeName: "valid_test_node2",
			lockName: "not_empty_not_valid",
		},
		{
			nodeName: "valid_test_node2",
			lockName: "",
		},
	}
	for i := range errTestCases {
		tc := errTestCases[i]
		err := ReleaseNodeLock(tc.nodeName, tc.lockName)
		if err != nil {
			t.Errorf("expected nil error for non-existent lock, got error on case %d: %v", i, err)
		}
	}
}

func TestObtainLockNode(t *testing.T) {
	b := make([]byte, 1000)
	for i := range b {
		b[i] = 'z'
	}
	bStr := string(b)
	errTestCases := []struct {
		nodeName string
		lockName string
	}{
		{
			nodeName: "",
			lockName: "",
		},
		{
			nodeName: bStr,
			lockName: bStr,
		},
		{
			nodeName: "valid_test_node4",
			lockName: "",
		},
		{
			nodeName: "",
			lockName: "valid_lock_key",
		},
		{
			nodeName: "not_empty_not_valid",
			lockName: "valid_lock_key",
		},
	}
	newMockClient()
	nodeMock := newMockNode("valid_test_node4")
	nodeMock.ObjectMeta.Annotations["valid_lock_key"] = time.Now().Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})
	for i := range errTestCases {
		tc := errTestCases[i]
		err := ObtainLockNode(tc.nodeName, tc.lockName)
		if err == nil {
			t.Errorf("no error on invalid input parameters: %d", i)
		}
	}
}

func TestObtainLockNodeOnErrorHandling(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("valid_test_node5")
	nodeMock.ObjectMeta.Annotations["valid_lock_key"] = time.Now().Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})
	err := ObtainLockNode("valid_test_node5", "new_lock_key")
	if err != nil {
		fmt.Println(err)
		t.Errorf("error while obtaining new node lock for valid input")
	}
	err = ObtainLockNode("valid_test_node5", "valid_lock_key")
	if err == nil {
		fmt.Println(err)
		t.Errorf("error while obtaining error it should be still locked")
	}

	nodeMock = newMockNode("valid_test_node6")
	expiredTime := time.Now().Add(-lockExpiredTime - time.Minute).Format(time.RFC3339)
	nodeMock.ObjectMeta.Annotations["valid_lock_key"] = expiredTime
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})
	err = ObtainLockNode("valid_test_node6", "new_lock_key")
	if err != nil {
		fmt.Println(err)
		t.Errorf("error while releasing node lock for valid input")
	}
	err = ObtainLockNode("valid_test_node6", "valid_lock_key")
	if err != nil {
		fmt.Println(err)
		t.Errorf("error while obtaining expired node lock for valid input")
	}
}

func TestReleaseNodeLockWithPod(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("test_node_pod")
	lockValue := formatLockValue("default", "pod-a")
	nodeMock.ObjectMeta.Annotations["vnpu/mutex.lock"] = lockValue
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	err := ReleaseNodeLockWithPod("test_node_pod", "vnpu/mutex.lock", "default", "pod-b")
	if err == nil {
		t.Errorf("should fail when pod identity does not match")
	}

	err = ReleaseNodeLockWithPod("test_node_pod", "vnpu/mutex.lock", "default", "pod-a")
	if err != nil {
		t.Errorf("should succeed when pod identity matches: %v", err)
	}
}

func TestObtainLockNodeWithPod(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("test_node_pod2")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	err := ObtainLockNodeWithPod("test_node_pod2", "vnpu/mutex.lock", "default", "pod-x")
	if err != nil {
		t.Errorf("should succeed on unlocked node: %v", err)
	}

	node, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "test_node_pod2", metav1.GetOptions{})
	state, parseErr := ParseNodeLock(node.Annotations["vnpu/mutex.lock"])
	if parseErr != nil {
		t.Errorf("failed to parse lock value: %v", parseErr)
	}
	if state.Namespace != "default" || state.PodName != "pod-x" {
		t.Errorf("lock pod identity mismatch: got %s/%s", state.Namespace, state.PodName)
	}
}

func TestParseNodeLock(t *testing.T) {
	state, err := ParseNodeLock(time.Now().Format(time.RFC3339))
	if err != nil {
		t.Errorf("failed to parse old format: %v", err)
	}
	if state.Namespace != "" || state.PodName != "" {
		t.Errorf("old format should have empty pod identity")
	}

	state, err = ParseNodeLock(formatLockValue("kube-system", "my-pod"))
	if err != nil {
		t.Errorf("failed to parse new format: %v", err)
	}
	if state.Namespace != "kube-system" || state.PodName != "my-pod" {
		t.Errorf("new format pod identity mismatch: got %s/%s", state.Namespace, state.PodName)
	}
}

func TestGetLockPodIdentity(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("test_identity")
	nodeMock.ObjectMeta.Annotations["vnpu/mutex.lock"] = formatLockValue("ns1", "pod1")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	ns, name, err := GetLockPodIdentity("test_identity", "vnpu/mutex.lock")
	if err != nil {
		t.Errorf("failed to get lock pod identity: %v", err)
	}
	if ns != "ns1" || name != "pod1" {
		t.Errorf("pod identity mismatch: got %s/%s", ns, name)
	}

	ns, name, err = GetLockPodIdentity("not_exist", "vnpu/mutex.lock")
	if err == nil {
		t.Errorf("should fail for non-existent node")
	}
}

func TestSetClient(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	SetClient(fakeClient)
	if GetClient() != fakeClient {
		t.Errorf("SetClient/GetClient round trip failed")
	}
}

func TestNewClient_InClusterError(t *testing.T) {
	patch := gomonkey.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
		return nil, fmt.Errorf("not in cluster")
	})
	defer patch.Reset()

	patchBuild := gomonkey.ApplyFunc(clientcmd.BuildConfigFromFlags, func(masterURL, kubeconfigPath string) (*rest.Config, error) {
		return &rest.Config{Host: "https://localhost:6443"}, nil
	})
	defer patchBuild.Reset()

	patchNew := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(c *rest.Config) (*kubernetes.Clientset, error) {
		return &kubernetes.Clientset{}, nil
	})
	defer patchNew.Reset()

	err := NewClient()
	if err != nil {
		t.Errorf("expected no error from NewClient, got %v", err)
	}
	if GetClient() == nil {
		t.Errorf("expected kubeClient to be set")
	}
}

func TestRetryUpdateNode(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("retry_node")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	updateErr := fmt.Errorf("update failed")
	patch := gomonkey.ApplyMethodFunc(kubeClient.CoreV1().Nodes(), "Update",
		func(ctx context.Context, node *corev1.Node, opts metav1.UpdateOptions) (*corev1.Node, error) {
			return nil, updateErr
		})
	defer patch.Reset()

	err := retryUpdateNode(context.Background(), "retry_node", func(n *corev1.Node) {
		n.Annotations["k"] = "v"
	})
	if err == nil {
		t.Errorf("expected error after exhausting retries")
	}
}

func TestNewClient_BuildConfigError(t *testing.T) {
	patch := gomonkey.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
		return nil, fmt.Errorf("not in cluster")
	})
	defer patch.Reset()

	patchBuild := gomonkey.ApplyFunc(clientcmd.BuildConfigFromFlags, func(masterURL, kubeconfigPath string) (*rest.Config, error) {
		return nil, fmt.Errorf("build config error")
	})
	defer patchBuild.Reset()

	err := NewClient()
	if err == nil {
		t.Errorf("expected error from NewClient when build config fails")
	}
}

func TestRetryUpdateNode_GetError(t *testing.T) {
	newMockClient()
	patch := gomonkey.ApplyMethodFunc(kubeClient.CoreV1().Nodes(), "Get",
		func(ctx context.Context, name string, opts metav1.GetOptions) (*corev1.Node, error) {
			return nil, fmt.Errorf("get node error")
		})
	defer patch.Reset()

	err := retryUpdateNode(context.Background(), "nonexistent", func(n *corev1.Node) {
		// 更新回调为空：本测试只关注 Get 重试失败路径，无需执行实际更新
	})
	if err == nil {
		t.Errorf("expected error after exhausting get retries")
	}
}

func TestRetryUpdateNode_Success(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("retry_success_node")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	err := retryUpdateNode(context.Background(), "retry_success_node", func(n *corev1.Node) {
		n.Annotations["k"] = "v"
	})
	if err != nil {
		t.Errorf("expected success from retryUpdateNode, got %v", err)
	}
}

func TestSetNodeLock_UpdateRetryError(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("retry_set_lock")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	patch := gomonkey.ApplyMethodFunc(kubeClient.CoreV1().Nodes(), "Update",
		func(ctx context.Context, node *corev1.Node, opts metav1.UpdateOptions) (*corev1.Node, error) {
			return nil, fmt.Errorf("update error")
		})
	defer patch.Reset()

	err := setNodeLock("retry_set_lock", "lock_key", "default", "pod")
	if err == nil {
		t.Errorf("expected error after update retries")
	}
}

func TestReleaseNodeLock_UpdateRetryError(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("retry_release_lock")
	nodeMock.ObjectMeta.Annotations["lock_key"] = formatLockValue("default", "pod")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	patch := gomonkey.ApplyMethodFunc(kubeClient.CoreV1().Nodes(), "Update",
		func(ctx context.Context, node *corev1.Node, opts metav1.UpdateOptions) (*corev1.Node, error) {
			return nil, fmt.Errorf("update error")
		})
	defer patch.Reset()

	err := ReleaseNodeLock("retry_release_lock", "lock_key")
	if err == nil {
		t.Errorf("expected error after update retries")
	}
}

func TestReleaseNodeLockWithPod_NotSet(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("no_lock_node")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	err := ReleaseNodeLockWithPod("no_lock_node", "vnpu/mutex.lock", "default", "pod")
	if err != nil {
		t.Errorf("expected nil for unset lock, got %v", err)
	}
}

func TestReleaseNodeLockWithPod_UpdateRetryError(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("retry_release_pod_lock")
	nodeMock.ObjectMeta.Annotations["lock_key"] = formatLockValue("default", "pod")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	patch := gomonkey.ApplyMethodFunc(kubeClient.CoreV1().Nodes(), "Update",
		func(ctx context.Context, node *corev1.Node, opts metav1.UpdateOptions) (*corev1.Node, error) {
			return nil, fmt.Errorf("update error")
		})
	defer patch.Reset()

	err := ReleaseNodeLockWithPod("retry_release_pod_lock", "lock_key", "default", "pod")
	if err == nil {
		t.Errorf("expected error after update retries")
	}
}

func TestObtainLockNodeWithPod_ParseError(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("parse_err_node")
	nodeMock.ObjectMeta.Annotations["lock_key"] = "invalid-lock-value"
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	err := ObtainLockNodeWithPod("parse_err_node", "lock_key", "default", "pod")
	if err == nil {
		t.Errorf("expected error from invalid lock value")
	}
}

func TestObtainLockNodeWithPod_Expired(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("expired_node")
	nodeMock.ObjectMeta.Annotations["lock_key"] = time.Now().Add(-1 * time.Hour).Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	err := ObtainLockNodeWithPod("expired_node", "lock_key", "default", "pod")
	if err != nil {
		t.Errorf("expected success on expired lock, got %v", err)
	}
}

func TestGetLockPodIdentity_GetError(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("identity_get_err")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	patch := gomonkey.ApplyMethodFunc(kubeClient.CoreV1().Nodes(), "Get",
		func(ctx context.Context, name string, opts metav1.GetOptions) (*corev1.Node, error) {
			return nil, fmt.Errorf("get error")
		})
	defer patch.Reset()

	_, _, err := GetLockPodIdentity("identity_get_err", "lock_key")
	if err == nil {
		t.Errorf("expected error from GetLockPodIdentity")
	}
}

func TestGetLockPodIdentity_ParseError(t *testing.T) {
	newMockClient()
	nodeMock := newMockNode("identity_parse_err")
	nodeMock.ObjectMeta.Annotations["lock_key"] = "invalid"
	kubeClient.CoreV1().Nodes().Create(context.TODO(), nodeMock, metav1.CreateOptions{})

	_, _, err := GetLockPodIdentity("identity_parse_err", "lock_key")
	if err == nil {
		t.Errorf("expected error from parse")
	}
}
