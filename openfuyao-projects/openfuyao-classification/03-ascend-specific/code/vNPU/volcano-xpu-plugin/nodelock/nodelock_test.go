/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package nodelock

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func newFakeClient() {
	kubeClient = fake.NewSimpleClientset()
}

func newFakeNode(name string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
	}
}

func newFakePod(ns, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
}

func TestLockNodeWithPod_Success(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-a")
	err := LockNodeWithPod("node1", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node1", metav1.GetOptions{})
	value, ok := updated.Annotations["vnpu/mutex.lock"]
	if !ok {
		t.Fatal("expected lock annotation to be set")
	}
	state, err := ParseNodeLock(value)
	if err != nil {
		t.Fatalf("failed to parse lock value: %v", err)
	}
	if state.Namespace != "default" || state.PodName != "pod-a" {
		t.Errorf("expected default/pod-a, got %s/%s", state.Namespace, state.PodName)
	}
}

func TestLockNodeWithPod_AlreadyLocked(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	node.Annotations["vnpu/mutex.lock"] = formatLockValue(newFakePod("default", "pod-a"))
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-b")
	err := LockNodeWithPod("node1", "vnpu/mutex.lock", pod)
	if err == nil {
		t.Fatal("expected error when node already locked")
	}
}

func TestLockNodeWithPod_ExpiredLock(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	expiredTime := time.Now().Add(-6 * time.Minute).Format(time.RFC3339)
	node.Annotations["vnpu/mutex.lock"] = expiredTime + ",default,pod-old"
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-new")
	err := LockNodeWithPod("node1", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success after expired lock, got: %v", err)
	}
}

func TestReleaseNodeLockWithPod_Match(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	node.Annotations["vnpu/mutex.lock"] = formatLockValue(newFakePod("default", "pod-a"))
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-a")
	err := ReleaseNodeLockWithPod("node1", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node1", metav1.GetOptions{})
	if _, ok := updated.Annotations["vnpu/mutex.lock"]; ok {
		t.Fatal("expected lock annotation to be removed")
	}
}

func TestReleaseNodeLockWithPod_Mismatch(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	node.Annotations["vnpu/mutex.lock"] = formatLockValue(newFakePod("default", "pod-a"))
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-b")
	err := ReleaseNodeLockWithPod("node1", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected nil (not an error) when pod identity mismatches, got: %v", err)
	}

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node1", metav1.GetOptions{})
	if _, ok := updated.Annotations["vnpu/mutex.lock"]; !ok {
		t.Fatal("lock should not be released on mismatch")
	}
}

func TestReleaseNodeLockWithPod_NotSet(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-a")
	err := ReleaseNodeLockWithPod("node1", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success when lock not set, got: %v", err)
	}
}

func TestReleaseNodeLockWithPod_OldFormat(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node1")
	node.Annotations["vnpu/mutex.lock"] = time.Now().Format(time.RFC3339)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-a")
	err := ReleaseNodeLockWithPod("node1", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success with old format (no pod identity), got: %v", err)
	}
}

func TestParseNodeLock_OldFormat(t *testing.T) {
	value := time.Now().Format(time.RFC3339)
	state, err := ParseNodeLock(value)
	if err != nil {
		t.Fatalf("failed to parse old format: %v", err)
	}
	if state.Namespace != "" || state.PodName != "" {
		t.Errorf("expected empty pod identity, got %s/%s", state.Namespace, state.PodName)
	}
	if state.Time.IsZero() {
		t.Error("expected non-zero lock time")
	}
}

func TestParseNodeLock_NewFormat(t *testing.T) {
	pod := newFakePod("kube-system", "my-pod")
	value := formatLockValue(pod)
	state, err := ParseNodeLock(value)
	if err != nil {
		t.Fatalf("failed to parse new format: %v", err)
	}
	if state.Namespace != "kube-system" || state.PodName != "my-pod" {
		t.Errorf("expected kube-system/my-pod, got %s/%s", state.Namespace, state.PodName)
	}
	if state.Time.IsZero() {
		t.Error("expected non-zero lock time")
	}
}

func TestParseNodeLock_Invalid(t *testing.T) {
	_, err := ParseNodeLock("not-a-timestamp")
	if err == nil {
		t.Fatal("expected error for invalid format")
	}
}

func TestFormatLockValue_NilPod(t *testing.T) {
	value := formatLockValue(nil)
	state, err := ParseNodeLock(value)
	if err != nil {
		t.Fatalf("failed to parse: %v", err)
	}
	if state.Namespace != "" || state.PodName != "" {
		t.Errorf("expected empty pod identity for nil pod, got %s/%s", state.Namespace, state.PodName)
	}
}

func TestFormatLockValue_WithPod(t *testing.T) {
	pod := newFakePod("ns1", "pod1")
	value := formatLockValue(pod)
	state, err := ParseNodeLock(value)
	if err != nil {
		t.Fatalf("failed to parse: %v", err)
	}
	if state.Namespace != "ns1" || state.PodName != "pod1" {
		t.Errorf("expected ns1/pod1, got %s/%s", state.Namespace, state.PodName)
	}
}

func TestLockNodeWithPod_NodeNotExist(t *testing.T) {
	newFakeClient()
	pod := newFakePod("default", "pod-a")
	err := LockNodeWithPod("non-existent", "vnpu/mutex.lock", pod)
	if err == nil {
		t.Fatal("expected error for non-existent node")
	}
}

func TestReleaseNodeLockWithPod_NodeNotExist(t *testing.T) {
	newFakeClient()
	pod := newFakePod("default", "pod-a")
	err := ReleaseNodeLockWithPod("non-existent", "vnpu/mutex.lock", pod)
	if err == nil {
		t.Fatal("expected error for non-existent node")
	}
}

func TestNodeLockManager_GetLock(t *testing.T) {
	mgr := newNodeLockManager()

	lock1 := mgr.getLock("node-a")
	lock2 := mgr.getLock("node-a")
	if lock1 != lock2 {
		t.Error("expected same mutex for same node name")
	}

	lock3 := mgr.getLock("node-b")
	if lock1 == lock3 {
		t.Error("expected different mutex for different node name")
	}
}

func TestNodeLockManager_DeleteLock(t *testing.T) {
	mgr := newNodeLockManager()

	mgr.getLock("node-a")
	mgr.deleteLock("node-a")

	mgr.mu.Lock()
	_, exists := mgr.locks["node-a"]
	mgr.mu.Unlock()
	if exists {
		t.Error("expected lock to be deleted")
	}
}

func TestNodeLockManager_ConcurrentAccess(t *testing.T) {
	mgr := newNodeLockManager()
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			nodeName := "node-concurrent"
			lock := mgr.getLock(nodeName)
			lock.Lock()
			lock.Unlock()
		}(i)
	}
	wg.Wait()
}

func TestDualLayerLock_ConcurrentAllocation(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-dual")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	var wg sync.WaitGroup
	var serializedCount int32

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			pod := newFakePod("default", "pod-concurrent-"+string(rune('A'+idx)))
			err := LockNodeWithPod("node-dual", "vnpu/mutex.lock", pod)
			if err == nil {
				atomic.AddInt32(&serializedCount, 1)
				ReleaseNodeLockWithPod("node-dual", "vnpu/mutex.lock", pod)
			}
		}(i)
	}
	wg.Wait()

	if serializedCount < 1 {
		t.Errorf("expected at least 1 serialized lock acquisition, got %d", serializedCount)
	}
}

func TestDualLayerLock_AllocateReleaseSequence(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-seq")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod1 := newFakePod("default", "pod-first")
	pod2 := newFakePod("default", "pod-second")

	err := LockNodeWithPod("node-seq", "vnpu/mutex.lock", pod1)
	if err != nil {
		t.Fatalf("first lock should succeed: %v", err)
	}

	err = ReleaseNodeLockWithPod("node-seq", "vnpu/mutex.lock", pod1)
	if err != nil {
		t.Fatalf("first release should succeed: %v", err)
	}

	err = LockNodeWithPod("node-seq", "vnpu/mutex.lock", pod2)
	if err != nil {
		t.Fatalf("second lock after release should succeed: %v", err)
	}

	err = ReleaseNodeLockWithPod("node-seq", "vnpu/mutex.lock", pod2)
	if err != nil {
		t.Fatalf("second release should succeed: %v", err)
	}
}

func TestDualLayerLock_Layer2FailReleasesLayer1(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-l2fail")
	holderPod := newFakePod("other", "pod-holder")
	node.Annotations["vnpu/mutex.lock"] = formatLockValue(holderPod)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
	kubeClient.CoreV1().Pods("other").Create(context.TODO(), holderPod, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-new")
	err := LockNodeWithPod("node-l2fail", "vnpu/mutex.lock", pod)
	if err == nil {
		t.Fatal("expected error when annotation lock already held by active pod")
	}

	nodeMemLock := GetNodeMemoryLock("node-l2fail")
	tryLock := func() bool {
		return nodeMemLock.TryLock()
	}
	if !tryLock() {
		t.Error("expected memory lock to be released when Layer 2 fails")
	} else {
		nodeMemLock.Unlock()
	}
}

func TestCleanupNodeLock(t *testing.T) {
	mgr := nodeLocks
	mgr.getLock("node-to-delete")

	CleanupNodeLock("node-to-delete")

	mgr.mu.Lock()
	_, exists := mgr.locks["node-to-delete"]
	mgr.mu.Unlock()
	if exists {
		t.Error("expected lock to be cleaned up")
	}
}

func TestGetNodeMemoryLock(t *testing.T) {
	lock := GetNodeMemoryLock("node-test")
	if lock == nil {
		t.Fatal("expected non-nil mutex")
	}

	lock2 := GetNodeMemoryLock("node-test")
	if lock != lock2 {
		t.Error("expected same mutex for same node name")
	}
}

func TestReleaseNodeLockAnnotationOnly(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-anno-only")
	node.Annotations["vnpu/mutex.lock"] = formatLockValue(newFakePod("default", "pod-a"))
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-a")
	err := ReleaseNodeLockAnnotationOnly("node-anno-only", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node-anno-only", metav1.GetOptions{})
	if _, ok := updated.Annotations["vnpu/mutex.lock"]; ok {
		t.Fatal("expected lock annotation to be removed")
	}
}

func TestLockNodeWithPod_DanglingLockPodDeleted(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-dangling")
	freshTime := time.Now().Format(time.RFC3339)
	node.Annotations["vnpu/mutex.lock"] = freshTime + ",default,pod-deleted"
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-new")
	err := LockNodeWithPod("node-dangling", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected success after dangling lock released, got: %v", err)
	}

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node-dangling", metav1.GetOptions{})
	value := updated.Annotations["vnpu/mutex.lock"]
	state, parseErr := ParseNodeLock(value)
	if parseErr != nil {
		t.Fatalf("failed to parse lock: %v", parseErr)
	}
	if state.Namespace != "default" || state.PodName != "pod-new" {
		t.Errorf("expected default/pod-new, got %s/%s", state.Namespace, state.PodName)
	}
}

func TestLockNodeWithPod_DanglingLockPodStillExists(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-active")
	freshTime := time.Now().Format(time.RFC3339)
	node.Annotations["vnpu/mutex.lock"] = freshTime + ",default,pod-active"
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	activePod := newFakePod("default", "pod-active")
	kubeClient.CoreV1().Pods("default").Create(context.TODO(), activePod, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-new")
	err := LockNodeWithPod("node-active", "vnpu/mutex.lock", pod)
	if err == nil {
		t.Fatal("expected error when active pod still holds the lock")
	}
}

func TestReleaseNodeMemoryLockOnly(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-memonly")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-memonly")
	err := LockNodeWithPod("node-memonly", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected lock success: %v", err)
	}

	ReleaseNodeMemoryLockOnly("node-memonly")

	nodeMemLock := GetNodeMemoryLock("node-memonly")
	if !nodeMemLock.TryLock() {
		t.Fatal("expected memory lock to be released after ReleaseNodeMemoryLockOnly")
	}
	nodeMemLock.Unlock()

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node-memonly", metav1.GetOptions{})
	if _, ok := updated.Annotations["vnpu/mutex.lock"]; !ok {
		t.Fatal("annotation lock should still be held after ReleaseNodeMemoryLockOnly")
	}
}

func TestAllocateSuccessPath_MemoryLockReleased(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-alloc")
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	pod := newFakePod("default", "pod-alloc")
	err := LockNodeWithPod("node-alloc", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected lock success: %v", err)
	}

	// Simulate Allocate success path: release memory lock only, annotation stays
	ReleaseNodeMemoryLockOnly("node-alloc")

	// Memory lock should be available for next operation
	nodeMemLock := GetNodeMemoryLock("node-alloc")
	if !nodeMemLock.TryLock() {
		t.Fatal("expected memory lock available after successful Allocate path")
	}
	nodeMemLock.Unlock()

	// Annotation lock should still exist (for device-plugin to release)
	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node-alloc", metav1.GetOptions{})
	value, ok := updated.Annotations["vnpu/mutex.lock"]
	if !ok {
		t.Fatal("annotation lock should still exist for device-plugin")
	}
	state, parseErr := ParseNodeLock(value)
	if parseErr != nil {
		t.Fatalf("failed to parse annotation: %v", parseErr)
	}
	if state.Namespace != "default" || state.PodName != "pod-alloc" {
		t.Errorf("expected default/pod-alloc in annotation, got %s/%s", state.Namespace, state.PodName)
	}

	// Device-plugin later releases annotation only
	err = ReleaseNodeLockAnnotationOnly("node-alloc", "vnpu/mutex.lock", pod)
	if err != nil {
		t.Fatalf("expected annotation release success: %v", err)
	}

	updated, _ = kubeClient.CoreV1().Nodes().Get(context.TODO(), "node-alloc", metav1.GetOptions{})
	if _, ok := updated.Annotations["vnpu/mutex.lock"]; ok {
		t.Fatal("annotation should be removed after device-plugin release")
	}
}

func TestDeallocatePath_UsesAnnotationOnly(t *testing.T) {
	newFakeClient()
	node := newFakeNode("node-dealloc")
	holderPod := newFakePod("default", "pod-holder")
	node.Annotations["vnpu/mutex.lock"] = formatLockValue(holderPod)
	kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	// Deallocate does NOT hold memory lock, so it must use AnnotationOnly
	deallocPod := newFakePod("default", "pod-holder")
	err := ReleaseNodeLockAnnotationOnly("node-dealloc", "vnpu/mutex.lock", deallocPod)
	if err != nil {
		t.Fatalf("expected annotation release success: %v", err)
	}

	updated, _ := kubeClient.CoreV1().Nodes().Get(context.TODO(), "node-dealloc", metav1.GetOptions{})
	if _, ok := updated.Annotations["vnpu/mutex.lock"]; ok {
		t.Fatal("annotation should be removed after deallocate")
	}

	// Memory lock should NOT have been touched (never locked in this path)
	nodeMemLock := GetNodeMemoryLock("node-dealloc")
	if !nodeMemLock.TryLock() {
		t.Fatal("memory lock should be available since it was never locked")
	}
	nodeMemLock.Unlock()
}
