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
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

const maxLockRetry = 5

var kubeClient kubernetes.Interface

// nodeLockManager manages per-node sync.Mutex instances for in-process locking (Layer 1).
// It allows concurrent operations on different nodes while serializing operations
// on the same node within a single scheduler process.
type nodeLockManager struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

var nodeLocks = newNodeLockManager()

func newNodeLockManager() nodeLockManager {
	return nodeLockManager{locks: make(map[string]*sync.Mutex)}
}

func (n *nodeLockManager) getLock(name string) *sync.Mutex {
	n.mu.Lock()
	defer n.mu.Unlock()
	if _, ok := n.locks[name]; !ok {
		n.locks[name] = &sync.Mutex{}
	}
	return n.locks[name]
}

func (n *nodeLockManager) deleteLock(name string) {
	n.mu.Lock()
	delete(n.locks, name)
	n.mu.Unlock()
}

// UseClient sets the Kubernetes client for node lock operations.
func UseClient(client kubernetes.Interface) {
	kubeClient = client
}

// CleanupNodeLock removes the in-process per-node mutex entry.
// Should be called when a node is removed from the cluster to prevent unbounded map growth.
func CleanupNodeLock(nodeName string) {
	nodeLocks.deleteLock(nodeName)
}

// GetNodeMemoryLock returns the in-process per-node mutex for the given node name.
// Used by Predicate phase to protect device state evaluation.
func GetNodeMemoryLock(nodeName string) *sync.Mutex {
	return nodeLocks.getLock(nodeName)
}

// LockNodeWithPod acquires a dual-layer node lock with pod identity for precise allocation tracking.
// Layer 1: in-process per-node sync.Mutex for same-process mutual exclusion.
// Layer 2: K8s node annotation with format "RFC3339,namespace,podname" for cross-process coordination.
// If the annotation lock has expired (>5 minutes), it is force-released and re-acquired.
// If Layer 2 fails, Layer 1 is automatically released.
func LockNodeWithPod(nodeName string, lockName string, pod *v1.Pod) error {
	nodeLock := nodeLocks.getLock(nodeName)
	nodeLock.Lock()

	err := lockNodeAnnotation(nodeName, lockName, pod)
	if err != nil {
		nodeLock.Unlock()
		return err
	}
	return nil
}

// ReleaseNodeLockWithPod releases both lock layers.
// It first releases the K8s annotation lock (with pod identity verification),
// then releases the in-process per-node mutex (Layer 1).
func ReleaseNodeLockWithPod(nodeName string, lockName string, pod *v1.Pod) error {
	err := releaseNodeLockAnnotation(nodeName, lockName, pod)

	nodeLock := nodeLocks.getLock(nodeName)
	nodeLock.Unlock()

	return err
}

// ReleaseNodeLockAnnotationOnly releases only the K8s annotation lock, without touching
// the in-process per-node mutex. This is intended for use by the device-plugin, which
// runs in a different process and does not hold the in-process lock.
// It is also used by DeallocateXPUForTask, which does not hold the in-process lock.
func ReleaseNodeLockAnnotationOnly(nodeName string, lockName string, pod *v1.Pod) error {
	return releaseNodeLockAnnotation(nodeName, lockName, pod)
}

// ReleaseNodeMemoryLockOnly releases only the in-process per-node mutex (Layer 1),
// without touching the K8s annotation lock. This is used after a successful Allocate,
// where the annotation lock is intentionally left held for the device-plugin to release later.
func ReleaseNodeMemoryLockOnly(nodeName string) {
	nodeLock := nodeLocks.getLock(nodeName)
	nodeLock.Unlock()
}

// lockNodeAnnotation acquires the K8s node annotation lock (Layer 2).
// Called internally after Layer 1 (in-process mutex) is held.
func lockNodeAnnotation(nodeName string, lockName string, pod *v1.Pod) error {
	ctx := context.Background()
	node, getErr := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if getErr != nil {
		return getErr
	}
	existing, hasLock := node.ObjectMeta.Annotations[lockName]
	if !hasLock {
		return setNodeLock(nodeName, lockName, pod)
	}
	st, parseErr := ParseNodeLock(existing)
	if parseErr != nil {
		return parseErr
	}
	shouldReset, evalErr := evaluateLockState(ctx, st, pod, nodeName, existing)
	if evalErr != nil {
		return evalErr
	}
	if !shouldReset {
		return fmt.Errorf("node %s has been locked within 5 minutes", nodeName)
	}
	if releaseErr := releaseNodeLock(nodeName, lockName); releaseErr != nil {
		klog.ErrorS(releaseErr, "Failed to release expired/dangling node lock", "node", nodeName)
		return releaseErr
	}
	return setNodeLock(nodeName, lockName, pod)
}

func evaluateLockState(ctx context.Context, st *LockState, pod *v1.Pod, nodeName, rawValue string) (bool, error) {
	if time.Since(st.Time) > time.Minute*5 {
		klog.V(3).InfoS("Node lock expired", "node", nodeName, "lockTime", st.Time)
		return true, nil
	}
	if st.Namespace == "" || st.PodName == "" {
		return false, nil
	}
	if pod != nil && st.Namespace == pod.Namespace && st.PodName == pod.Name {
		return false, nil
	}
	_, podErr := kubeClient.CoreV1().Pods(st.Namespace).Get(ctx, st.PodName, metav1.GetOptions{})
	if podErr == nil {
		return false, nil
	}
	if !apierrors.IsNotFound(podErr) {
		klog.ErrorS(podErr, "Failed to get pod of NodeLock", "podName", st.PodName, "namespace", st.Namespace)
		return false, podErr
	}
	klog.InfoS("Previous pod of NodeLock not found, releasing lock",
		"podName", st.PodName, "namespace", st.Namespace, "nodeLock", rawValue)
	return true, nil
}

// releaseNodeLockAnnotation releases the K8s node annotation lock with pod identity verification.
// Internal function, does not touch the in-process mutex (Layer 1).
// Only releases if the pod identity in the annotation matches the requesting pod.
// If the annotation has no pod identity (old format), the lock is released unconditionally.
func releaseNodeLockAnnotation(nodeName string, lockName string, pod *v1.Pod) error {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	value, ok := node.ObjectMeta.Annotations[lockName]
	if !ok {
		klog.V(3).InfoS("Node lock not set", "node", nodeName)
		return nil
	}
	state, parseErr := ParseNodeLock(value)
	if parseErr == nil && state.Namespace != "" && state.PodName != "" {
		if pod == nil {
			klog.InfoS("NodeLock is not set by nil pod", lockName, value)
			return nil
		}
		if pod.Namespace != state.Namespace || pod.Name != state.PodName {
			klog.InfoS("NodeLock is not set by this pod", lockName, value,
				"podName", pod.Name, "podNamespace", pod.Namespace)
			return nil
		}
	}
	return releaseNodeLock(nodeName, lockName)
}

// LockState represents the current state of a node lock.
type LockState struct {
	Time      time.Time
	Namespace string
	PodName   string
}

// IsHeld returns true if the lock is held by a pod.
func (s *LockState) IsHeld() bool {
	return s != nil && s.PodName != ""
}

// GetLockState returns the current lock state on the node.
// Returns nil state if the lock is not held.
func GetLockState(nodeName string, lockName string) (*LockState, error) {
	ctx := context.Background()
	node, err := kubeClient.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	value, ok := node.ObjectMeta.Annotations[lockName]
	if !ok {
		return nil, nil
	}
	state, err := ParseNodeLock(value)
	if err != nil {
		return nil, err
	}
	return state, nil
}

// ParseNodeLock parses a lock annotation value into a LockState.
// Supports both old format ("RFC3339") and new format ("RFC3339,namespace,podname").
func ParseNodeLock(value string) (*LockState, error) {
	parts := strings.SplitN(value, ",", 3)
	t, err := time.Parse(time.RFC3339, parts[0])
	if err != nil {
		return nil, err
	}
	state := &LockState{Time: t}
	if len(parts) >= 3 {
		state.Namespace = parts[1]
		state.PodName = parts[2]
	}
	return state, nil
}

func setNodeLock(nodeName string, lockName string, pod *v1.Pod) error {
	val := formatLockValue(pod)
	opLabel := "setNodeLock"
	if pod != nil {
		opLabel = fmt.Sprintf("setNodeLock(pod=%s/%s)", pod.Namespace, pod.Name)
	}
	if err := applyNodeAnnotationWithRetry(nodeName, lockName, val, false, opLabel); err != nil {
		return err
	}
	klog.InfoS("Node lock set", "node", nodeName, "pod", val)
	return nil
}

func releaseNodeLock(nodeName string, lockName string) error {
	if err := applyNodeAnnotationWithRetry(nodeName, lockName, "", true, "releaseNodeLock"); err != nil {
		return err
	}
	klog.InfoS("Node lock released", "node", nodeName)
	return nil
}

func applyNodeAnnotationWithRetry(nodeName, lockName, value string, remove bool, opLabel string) error {
	ctx := context.Background()
	nodes := kubeClient.CoreV1().Nodes()

	mutate := func(annotations map[string]string) {
		if remove {
			delete(annotations, lockName)
			return
		}
		annotations[lockName] = value
	}

	for attempt := 0; attempt <= maxLockRetry; attempt++ {
		current, getErr := nodes.Get(ctx, nodeName, metav1.GetOptions{})
		if getErr != nil {
			klog.ErrorS(getErr, opLabel+": get node failed", "node", nodeName, "attempt", attempt)
			if attempt == maxLockRetry {
				return fmt.Errorf("%s on node %s: %w", opLabel, nodeName, getErr)
			}
			time.Sleep(100 * time.Millisecond)
			continue
		}

		if !remove {
			if _, locked := current.ObjectMeta.Annotations[lockName]; locked && attempt == 0 {
				klog.V(3).Infof("node %s is locked", nodeName)
				return fmt.Errorf("node %s is locked", nodeName)
			}
		} else if attempt == 0 {
			if _, set := current.ObjectMeta.Annotations[lockName]; !set {
				klog.V(3).InfoS("Node lock not set", "node", nodeName)
				return nil
			}
		} else {
			// retry attempt: skip initial-state checks, proceed to mutate+update
		}

		patched := current.DeepCopy()
		mutate(patched.ObjectMeta.Annotations)
		if _, updateErr := nodes.Update(ctx, patched, metav1.UpdateOptions{}); updateErr != nil {
			klog.ErrorS(updateErr, opLabel+": update failed, will retry", "node", nodeName, "attempt", attempt)
			if attempt == maxLockRetry {
				return fmt.Errorf("%s exceeds retry count %d", opLabel, maxLockRetry)
			}
			time.Sleep(100 * time.Millisecond)
			continue
		}
		return nil
	}
	return fmt.Errorf("%s exceeds retry count %d", opLabel, maxLockRetry)
}

func formatLockValue(pod *v1.Pod) string {
	if pod == nil {
		return time.Now().Format(time.RFC3339)
	}
	return fmt.Sprintf("%s,%s,%s", time.Now().Format(time.RFC3339), pod.Namespace, pod.Name)
}
