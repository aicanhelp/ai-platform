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

package discovery

import (
	"errors"
	"testing"

	"github.com/go-logr/logr"
)

const (
	testPodIP1      = "10.0.0.1"
	testPodIP2      = "10.0.0.2"
	testPodIP3      = "10.0.0.3"
	testUID1        = "uid-1"
	testUID2        = "uid-2"
	testZMQPort1    = int32(5557)
	testZMQPort2    = int32(5558)
	testHTTPPort    = int32(8000)
	testRPCPort     = int32(9000)
	testPodName1    = "pod-1"
	testPodName2    = "pod-2"
	testNamespace   = "test-ns"
	testErrorString = "test error"
)

// TestNewReconciler verifies reconciler creation with valid subscriber.
func TestNewReconciler(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	if r == nil {
		t.Fatal("expected non-nil reconciler")
	}
	reconciler, ok := r.(*reconciler)
	if !ok {
		t.Fatal("expected *reconciler type")
	}
	if reconciler.subscriber == nil {
		t.Error("expected non-nil subscriber")
	}
}

// TestNewReconciler_NilSubscriber verifies reconciler accepts nil subscriber.
func TestNewReconciler_NilSubscriber(t *testing.T) {
	r := NewReconciler(logr.Discard(), nil)

	if r == nil {
		t.Fatal("expected non-nil reconciler")
	}
	reconciler, ok := r.(*reconciler)
	if !ok {
		t.Fatal("expected *reconciler type")
	}
	if reconciler.subscriber != nil {
		t.Error("expected nil subscriber")
	}
}

// TestReconcile_NextNil verifies Reconcile returns early when next snapshot is nil.
func TestReconcile_NextNil(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	err := r.Reconcile(nil, nil)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {PodIP: testPodIP1},
		},
	}
	err = r.Reconcile(prev, nil)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

// TestReconcile_PrevNil verifies handling initial snapshot with no previous state.
func TestReconcile_PrevNil(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP:   testPodIP1,
				ZMQPort: testZMQPort1,
				Identity: PodIdentity{
					Name: testPodName1,
					UID:  testUID1,
				},
				MooncakeClient: MooncakeClientEndpoint{
					Resolved: true,
				},
			},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{
			PodIP: testPodIP2,
		},
	}

	err := r.Reconcile(nil, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 1 || sub.added[0] != testPodIP1 {
		t.Errorf("added = %v, want [%s]", sub.added, testPodIP1)
	}
	if len(sub.removed) != 0 {
		t.Errorf("removed = %v, want []", sub.removed)
	}
	if sub.masterCalls != 1 {
		t.Errorf("masterCalls = %d, want 1", sub.masterCalls)
	}
}

// TestReconcile_AddPod verifies OnVLLMAdded callback when new pod appears.
func TestReconcile_AddPod(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP:   testPodIP1,
				ZMQPort: testZMQPort1,
				Identity: PodIdentity{
					Name: testPodName1,
					UID:  testUID1,
				},
				MooncakeClient: MooncakeClientEndpoint{
					Resolved: true,
				},
			},
		},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 1 || sub.added[0] != testPodIP1 {
		t.Errorf("added = %v, want [%s]", sub.added, testPodIP1)
	}
	if len(sub.removed) != 0 {
		t.Errorf("removed = %v, want []", sub.removed)
	}
}

// TestReconcile_RemovePod verifies OnVLLMRemoved callback when pod disappears.
func TestReconcile_RemovePod(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
			},
		},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 0 {
		t.Errorf("added = %v, want []", sub.added)
	}
	if len(sub.removed) != 1 || sub.removed[0] != testPodIP1 {
		t.Errorf("removed = %v, want [%s]", sub.removed, testPodIP1)
	}
}

// TestReconcile_RebuildPod_UIDChanged triggers remove+add when pod UID changes.
func TestReconcile_RebuildPod_UIDChanged(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
				ZMQPort: testZMQPort1,
			},
		},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID2,
				},
				ZMQPort: testZMQPort1,
			},
		},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 1 || sub.added[0] != testPodIP1 {
		t.Errorf("added = %v, want [%s]", sub.added, testPodIP1)
	}
	if len(sub.removed) != 1 || sub.removed[0] != testPodIP1 {
		t.Errorf("removed = %v, want [%s]", sub.removed, testPodIP1)
	}
}

// TestReconcile_RebuildPod_ZMQPortChanged triggers remove+add when ZMQ port changes.
func TestReconcile_RebuildPod_ZMQPortChanged(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
				ZMQPort: testZMQPort1,
			},
		},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
				ZMQPort: testZMQPort2,
			},
		},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 1 || sub.added[0] != testPodIP1 {
		t.Errorf("added = %v, want [%s]", sub.added, testPodIP1)
	}
	if len(sub.removed) != 1 || sub.removed[0] != testPodIP1 {
		t.Errorf("removed = %v, want [%s]", sub.removed, testPodIP1)
	}
}

// TestReconcile_NoChange verifies no callbacks when snapshots are identical.
func TestReconcile_NoChange(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
				ZMQPort: testZMQPort1,
			},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{
			PodIP:    testPodIP2,
			HTTPPort: testHTTPPort,
			RPCPort:  testRPCPort,
			Identity: PodIdentity{
				UID: testUID1,
			},
		},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
				ZMQPort: testZMQPort1,
			},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{
			PodIP:    testPodIP2,
			HTTPPort: testHTTPPort,
			RPCPort:  testRPCPort,
			Identity: PodIdentity{
				UID: testUID1,
			},
		},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 0 {
		t.Errorf("added = %v, want []", sub.added)
	}
	if len(sub.removed) != 0 {
		t.Errorf("removed = %v, want []", sub.removed)
	}
	if sub.masterCalls != 0 {
		t.Errorf("masterCalls = %d, want 0", sub.masterCalls)
	}
}

// TestReconcile_MooncakeMasterChanged verifies OnMooncakeMasterChanged when master IP changes.
func TestReconcile_MooncakeMasterChanged(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods:          map[string]*VLLMPodEndpoint{},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP1},
	}
	next := &Snapshot{
		VLLMPods:          map[string]*VLLMPodEndpoint{},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP2},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if sub.masterCalls != 1 {
		t.Errorf("masterCalls = %d, want 1", sub.masterCalls)
	}
}

// TestReconcile_MooncakeMasterAdded verifies OnMooncakeMasterChanged when master appears.
func TestReconcile_MooncakeMasterAdded(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods:          map[string]*VLLMPodEndpoint{},
		MooncakeMasterPod: nil,
	}
	next := &Snapshot{
		VLLMPods:          map[string]*VLLMPodEndpoint{},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP1},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if sub.masterCalls != 1 {
		t.Errorf("masterCalls = %d, want 1", sub.masterCalls)
	}
}

// TestReconcile_MooncakeMasterRemoved verifies OnMooncakeMasterChanged when master disappears.
func TestReconcile_MooncakeMasterRemoved(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods:          map[string]*VLLMPodEndpoint{},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP1},
	}
	next := &Snapshot{
		VLLMPods:          map[string]*VLLMPodEndpoint{},
		MooncakeMasterPod: nil,
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if sub.masterCalls != 1 {
		t.Errorf("masterCalls = %d, want 1", sub.masterCalls)
	}
}

// TestReconcile_NilSubscriber verifies Reconcile works correctly without subscriber.
func TestReconcile_NilSubscriber(t *testing.T) {
	r := NewReconciler(logr.Discard(), nil)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {PodIP: testPodIP1},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP2},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP2: {PodIP: testPodIP2},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP3},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

// TestReconcile_SubscriberErrors verifies errors from subscriber callbacks are logged but not returned.
func TestReconcile_SubscriberErrors(t *testing.T) {
	testErr := errors.New(testErrorString)
	sub := &errorSubscriber{err: testErr}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {PodIP: testPodIP1},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP1},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP2: {
				PodIP: testPodIP2,
				Identity: PodIdentity{
					UID: testUID2,
				},
			},
		},
		MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: testPodIP2},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error (errors are logged, not returned), got %v", err)
	}
}

// TestReconcile_MultiplePods verifies add/remove operations with multiple pod changes.
func TestReconcile_MultiplePods(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP1: {
				PodIP: testPodIP1,
				Identity: PodIdentity{
					UID: testUID1,
				},
				ZMQPort: testZMQPort1,
			},
			testPodIP2: {
				PodIP: testPodIP2,
				Identity: PodIdentity{
					UID: testUID2,
				},
				ZMQPort: testZMQPort1,
			},
		},
	}
	next := &Snapshot{
		VLLMPods: map[string]*VLLMPodEndpoint{
			testPodIP2: {
				PodIP: testPodIP2,
				Identity: PodIdentity{
					UID: testUID2,
				},
				ZMQPort: testZMQPort1,
			},
			testPodIP3: {
				PodIP: testPodIP3,
				Identity: PodIdentity{
					UID: "uid-3",
				},
				ZMQPort: testZMQPort1,
			},
		},
	}

	err := r.Reconcile(prev, next)
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}

	if len(sub.added) != 1 || sub.added[0] != testPodIP3 {
		t.Errorf("added = %v, want [%s]", sub.added, testPodIP3)
	}
	if len(sub.removed) != 1 || sub.removed[0] != testPodIP1 {
		t.Errorf("removed = %v, want [%s]", sub.removed, testPodIP1)
	}
}

// TestMooncakeMasterChanged_BothNil returns false when both endpoints are nil.
func TestMooncakeMasterChanged_BothNil(t *testing.T) {
	changed := mooncakeMasterChanged(nil, nil)
	if changed {
		t.Error("expected false when both are nil")
	}
}

// TestMooncakeMasterChanged_ANil returns true when previous endpoint is nil.
func TestMooncakeMasterChanged_ANil(t *testing.T) {
	b := &MooncakeMasterEndpoint{PodIP: testPodIP1}
	changed := mooncakeMasterChanged(nil, b)
	if !changed {
		t.Error("expected true when a is nil and b is not")
	}
}

// TestMooncakeMasterChanged_BNil returns true when next endpoint is nil.
func TestMooncakeMasterChanged_BNil(t *testing.T) {
	a := &MooncakeMasterEndpoint{PodIP: testPodIP1}
	changed := mooncakeMasterChanged(a, nil)
	if !changed {
		t.Error("expected true when b is nil and a is not")
	}
}

// TestMooncakeMasterChanged_PodIPChanged detects PodIP field change.
func TestMooncakeMasterChanged_PodIPChanged(t *testing.T) {
	a := &MooncakeMasterEndpoint{PodIP: testPodIP1}
	b := &MooncakeMasterEndpoint{PodIP: testPodIP2}
	changed := mooncakeMasterChanged(a, b)
	if !changed {
		t.Error("expected true when PodIP changed")
	}
}

// TestMooncakeMasterChanged_HTTPPortChanged detects HTTPPort field change.
func TestMooncakeMasterChanged_HTTPPortChanged(t *testing.T) {
	a := &MooncakeMasterEndpoint{PodIP: testPodIP1, HTTPPort: testHTTPPort}
	b := &MooncakeMasterEndpoint{PodIP: testPodIP1, HTTPPort: testHTTPPort + 1}
	changed := mooncakeMasterChanged(a, b)
	if !changed {
		t.Error("expected true when HTTPPort changed")
	}
}

// TestMooncakeMasterChanged_RPCPortChanged detects RPCPort field change.
func TestMooncakeMasterChanged_RPCPortChanged(t *testing.T) {
	a := &MooncakeMasterEndpoint{PodIP: testPodIP1, RPCPort: testRPCPort}
	b := &MooncakeMasterEndpoint{PodIP: testPodIP1, RPCPort: testRPCPort + 1}
	changed := mooncakeMasterChanged(a, b)
	if !changed {
		t.Error("expected true when RPCPort changed")
	}
}

// TestMooncakeMasterChanged_UIDChanged detects UID field change.
func TestMooncakeMasterChanged_UIDChanged(t *testing.T) {
	a := &MooncakeMasterEndpoint{
		PodIP: testPodIP1,
		Identity: PodIdentity{
			UID: testUID1,
		},
	}
	b := &MooncakeMasterEndpoint{
		PodIP: testPodIP1,
		Identity: PodIdentity{
			UID: testUID2,
		},
	}
	changed := mooncakeMasterChanged(a, b)
	if !changed {
		t.Error("expected true when UID changed")
	}
}

// TestMooncakeMasterChanged_NoChange returns false when all fields match.
func TestMooncakeMasterChanged_NoChange(t *testing.T) {
	a := &MooncakeMasterEndpoint{
		PodIP:    testPodIP1,
		HTTPPort: testHTTPPort,
		RPCPort:  testRPCPort,
		Identity: PodIdentity{
			UID: testUID1,
		},
	}
	b := &MooncakeMasterEndpoint{
		PodIP:    testPodIP1,
		HTTPPort: testHTTPPort,
		RPCPort:  testRPCPort,
		Identity: PodIdentity{
			UID: testUID1,
		},
	}
	changed := mooncakeMasterChanged(a, b)
	if changed {
		t.Error("expected false when nothing changed")
	}
}

// TestMasterIP_Nil returns empty string for nil endpoint.
func TestMasterIP_Nil(t *testing.T) {
	ip := masterIP(nil)
	if ip != "" {
		t.Errorf("expected empty string for nil, got %q", ip)
	}
}

// TestMasterIP_NonNil returns PodIP for valid endpoint.
func TestMasterIP_NonNil(t *testing.T) {
	m := &MooncakeMasterEndpoint{PodIP: testPodIP1}
	ip := masterIP(m)
	if ip != testPodIP1 {
		t.Errorf("expected %s, got %s", testPodIP1, ip)
	}
}

type errorSubscriber struct {
	err error
}

func (e *errorSubscriber) OnVLLMAdded(ep *VLLMPodEndpoint) error {
	return e.err
}

func (e *errorSubscriber) OnVLLMRemoved(podIP string) error {
	return e.err
}

func (e *errorSubscriber) OnMooncakeMasterChanged(endpoint *MooncakeMasterEndpoint) error {
	return e.err
}
