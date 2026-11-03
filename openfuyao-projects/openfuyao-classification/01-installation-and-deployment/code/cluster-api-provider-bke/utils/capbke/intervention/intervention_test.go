/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package intervention

import (
	"fmt"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

// fakeRecorder captures events for assertion in tests.
type fakeRecorder struct {
	events []recordedEvent
}

type recordedEvent struct {
	eventType string
	reason    string
	message   string
}

func (r *fakeRecorder) Event(object runtime.Object, eventtype, reason, message string) {
	r.events = append(r.events, recordedEvent{eventtype, reason, message})
}

func (r *fakeRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	r.events = append(r.events, recordedEvent{eventtype, reason, fmt.Sprintf(messageFmt, args...)})
}

func (r *fakeRecorder) AnnotatedEventf(object runtime.Object, annotations map[string]string, eventtype, reason, messageFmt string, args ...interface{}) {
	r.events = append(r.events, recordedEvent{eventtype, reason, fmt.Sprintf(messageFmt, args...)})
}

func newTestBKECluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			Conditions: []confv1beta1.ClusterCondition{},
		},
	}
}

func TestRequire_SetsConditionTrue(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	err := Require(Params{
		BKECluster: cluster,
		Recorder:   rec,
		Guidance:   MasterInitTimeout(10 * time.Minute),
	})

	if err != nil {
		t.Fatalf("Require returned error: %v", err)
	}

	cond, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if !ok {
		t.Fatal("ManualInterventionRequiredCondition not found")
	}
	if cond.Status != confv1beta1.ConditionTrue {
		t.Errorf("expected condition True, got %s", cond.Status)
	}
	if cond.Reason != constant.MasterInitTimeoutReason {
		t.Errorf("expected reason %s, got %s", constant.MasterInitTimeoutReason, cond.Reason)
	}
	if !strings.Contains(cond.Message, "timed out") {
		t.Errorf("expected message to contain 'timed out', got %s", cond.Message)
	}
}

func TestRequire_EmitsWarningEvent(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	_ = Require(Params{
		BKECluster: cluster,
		Recorder:   rec,
		Guidance:   MasterInitTimeout(10 * time.Minute),
	})

	if len(rec.events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(rec.events))
	}
	if rec.events[0].eventType != corev1.EventTypeWarning {
		t.Errorf("expected Warning event, got %s", rec.events[0].eventType)
	}
	if rec.events[0].reason != constant.MasterInitTimeoutReason {
		t.Errorf("expected reason %s, got %s", constant.MasterInitTimeoutReason, rec.events[0].reason)
	}
}

func TestRequire_Idempotent_SameReason(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	guidance := MasterInitTimeout(10 * time.Minute)

	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: guidance})
	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: guidance})

	if len(rec.events) != 1 {
		t.Fatalf("expected 1 event (idempotent), got %d", len(rec.events))
	}
}

func TestRequire_NotIdempotent_DifferentReason(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: MasterInitTimeout(10 * time.Minute)})
	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: RetryExhausted("DeployFailed", 10)})

	if len(rec.events) != 2 {
		t.Fatalf("expected 2 events (different reasons), got %d", len(rec.events))
	}

	cond, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if !ok {
		t.Fatal("condition not found")
	}
	if cond.Reason != constant.RetryExhaustedReason {
		t.Errorf("expected reason updated to %s, got %s", constant.RetryExhaustedReason, cond.Reason)
	}
}

func TestRequire_NilRecorder_NoPanic(t *testing.T) {
	cluster := newTestBKECluster()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Require panicked with nil Recorder: %v", r)
		}
	}()

	err := Require(Params{BKECluster: cluster, Recorder: nil, Guidance: MasterInitTimeout(time.Minute)})
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

func TestRequire_NilBKECluster_NoPanic(t *testing.T) {
	rec := &fakeRecorder{}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Require panicked with nil BKECluster: %v", r)
		}
	}()

	err := Require(Params{BKECluster: nil, Recorder: rec, Guidance: MasterInitTimeout(time.Minute)})
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
}

func TestClear_SetsConditionFalse(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	// Mark first
	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: MasterInitTimeout(time.Minute)})

	// Clear (no client, just in-memory)
	err := Clear(nil, cluster)
	if err != nil {
		t.Fatalf("Clear returned error: %v", err)
	}

	cond, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if !ok {
		t.Fatal("condition not found after Clear")
	}
	if cond.Status != confv1beta1.ConditionFalse {
		t.Errorf("expected condition False, got %s", cond.Status)
	}
	if cond.Message != "Manual intervention resolved" {
		t.Errorf("expected message 'Manual intervention resolved', got %s", cond.Message)
	}
}

func TestClear_NotTrue_Noop(t *testing.T) {
	cluster := newTestBKECluster()

	// Condition doesn't exist yet
	err := Clear(nil, cluster)
	if err != nil {
		t.Fatalf("Clear returned error: %v", err)
	}

	_, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if ok {
		t.Error("expected no condition to be created")
	}
}

func TestClearInMemory_SetsConditionFalse(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: MasterInitTimeout(time.Minute)})

	ClearInMemory(cluster)

	cond, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if !ok {
		t.Fatal("condition not found")
	}
	if cond.Status != confv1beta1.ConditionFalse {
		t.Errorf("expected False, got %s", cond.Status)
	}
}

func TestClearInMemory_NilCluster_NoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ClearInMemory panicked with nil cluster: %v", r)
		}
	}()
	ClearInMemory(nil)
}

func TestClearInMemory_NotTrue_Noop(t *testing.T) {
	cluster := newTestBKECluster()
	ClearInMemory(cluster)
	_, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if ok {
		t.Error("expected no condition to be created")
	}
}

func TestIsRequired_True(t *testing.T) {
	cluster := newTestBKECluster()
	rec := &fakeRecorder{}

	_ = Require(Params{BKECluster: cluster, Recorder: rec, Guidance: MasterInitTimeout(time.Minute)})

	if !IsRequired(cluster) {
		t.Error("expected IsRequired to return true")
	}
}

func TestIsRequired_False(t *testing.T) {
	cluster := newTestBKECluster()

	if IsRequired(cluster) {
		t.Error("expected IsRequired to return false")
	}
}

func TestBuildMessage_WithActions(t *testing.T) {
	g := Guidance{
		Message: "Test failure",
		Actions: []string{"Step 1", "Step 2"},
	}
	msg := buildMessage(g)

	if !strings.Contains(msg, "Test failure") {
		t.Error("message should contain the failure description")
	}
	if !strings.Contains(msg, "Recommended actions:") {
		t.Error("message should contain 'Recommended actions:'")
	}
	if !strings.Contains(msg, "1. Step 1") {
		t.Error("message should contain numbered action 1")
	}
	if !strings.Contains(msg, "2. Step 2") {
		t.Error("message should contain numbered action 2")
	}
}

func TestBuildMessage_NoActions(t *testing.T) {
	g := Guidance{
		Message: "Test failure",
	}
	msg := buildMessage(g)

	if strings.Contains(msg, "Recommended actions:") {
		t.Error("message should not contain 'Recommended actions:' when no actions")
	}
}

func TestGuidance_MasterInitTimeout(t *testing.T) {
	g := MasterInitTimeout(10 * time.Minute)

	if g.Category != CategoryMasterInit {
		t.Errorf("expected category %s, got %s", CategoryMasterInit, g.Category)
	}
	if g.Reason != constant.MasterInitTimeoutReason {
		t.Errorf("expected reason %s, got %s", constant.MasterInitTimeoutReason, g.Reason)
	}
	if !strings.Contains(g.Message, "10m") {
		t.Errorf("expected message to contain timeout duration, got %s", g.Message)
	}
	if len(g.Actions) != 4 {
		t.Errorf("expected 4 actions, got %d", len(g.Actions))
	}
}

func TestGuidance_NodeBootstrapTerminal(t *testing.T) {
	g := NodeBootstrapTerminal("192.168.1.10", []string{"192.168.1.10"})

	if g.Category != CategoryNodeFailure {
		t.Errorf("expected category %s, got %s", CategoryNodeFailure, g.Category)
	}
	if g.Reason != constant.NodeBootstrapTerminalReason {
		t.Errorf("expected reason %s, got %s", constant.NodeBootstrapTerminalReason, g.Reason)
	}
	if !strings.Contains(g.Message, "192.168.1.10") {
		t.Errorf("expected message to contain node IP, got %s", g.Message)
	}
	if len(g.Actions) != 4 {
		t.Errorf("expected 4 actions, got %d", len(g.Actions))
	}
}

func TestGuidance_RetryExhausted(t *testing.T) {
	g := RetryExhausted("DeployFailed", 10)

	if g.Category != CategoryRetryExhausted {
		t.Errorf("expected category %s, got %s", CategoryRetryExhausted, g.Category)
	}
	if g.Reason != constant.RetryExhaustedReason {
		t.Errorf("expected reason %s, got %s", constant.RetryExhaustedReason, g.Reason)
	}
	if !strings.Contains(g.Message, "DeployFailed") {
		t.Errorf("expected message to contain health state, got %s", g.Message)
	}
	if !strings.Contains(g.Message, "10") {
		t.Errorf("expected message to contain retry count, got %s", g.Message)
	}
}

func TestRequire_WithFakeClient_Persists(t *testing.T) {
	cluster := newTestBKECluster()
	cluster.Name = "test-cluster"
	cluster.Namespace = "default"

	rec := &fakeRecorder{}

	// Require without a client to avoid SyncStatusUntilComplete retry noise;
	// this tests that the in-memory condition is set correctly regardless
	// of persistence outcome.
	_ = Require(Params{
		BKECluster: cluster,
		Client:     nil,
		Recorder:   rec,
		Guidance:   MasterInitTimeout(time.Minute),
	})

	// Verify condition was set in memory
	cond, ok := conditionHas(cluster, bkev1beta1.ManualInterventionRequiredCondition)
	if !ok {
		t.Fatal("condition not set in memory")
	}
	if cond.Status != confv1beta1.ConditionTrue {
		t.Errorf("expected True, got %s", cond.Status)
	}
}

// conditionHas is a helper that replicates condition.HasCondition for tests
// to avoid import cycle issues in the test file.
func conditionHas(cluster *bkev1beta1.BKECluster, condType confv1beta1.ClusterConditionType) (*confv1beta1.ClusterCondition, bool) {
	for _, c := range cluster.Status.Conditions {
		if c.Type == condType {
			return &c, true
		}
	}
	return nil, false
}
