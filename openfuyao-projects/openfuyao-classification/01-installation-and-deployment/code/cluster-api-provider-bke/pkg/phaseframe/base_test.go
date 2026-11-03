/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseframe

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestNewBasePhase(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	phaseName := confv1beta1.BKEClusterPhase("TestPhase")
	bp := NewBasePhase(pc, phaseName)
	if bp.PhaseName != phaseName {
		t.Error("PhaseName not set correctly")
	}
	if bp.Ctx != pc {
		t.Error("Context not set correctly")
	}
}

func TestBasePhase_Name(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	phaseName := confv1beta1.BKEClusterPhase("TestPhase")
	bp := NewBasePhase(pc, phaseName)
	if bp.Name() != phaseName {
		t.Error("Name() returned incorrect value")
	}
}

func TestBasePhase_CName(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	if bp.CName() == "" {
		t.Error("CName() returned empty string")
	}

	bp.SetCName("测试阶段")
	if bp.CName() != "测试阶段" {
		t.Error("SetCName() not working")
	}
}

func TestBasePhase_SetStatus(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStatus(bkev1beta1.PhaseRunning)
	if bp.GetStatus() != bkev1beta1.PhaseRunning {
		t.Error("Status not set correctly")
	}
}

func TestBasePhase_SetStartTime(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	now := metav1.Now()
	bp.SetStartTime(now)
	if bp.GetStartTime() != now {
		t.Error("StartTime not set correctly")
	}
}

func TestBasePhase_GetPhaseContext(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	if bp.GetPhaseContext() != pc {
		t.Error("GetPhaseContext() returned incorrect value")
	}
}

func TestBasePhase_SetPhaseContext(t *testing.T) {
	pc1 := NewReconcilePhaseCtx(context.Background())
	pc1.SetLogger(&bkev1beta1.BKELogger{})
	pc1.SetBKECluster(&bkev1beta1.BKECluster{})

	pc2 := NewReconcilePhaseCtx(context.Background())
	pc2.SetLogger(&bkev1beta1.BKELogger{})
	pc2.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc1, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetPhaseContext(pc2)
	if bp.GetPhaseContext() != pc2 {
		t.Error("SetPhaseContext() not working")
	}
}

func TestBasePhase_RegisterPreHooks(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	hook := func(p Phase) error { return nil }
	bp.RegisterPreHooks(hook)
	if len(bp.CustomPreHookFuncs) != 1 {
		t.Error("PreHook not registered")
	}
}

func TestBasePhase_RegisterPostHooks(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	hook := func(p Phase, err error) error { return nil }
	bp.RegisterPostHooks(hook)
	if len(bp.CustomPostHookFuncs) != 1 {
		t.Error("PostHook not registered")
	}
}

func TestBasePhase_CheckCommonNeedExecute(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))

	tests := []struct {
		name     string
		cluster  *bkev1beta1.BKECluster
		expected bool
	}{
		{"normal", &bkev1beta1.BKECluster{}, true},
		{"paused", &bkev1beta1.BKECluster{Spec: confv1beta1.BKEClusterSpec{Pause: true}}, false},
		{"dryrun", &bkev1beta1.BKECluster{Spec: confv1beta1.BKEClusterSpec{DryRun: true}}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := bp.checkCommonNeedExecute(tt.cluster)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestBasePhase_NormalNeedExecute(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}

	result := bp.NormalNeedExecute(old, new)
	if !result {
		t.Error("NormalNeedExecute should return true for normal cluster")
	}
}

func TestBasePhase_HandleSkippedStatus(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	status := []confv1beta1.PhaseState{}
	result := bp.handleSkippedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 1 {
		t.Error("handleSkippedStatus should add one status")
	}
	if result[0].Status != bkev1beta1.PhaseSkipped {
		t.Error("Status should be skipped")
	}
}

func TestBasePhase_HandleWaitingStatus(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStatus(bkev1beta1.PhaseWaiting)
	status := []confv1beta1.PhaseState{}
	result := bp.handleWaitingStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 1 {
		t.Error("handleWaitingStatus should add one status")
	}
}

func TestBasePhase_HandleRunningStatus(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	bkeCluster := &bkev1beta1.BKECluster{}
	pc.SetBKECluster(bkeCluster)

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	status := []confv1beta1.PhaseState{}
	result := bp.handleRunningStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), bkeCluster)
	if len(result) != 1 {
		t.Error("handleRunningStatus should add one status")
	}
	if result[0].Status != bkev1beta1.PhaseRunning {
		t.Error("Status should be running")
	}
}

func TestBasePhase_HandleCompletedStatus(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	bp.SetStatus(bkev1beta1.PhaseSucceeded)
	status := []confv1beta1.PhaseState{}
	result := bp.handleCompletedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), "test message")
	if len(result) != 1 {
		t.Error("handleCompletedStatus should add one status")
	}
	if result[0].Message != "test message" {
		t.Error("Message not set correctly")
	}
}

func TestBasePhase_NeedExecute(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}
	result := bp.NeedExecute(old, new)
	if result {
		t.Log("NeedExecute returned true")
	}
}

func TestBasePhase_Execute(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	defer func() {
		if r := recover(); r == nil {
			t.Error("Execute should panic")
		}
	}()
	bp.Execute()
}

func TestBasePhase_DefaultNeedExecute(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}
	result := bp.DefaultNeedExecute(old, new)
	if result {
		t.Log("DefaultNeedExecute returned true")
	}
}

func TestBasePhase_DefaultNeedExecute_WithType(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	old := &bkev1beta1.BKECluster{}
	new := &bkev1beta1.BKECluster{}
	result := bp.DefaultNeedExecute(old, new)
	if result {
		t.Log("DefaultNeedExecute returned result")
	}
}

func TestBasePhase_Report_OnlyRecord(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStatus(bkev1beta1.PhaseRunning)
	bp.SetStartTime(metav1.Now())

	err := bp.Report("test", true)
	if err != nil {
		t.Errorf("Report with onlyRecord failed: %v", err)
	}
}

func TestBasePhase_HandleSkippedStatus_WithExisting(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseWaiting},
	}
	result := bp.handleSkippedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 1 || result[0].Status != bkev1beta1.PhaseSkipped {
		t.Error("handleSkippedStatus failed")
	}
}

func TestBasePhase_HandleWaitingStatus_WithSucceeded(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStatus(bkev1beta1.PhaseWaiting)
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseSucceeded},
	}
	result := bp.handleWaitingStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 1 {
		t.Error("handleWaitingStatus failed")
	}
}

func TestBasePhase_HandleRunningStatus_WithExisting(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	bkeCluster := &bkev1beta1.BKECluster{}
	pc.SetBKECluster(bkeCluster)

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseWaiting},
	}
	result := bp.handleRunningStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), bkeCluster)
	if len(result) != 1 || result[0].Status != bkev1beta1.PhaseRunning {
		t.Error("handleRunningStatus failed")
	}
}

func TestBasePhase_HandleCompletedStatus_WithExisting(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	bp.SetStatus(bkev1beta1.PhaseSucceeded)
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseRunning},
	}
	result := bp.handleCompletedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), "success")
	if len(result) != 1 || result[0].Status != bkev1beta1.PhaseSucceeded {
		t.Error("handleCompletedStatus failed")
	}
}

func TestBasePhase_CheckCommonNeedExecute_WithDeletion(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	now := metav1.Now()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{DeletionTimestamp: &now},
	}
	if bp.checkCommonNeedExecute(cluster) {
		t.Error("Should not execute for deleted cluster")
	}
}

func TestBasePhase_Report_AllStatuses(t *testing.T) {
	statuses := []confv1beta1.BKEClusterPhaseStatus{
		bkev1beta1.PhaseSkipped,
		bkev1beta1.PhaseWaiting,
		bkev1beta1.PhaseRunning,
		bkev1beta1.PhaseSucceeded,
		bkev1beta1.PhaseFailed,
	}

	for _, status := range statuses {
		pc := NewReconcilePhaseCtx(context.Background())
		pc.SetLogger(&bkev1beta1.BKELogger{})
		pc.SetBKECluster(&bkev1beta1.BKECluster{})

		bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
		bp.SetStatus(status)
		bp.SetStartTime(metav1.Now())
		bp.Report("test", true)
	}
}

func TestBKEClusterPhase_Methods(t *testing.T) {
	phase := confv1beta1.BKEClusterPhase("TestPhase")
	if phase.String() != "TestPhase" {
		t.Error("String() failed")
	}

	phases := confv1beta1.BKEClusterPhases{phase}
	if !phase.In(phases) {
		t.Error("In() failed")
	}
	if phase.NotIn(phases) {
		t.Error("NotIn() failed")
	}

	phases.Add(confv1beta1.BKEClusterPhase("Phase2"))
	if len(phases) != 2 {
		t.Error("Add() failed")
	}
}

func TestBasePhase_HandleWaitingStatus_WithFailed(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStatus(bkev1beta1.PhaseWaiting)
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseFailed},
	}
	result := bp.handleWaitingStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 2 {
		t.Error("handleWaitingStatus should append after failed")
	}
}

func TestBasePhase_HandleWaitingStatus_WithRunning(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStatus(bkev1beta1.PhaseWaiting)
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseRunning},
	}
	result := bp.handleWaitingStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 1 {
		t.Error("handleWaitingStatus should update running")
	}
}

func TestBasePhase_HandleSkippedStatus_WithRunning(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseRunning},
	}
	result := bp.handleSkippedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"))
	if len(result) != 1 || result[0].Status != bkev1beta1.PhaseSkipped {
		t.Error("handleSkippedStatus should update running to skipped")
	}
}

func TestBasePhase_HandleCompletedStatus_WithWaiting(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	bp.SetStatus(bkev1beta1.PhaseSucceeded)
	status := []confv1beta1.PhaseState{
		{Name: confv1beta1.BKEClusterPhase("TestPhase"), Status: bkev1beta1.PhaseWaiting},
	}
	result := bp.handleCompletedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), "done")
	if len(result) != 1 {
		t.Error("handleCompletedStatus should update waiting")
	}
}

func TestBasePhase_CustomHooks(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))

	called := 0
	preHook := func(p Phase) error { called++; return nil }
	postHook := func(p Phase, err error) error { called++; return nil }

	bp.RegisterPreHooks(preHook)
	bp.RegisterPostHooks(postHook)

	for _, f := range bp.CustomPreHookFuncs {
		f(&bp)
	}
	for _, f := range bp.CustomPostHookFuncs {
		f(&bp, nil)
	}

	if called != 2 {
		t.Error("Hooks not called")
	}
}

func TestBasePhase_Report_EmptyStatus(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	err := bp.Report("test", true)
	if err != nil {
		t.Errorf("Report failed: %v", err)
	}
}

func TestBasePhase_CheckCommonNeedExecute_FailedHealth(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{ClusterStatus: "InitializationFailed"},
	}
	if bp.checkCommonNeedExecute(cluster) {
		t.Error("Should not execute for failed status")
	}
}

func TestBasePhase_HandleRunningStatus_Empty(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	bkeCluster := &bkev1beta1.BKECluster{}
	pc.SetBKECluster(bkeCluster)

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	status := []confv1beta1.PhaseState{}
	result := bp.handleRunningStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), bkeCluster)
	if len(result) != 1 {
		t.Error("handleRunningStatus should add status")
	}
}

func TestBasePhase_HandleCompletedStatus_Empty(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.SetLogger(&bkev1beta1.BKELogger{})
	pc.SetBKECluster(&bkev1beta1.BKECluster{})

	bp := NewBasePhase(pc, confv1beta1.BKEClusterPhase("TestPhase"))
	bp.SetStartTime(metav1.Now())
	bp.SetStatus(bkev1beta1.PhaseSucceeded)
	status := []confv1beta1.PhaseState{}
	result := bp.handleCompletedStatus(status, confv1beta1.BKEClusterPhase("TestPhase"), "done")
	if len(result) != 1 {
		t.Error("handleCompletedStatus should add status")
	}
}
