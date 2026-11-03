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

package job

import (
	"context"
	"sync"
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
)

func TestNewJob(t *testing.T) {
	cl := fake.NewClientBuilder().Build()
	j, err := NewJob(cl)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if j.BuiltIn == nil {
		t.Fatal("Expected BuiltIn to be not nil")
	}
	if j.K8s == nil {
		t.Fatal("Expected K8s to be not nil")
	}
	if j.Shell == nil {
		t.Fatal("Expected Shell to be not nil")
	}
	if j.Task == nil {
		t.Fatal("Expected Task map to be not nil")
	}
}

func TestTaskSafeClose(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	task := &Task{
		Ctx:      ctx,
		Cancel:   cancel,
		StopChan: make(chan struct{}),
		Once:     &sync.Once{},
	}

	task.SafeClose()

	select {
	case <-task.Ctx.Done():
	default:
		t.Fatal("Expected task Ctx to be cancelled")
	}
	_, ok := <-task.StopChan
	if ok {
		t.Fatal("Expected channel to be closed")
	}
}

func TestTaskSafeCloseMultipleCalls(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	task := &Task{
		Ctx:      ctx,
		Cancel:   cancel,
		StopChan: make(chan struct{}),
		Once:     &sync.Once{},
	}

	task.SafeClose()
	task.SafeClose()
	task.SafeClose()
}

func TestJobStruct(t *testing.T) {
	cl := fake.NewClientBuilder().Build()
	j, err := NewJob(cl)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if j.BuiltIn == nil {
		t.Fatal("Expected BuiltIn to be initialized")
	}
	if j.K8s == nil {
		t.Fatal("Expected K8s to be initialized")
	}
	if j.Shell == nil {
		t.Fatal("Expected Shell to be initialized")
	}
	if j.Task == nil {
		t.Fatal("Expected Task map to be initialized")
	}
}

func TestTaskFields(t *testing.T) {
	task := &Task{
		StopChan:                make(chan struct{}),
		Phase:                   v1beta1.CommandRunning,
		ResourceVersion:         "12345",
		Generation:              1,
		TTLSecondsAfterFinished: 300,
		HasAddTimer:             true,
		Once:                    &sync.Once{},
	}

	if task.GetPhase() != v1beta1.CommandRunning {
		t.Errorf("Expected Phase to be Running, got %v", task.GetPhase())
	}
	if task.GetResourceVersion() != "12345" {
		t.Errorf("Expected ResourceVersion to be 12345, got %s", task.GetResourceVersion())
	}
	if task.GetGeneration() != 1 {
		t.Errorf("Expected Generation to be 1, got %d", task.GetGeneration())
	}
	if !task.ShouldProcessTTL() && task.HasAddTimer {
		t.Log("HasAddTimer blocks TTL processing as expected")
	}
	if task.Once == nil {
		t.Error("Expected Once to be not nil")
	}
}

func TestTaskConcurrentPhaseAndTTLAccess(t *testing.T) {
	task := &Task{
		StopChan:                make(chan struct{}),
		Phase:                   v1beta1.CommandRunning,
		TTLSecondsAfterFinished: 600,
		Once:                    &sync.Once{},
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 1000; i++ {
			_ = task.ShouldProcessTTL()
		}
	}()

	for i := 0; i < 1000; i++ {
		task.SetPhase(v1beta1.CommandComplete)
		task.SetPhase(v1beta1.CommandRunning)
	}
	<-done
}
