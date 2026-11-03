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

	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/k8s"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/shell"
)

// Task Global tasks
type Task struct {
	mu sync.RWMutex
	// Ctx is cancelled on delete / ActiveDeadline / SafeClose. Prefer Ctx.Done() over StopChan.
	Ctx                     context.Context
	Cancel                  context.CancelFunc
	StopChan                chan struct{}        `json:"stopChan"` // closed by SafeClose for legacy listeners
	Phase                   v1beta1.CommandPhase `json:"phase"`
	ResourceVersion         string               `json:"resourceVersion"`
	Generation              int64                `json:"generation"`
	TTLSecondsAfterFinished int                  `json:"ttlSecondsAfterFinished"`
	HasAddTimer             bool                 `json:"hasAddTimer"`
	Once                    *sync.Once           `json:"once"`
}

func (t *Task) SetPhase(phase v1beta1.CommandPhase) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Phase = phase
}

func (t *Task) GetPhase() v1beta1.CommandPhase {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.Phase
}

func (t *Task) GetGeneration() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.Generation
}

func (t *Task) GetResourceVersion() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.ResourceVersion
}

// ShouldProcessTTL reports whether the task is complete and eligible for TTL cleanup.
func (t *Task) ShouldProcessTTL() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return !t.HasAddTimer && t.TTLSecondsAfterFinished != 0 && t.Phase == v1beta1.CommandComplete
}

// MarkTimerAdded marks the task as scheduled for TTL deletion.
// Returns ttl seconds and true when the timer is newly scheduled.
func (t *Task) MarkTimerAdded() (int, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.HasAddTimer {
		return 0, false
	}
	t.HasAddTimer = true
	return t.TTLSecondsAfterFinished, true
}

type Job struct {
	BuiltIn builtin.BuiltIn
	K8s     k8s.K8s
	Shell   shell.Shell
	Task    map[string]*Task
	taskMu  sync.RWMutex
}

func NewJob(client client.Client) (Job, error) {

	var j Job
	commandExec := &exec.CommandExecutor{}

	j.BuiltIn = builtin.New(commandExec, client)
	j.K8s = &k8s.Task{
		K8sClient: client,
		Exec:      commandExec,
	}
	j.Shell = &shell.Task{
		Exec: commandExec,
	}
	j.Task = map[string]*Task{}

	return j, nil
}

// SafeClose cancels the task context (and closes StopChan once) so in-flight BuiltIn work can exit.
func (t *Task) SafeClose() {
	t.Once.Do(func() {
		if t.Cancel != nil {
			t.Cancel()
		}
		if t.StopChan != nil {
			close(t.StopChan)
		}
	})
}

func (j *Job) GetTask(gid string) (*Task, bool) {
	j.taskMu.RLock()
	defer j.taskMu.RUnlock()
	t, ok := j.Task[gid]
	return t, ok
}

func (j *Job) SetTask(gid string, task *Task) {
	j.taskMu.Lock()
	defer j.taskMu.Unlock()
	j.Task[gid] = task
}

func (j *Job) DeleteTask(gid string) {
	j.taskMu.Lock()
	defer j.taskMu.Unlock()
	delete(j.Task, gid)
}

func (j *Job) SnapshotTasks() map[string]*Task {
	j.taskMu.RLock()
	defer j.taskMu.RUnlock()
	snap := make(map[string]*Task, len(j.Task))
	for k, v := range j.Task {
		snap[k] = v
	}
	return snap
}
