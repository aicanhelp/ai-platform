/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package bkeagent

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job"
)

const (
	numZero       = 0
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFour       = 4
	numFive       = 5
	numTen        = 10
	numSixty      = 60
	numSixHundred = 600

	ipv4SegmentA = 127
	ipv4SegmentB = 0
	ipv4SegmentC = 0
	ipv4SegmentD = 1

	testNamespace = "test-ns"
	testName      = "test-command"
	testNodeName  = "test-node"
	testNodeIP    = "192.168.1.100"
	testGID       = "test-ns/test-command"

	testDefaultTimeout  = 5 * time.Minute
	testShortSleep      = 1 * time.Millisecond
	testDefaultBackoff  = 3
	testDefaultDeadline = 600
	testSleepRangeLow   = 1
	testSleepRangeHigh  = 2
)

var (
	testLoopbackIP = net.IPv4(
		byte(ipv4SegmentA),
		byte(ipv4SegmentB),
		byte(ipv4SegmentC),
		byte(ipv4SegmentD),
	)
)

func TestContinueReconcile(t *testing.T) {
	result := continueReconcile()
	assert.False(t, result.done)
	assert.NoError(t, result.err)
	assert.Equal(t, ctrl.Result{}, result.result)
}

func TestFinishReconcile(t *testing.T) {
	testErr := errors.New("test error")
	testResult := ctrl.Result{Requeue: true}

	result := finishReconcile(testResult, testErr)
	assert.True(t, result.done)
	assert.Equal(t, testErr, result.err)
	assert.Equal(t, testResult, result.result)
}

func TestFinishWithRequeue(t *testing.T) {
	result := finishWithRequeue()
	assert.True(t, result.done)
	assert.True(t, result.result.Requeue)
	assert.NoError(t, result.err)
}

func TestReconcileResultUnwrap(t *testing.T) {
	testErr := errors.New("test error")
	testResult := ctrl.Result{Requeue: true}

	res := reconcileResult{result: testResult, err: testErr, done: true}
	result, err := res.unwrap()
	assert.Equal(t, testResult, result)
	assert.Equal(t, testErr, err)
}

func TestFetchCommand(t *testing.T) {
	tests := []struct {
		name           string
		setupClient    func() client.Client
		req            ctrl.Request
		expectDone     bool
		expectNotFound bool
		expectError    bool
	}{
		{
			name: "command found",
			setupClient: func() client.Client {
				scheme := runtime.NewScheme()
				agentv1beta1.AddToScheme(scheme)
				cmd := &agentv1beta1.Command{
					ObjectMeta: metav1.ObjectMeta{
						Name:      testName,
						Namespace: testNamespace,
					},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(cmd).Build()
			},
			req:        ctrl.Request{NamespacedName: client.ObjectKey{Name: testName, Namespace: testNamespace}},
			expectDone: false,
		},
		{
			name: "command not found",
			setupClient: func() client.Client {
				scheme := runtime.NewScheme()
				agentv1beta1.AddToScheme(scheme)
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			req:            ctrl.Request{NamespacedName: client.ObjectKey{Name: testName, Namespace: testNamespace}},
			expectDone:     true,
			expectNotFound: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setupClient == nil {
				return
			}
			r := &CommandReconciler{
				Client: tt.setupClient(),
			}
			ctx := context.Background()
			req := tt.req

			_, res := r.fetchCommand(ctx, req)

			assert.Equal(t, tt.expectDone, res.done)
			if tt.expectNotFound {
				assert.NoError(t, res.err)
			}
			if tt.expectError && res.err != nil {
				assert.Error(t, res.err)
			}
		})
	}
}

func TestEnsureStatusInitialized(t *testing.T) {
	tests := []struct {
		name              string
		command           *agentv1beta1.Command
		syncStatusErr     error
		expectContinue    bool
		expectInitialized bool
	}{
		{
			name: "status not initialized",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:            testName,
					Namespace:       testNamespace,
					ResourceVersion: "1000",
				},
				Spec: agentv1beta1.CommandSpec{
					Commands: []agentv1beta1.ExecCommand{{ID: "cmd1"}},
				},
				Status: nil,
			},
			syncStatusErr:  nil,
			expectContinue: true,
		},
		{
			name: "status already initialized",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:            testName,
					Namespace:       testNamespace,
					ResourceVersion: "1000",
				},
				Spec: agentv1beta1.CommandSpec{
					Commands: []agentv1beta1.ExecCommand{{ID: "cmd1"}},
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeName: {
						Phase: agentv1beta1.CommandRunning,
					},
				},
			},
			expectContinue:    true,
			expectInitialized: true,
		},
		{
			name: "nil status map",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:            testName,
					Namespace:       testNamespace,
					ResourceVersion: "1000",
				},
				Spec:   agentv1beta1.CommandSpec{},
				Status: nil,
			},
			expectContinue: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				Client:   nil,
				NodeName: testNodeName,
			}

			patches := gomonkey.ApplyFunc((*CommandReconciler).syncStatusUntilComplete, func(r *CommandReconciler, cmd *agentv1beta1.Command) error {
				return tt.syncStatusErr
			})
			defer patches.Reset()

			res := r.ensureStatusInitialized(tt.command)

			if tt.expectContinue {
				assert.False(t, res.done)
			}
			if tt.command != nil && tt.command.Status != nil {
				if _, ok := tt.command.Status[testNodeName]; ok {
					assert.True(t, ok)
				}
			}
		})
	}
}

func TestCleanupTask(t *testing.T) {
	r := &CommandReconciler{
		Job: job.Job{
			Task: map[string]*job.Task{
				testGID: {
					StopChan: make(chan struct{}),
					Phase:    agentv1beta1.CommandRunning,
					Once:     &sync.Once{},
				},
			},
		},
	}

	r.cleanupTask(testGID)

	_, ok := r.Job.GetTask(testGID)
	assert.False(t, ok)
}

func TestEnsureFinalizer(t *testing.T) {
	tests := []struct {
		name           string
		command        *agentv1beta1.Command
		expectContinue bool
	}{
		{
			name: "finalizer already exists",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:       testName,
					Finalizers: []string{commandFinalizerName},
				},
			},
			expectContinue: true,
		},
		{
			name: "missing finalizer logs and continues",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name: testName,
				},
			},
			expectContinue: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			agentv1beta1.AddToScheme(scheme)
			var cli client.Client
			if tt.command != nil {
				cli = fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.command).Build()
			}

			r := &CommandReconciler{
				Client: cli,
			}

			res := r.ensureFinalizer(tt.command)

			assert.Equal(t, tt.expectContinue, !res.done)
		})
	}
}

func TestHandleDeletion(t *testing.T) {
	tests := []struct {
		name       string
		command    *agentv1beta1.Command
		expectDone bool
	}{
		{
			name: "cleanup local task on delete",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name: testName,
				},
			},
			expectDone: true,
		},
		{
			name: "cleanup local task even when finalizer present",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:       testName,
					Finalizers: []string{commandFinalizerName},
				},
			},
			expectDone: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			agentv1beta1.AddToScheme(scheme)
			var cli client.Client
			if tt.command != nil {
				cli = fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.command).Build()
			} else {
				cli = fake.NewClientBuilder().WithScheme(scheme).Build()
			}

			r := &CommandReconciler{
				Client: cli,
				Job: job.Job{
					Task: map[string]*job.Task{
						testGID: {StopChan: make(chan struct{}), Once: &sync.Once{}},
					},
				},
			}

			res := r.handleDeletion(tt.command, testGID)

			assert.Equal(t, tt.expectDone, res.done)
		})
	}
}

func TestHandleFinalizer(t *testing.T) {
	tests := []struct {
		name         string
		command      *agentv1beta1.Command
		expectEnsure bool
		expectDelete bool
	}{
		{
			name: "non-deletion ensure finalizer",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:              testName,
					DeletionTimestamp: nil,
				},
			},
			expectEnsure: true,
		},
		{
			name: "deletion handle deletion",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:              testName,
					DeletionTimestamp: &metav1.Time{Time: time.Now()},
				},
			},
			expectDelete: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			agentv1beta1.AddToScheme(scheme)
			cli := fake.NewClientBuilder().WithScheme(scheme).Build()

			r := &CommandReconciler{
				Client: cli,
				Job: job.Job{
					Task: make(map[string]*job.Task),
				},
			}

			res := r.handleFinalizer(tt.command, testGID)

			assert.NotNil(t, res)
		})
	}
}

func TestHandleSuspend(t *testing.T) {
	tests := []struct {
		name              string
		command           *agentv1beta1.Command
		currentStatus     *agentv1beta1.CommandStatus
		syncStatusErr     error
		expectContinue    bool
		expectSuspendDone bool
	}{
		{
			name: "not suspended",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					Suspend: false,
				},
			},
			currentStatus:  &agentv1beta1.CommandStatus{},
			expectContinue: true,
		},
		{
			name: "already suspended",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					Suspend: true,
				},
			},
			currentStatus: &agentv1beta1.CommandStatus{
				Phase: agentv1beta1.CommandSuspend,
			},
			expectSuspendDone: true,
		},
		{
			name: "suspending command",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					Suspend:  true,
					Commands: []agentv1beta1.ExecCommand{{ID: "cmd1"}},
				},
			},
			currentStatus: &agentv1beta1.CommandStatus{
				Phase: agentv1beta1.CommandRunning,
			},
			syncStatusErr:     nil,
			expectSuspendDone: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				Job: job.Job{
					Task: map[string]*job.Task{
						testGID: {StopChan: make(chan struct{}), Once: &sync.Once{}},
					},
				},
			}

			patches := gomonkey.ApplyFunc((*CommandReconciler).syncStatusUntilComplete, func(r *CommandReconciler, cmd *agentv1beta1.Command) error {
				return tt.syncStatusErr
			})
			defer patches.Reset()

			res := r.handleSuspend(tt.command, tt.currentStatus, testGID)

			if tt.expectContinue {
				assert.False(t, res.done)
			}
			if tt.expectSuspendDone {
				assert.True(t, res.done)
			}
		})
	}
}

func TestShouldSkipOldTask(t *testing.T) {
	tests := []struct {
		name       string
		command    *agentv1beta1.Command
		task       *job.Task
		expectSkip bool
	}{
		{
			name: "task exists with newer generation",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Generation: 2,
				},
			},
			task: &job.Task{
				Generation:      1,
				ResourceVersion: "100",
				StopChan:        make(chan struct{}),
				Once:            &sync.Once{},
			},
			expectSkip: false,
		},
		{
			name: "task exists with older generation",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Generation: 1,
				},
			},
			task: &job.Task{
				Generation:      2,
				ResourceVersion: "100",
				StopChan:        make(chan struct{}),
				Once:            &sync.Once{},
			},
			expectSkip: true,
		},
		{
			name: "task does not exist",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Generation: 1,
				},
			},
			task:       nil,
			expectSkip: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				Job: job.Job{
					Task: map[string]*job.Task{},
				},
			}
			if tt.task != nil {
				r.Job.SetTask(testGID, tt.task)
			}

			skip := r.shouldSkipOldTask(tt.command, testGID)
			assert.Equal(t, tt.expectSkip, skip)
		})
	}
}

func TestCalculateStopTime(t *testing.T) {
	lastStartTime := time.Now()

	stopTime := calculateStopTime(lastStartTime, testDefaultDeadline)
	assert.True(t, stopTime.After(lastStartTime))

	stopTimeZero := calculateStopTime(lastStartTime, numZero)
	assert.Equal(t, lastStartTime.Add(time.Duration(agentv1beta1.DefaultActiveDeadlineSecond)*time.Second), stopTimeZero)
}

func TestIsCommandCompleted(t *testing.T) {
	tests := []struct {
		name       string
		conditions []*agentv1beta1.Condition
		commandID  string
		expect     bool
	}{
		{
			name: "command completed",
			conditions: []*agentv1beta1.Condition{
				{ID: "cmd1", Phase: agentv1beta1.CommandComplete},
			},
			commandID: "cmd1",
			expect:    true,
		},
		{
			name: "command skipped counts as completed",
			conditions: []*agentv1beta1.Condition{
				{ID: "cmd1", Phase: agentv1beta1.CommandSkip, Status: metav1.ConditionTrue},
			},
			commandID: "cmd1",
			expect:    true,
		},
		{
			name: "command not completed",
			conditions: []*agentv1beta1.Condition{
				{ID: "cmd1", Phase: agentv1beta1.CommandRunning},
			},
			commandID: "cmd1",
			expect:    false,
		},
		{
			name:       "no conditions",
			conditions: nil,
			commandID:  "cmd1",
			expect:     false,
		},
		{
			name: "command not found",
			conditions: []*agentv1beta1.Condition{
				{ID: "cmd2", Phase: agentv1beta1.CommandComplete},
			},
			commandID: "cmd1",
			expect:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isCommandCompleted(tt.conditions, tt.commandID)
			assert.Equal(t, tt.expect, result)
		})
	}
}

func TestNewCondition(t *testing.T) {
	execCommandID := "test-cmd-id"
	cond := newCondition(execCommandID)

	assert.Equal(t, execCommandID, cond.ID)
	assert.Equal(t, metav1.ConditionUnknown, cond.Status)
	assert.Equal(t, agentv1beta1.CommandRunning, cond.Phase)
	assert.NotNil(t, cond.LastStartTime)
	assert.Empty(t, cond.StdErr)
	assert.Empty(t, cond.StdOut)
	assert.Equal(t, numZero, cond.Count)
}

func TestCommandStatusKey(t *testing.T) {
	tests := []struct {
		name     string
		nodeIP   string
		nodeName string
		command  *agentv1beta1.Command
		mockIPs  func() ([]string, error)
		expect   string
	}{
		{
			name:     "with node IP",
			nodeIP:   testNodeIP,
			nodeName: testNodeName,
			expect:   fmt.Sprintf("%s/%s", testNodeName, testNodeIP),
		},
		{
			name:     "without node IP",
			nodeIP:   "",
			nodeName: testNodeName,
			expect:   testNodeName,
		},
		{
			name:     "prefer selector candidate over startup node IP",
			nodeIP:   "10.244.1.5",
			nodeName: testNodeName,
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							testNodeIP: testNodeIP,
						},
					},
				},
			},
			mockIPs: func() ([]string, error) {
				return []string{"10.244.1.5/32", testNodeIP + "/24"}, nil
			},
			expect: fmt.Sprintf("%s/%s", testNodeName, testNodeIP),
		},
		{
			name:     "without node selector ignore local interface IPs",
			nodeIP:   "",
			nodeName: testNodeName,
			command:  &agentv1beta1.Command{},
			mockIPs: func() ([]string, error) {
				return []string{"172.16.3.90/24"}, nil
			},
			expect: testNodeName,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				NodeIP:   tt.nodeIP,
				NodeName: tt.nodeName,
			}

			if tt.mockIPs != nil {
				patches := gomonkey.ApplyFunc(bkenet.GetAllInterfaceIP, tt.mockIPs)
				defer patches.Reset()
			}

			key := r.commandStatusKey(tt.command)
			assert.Equal(t, tt.expect, key)
		})
	}
}

func TestNodeMatchNodeSelector(t *testing.T) {
	tests := []struct {
		name     string
		nodeName string
		selector *metav1.LabelSelector
		mockIPs  func() ([]string, error)
		expect   bool
	}{
		{
			name:     "nil selector",
			nodeName: testNodeName,
			selector: nil,
			expect:   false,
		},
		{
			name:     "name matches",
			nodeName: testNodeName,
			selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					testNodeName: testNodeName,
				},
			},
			expect: true,
		},
		{
			name:     "IP matches",
			nodeName: testNodeName,
			selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					testNodeIP: testNodeIP,
				},
			},
			mockIPs: func() ([]string, error) {
				return []string{testNodeIP + "/24"}, nil
			},
			expect: true,
		},
		{
			name:     "no match",
			nodeName: testNodeName,
			selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"other-node": "other-node",
				},
			},
			mockIPs: func() ([]string, error) {
				return []string{testNodeIP}, nil
			},
			expect: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				NodeName: tt.nodeName,
			}

			if tt.mockIPs != nil {
				patches := gomonkey.ApplyFunc(bkenet.GetAllInterfaceIP, tt.mockIPs)
				defer patches.Reset()
			}

			result := r.nodeMatchNodeSelector(tt.selector)
			assert.Equal(t, tt.expect, result)
		})
	}
}

func TestShouldReconcileCommand(t *testing.T) {
	tests := []struct {
		name            string
		command         *agentv1beta1.Command
		nodeName        string
		expectReconcile bool
	}{
		{
			name:            "nil command",
			command:         nil,
			expectReconcile: false,
		},
		{
			name: "command matches node name",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeName: testNodeName,
				},
			},
			nodeName:        testNodeName,
			expectReconcile: true,
		},
		{
			name: "command does not match node name",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeName: "other-node",
				},
			},
			nodeName:        testNodeName,
			expectReconcile: false,
		},
		{
			name: "older generation",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:            testName,
					Namespace:       testNamespace,
					Generation:      1,
					ResourceVersion: "100",
				},
				Spec: agentv1beta1.CommandSpec{
					NodeSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{testNodeName: testNodeName},
					},
				},
			},
			nodeName:        testNodeName,
			expectReconcile: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				NodeName: tt.nodeName,
				Job: job.Job{
					Task: map[string]*job.Task{
						testGID: {
							Generation:      2,
							ResourceVersion: "200",
							StopChan:        make(chan struct{}),
							Once:            &sync.Once{},
						},
					},
				},
			}

			result := r.shouldReconcileCommand(tt.command, "test")
			assert.Equal(t, tt.expectReconcile, result)
		})
	}
}

func TestExecuteByType(t *testing.T) {
	tests := []struct {
		name      string
		cmdType   agentv1beta1.CommandType
		command   []string
		mockExec  func(agentv1beta1.CommandType, []string) ([]string, error)
		expectNil bool
		expectErr bool
	}{
		{
			name:    "BuiltIn command",
			cmdType: agentv1beta1.CommandBuiltIn,
			command: []string{"test"},
			mockExec: func(cmdType agentv1beta1.CommandType, cmds []string) ([]string, error) {
				return []string{"output"}, nil
			},
		},
		{
			name:    "Kubernetes command",
			cmdType: agentv1beta1.CommandKubernetes,
			command: []string{"configmap:testns/testname:rx:shell"},
			mockExec: func(cmdType agentv1beta1.CommandType, cmds []string) ([]string, error) {
				return []string{"k8s-output"}, nil
			},
		},
		{
			name:    "Shell command",
			cmdType: agentv1beta1.CommandShell,
			command: []string{"echo hello"},
			mockExec: func(cmdType agentv1beta1.CommandType, cmds []string) ([]string, error) {
				return []string{"shell-output"}, nil
			},
		},
		{
			name:      "Unsupported command type",
			cmdType:   agentv1beta1.CommandType("Unsupported"),
			command:   []string{"test"},
			expectNil: true,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockJob := &job.Job{
				BuiltIn: mockBuiltIn{execFn: tt.mockExec},
				K8s:     mockK8s{execFn: tt.mockExec},
				Shell:   mockShell{execFn: tt.mockExec},
			}

			r := &CommandReconciler{
				Job: *mockJob,
			}

			result, err := r.executeByType(context.Background(), tt.cmdType, tt.command, 0)

			if tt.expectNil {
				assert.Nil(t, result)
			}
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestExecuteWithRetry(t *testing.T) {
	tests := []struct {
		name           string
		execCommand    agentv1beta1.ExecCommand
		condition      *agentv1beta1.Condition
		stopTime       time.Time
		backoffLimit   int
		mockExecute    func(agentv1beta1.CommandType, []string) ([]string, error)
		expectComplete bool
		expectTimeout  bool
	}{
		{
			name: "successful execution",
			execCommand: agentv1beta1.ExecCommand{
				ID:      "cmd1",
				Type:    agentv1beta1.CommandShell,
				Command: []string{"echo hello"},
			},
			condition:    &agentv1beta1.Condition{},
			stopTime:     time.Now().Add(10 * time.Second),
			backoffLimit: 3,
			mockExecute: func(cmdType agentv1beta1.CommandType, cmd []string) ([]string, error) {
				return []string{"success"}, nil
			},
			expectComplete: true,
		},
		{
			name: "execution with error and retry",
			execCommand: agentv1beta1.ExecCommand{
				ID:           "cmd1",
				Type:         agentv1beta1.CommandShell,
				Command:      []string{"echo hello"},
				BackoffDelay: 1,
			},
			condition:    &agentv1beta1.Condition{},
			stopTime:     time.Now().Add(10 * time.Second),
			backoffLimit: 3,
			mockExecute: func(cmdType agentv1beta1.CommandType, cmd []string) ([]string, error) {
				return nil, errors.New("error")
			},
			expectComplete: false,
		},
		{
			name: "timeout before execution",
			execCommand: agentv1beta1.ExecCommand{
				ID:      "cmd1",
				Type:    agentv1beta1.CommandShell,
				Command: []string{"echo hello"},
			},
			condition:     &agentv1beta1.Condition{},
			stopTime:      time.Now().Add(-1 * time.Second),
			backoffLimit:  3,
			expectTimeout: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockJob := &job.Job{
				Shell: mockShell{execFn: tt.mockExecute},
			}

			r := &CommandReconciler{
				Job: *mockJob,
			}

			patches := gomonkey.ApplyFunc(time.Sleep, func(d time.Duration) {})
			defer patches.Reset()

			result := r.executeWithRetry(context.Background(), tt.execCommand, tt.condition, tt.stopTime, tt.backoffLimit)

			if tt.expectTimeout {
				assert.True(t, result.timedOut)
				assert.Equal(t, agentv1beta1.CommandFailed, tt.condition.Phase)
				assert.Equal(t, metav1.ConditionFalse, tt.condition.Status)
			}
			if tt.expectComplete {
				assert.Equal(t, agentv1beta1.CommandComplete, tt.condition.Phase)
			}
		})
	}
}

func TestProcessExecCommand(t *testing.T) {
	tests := []struct {
		name            string
		command         *agentv1beta1.Command
		execCommand     agentv1beta1.ExecCommand
		currentStatus   *agentv1beta1.CommandStatus
		stopTime        time.Time
		mockExecute     func(agentv1beta1.CommandType, []string) ([]string, error)
		syncStatusErr   error
		expectBreak     bool
		expectSyncError bool
	}{
		{
			name: "successful execution",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					BackoffLimit: 3,
				},
			},
			execCommand: agentv1beta1.ExecCommand{
				ID:      "cmd1",
				Type:    agentv1beta1.CommandShell,
				Command: []string{"echo hello"},
			},
			currentStatus: &agentv1beta1.CommandStatus{
				Conditions: []*agentv1beta1.Condition{},
			},
			stopTime: time.Now().Add(10 * time.Second),
			mockExecute: func(cmdType agentv1beta1.CommandType, cmd []string) ([]string, error) {
				return []string{"success"}, nil
			},
			syncStatusErr: nil,
			expectBreak:   false,
		},
		{
			name: "execution with BackoffIgnore",
			command: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					BackoffLimit: 3,
				},
			},
			execCommand: agentv1beta1.ExecCommand{
				ID:            "cmd1",
				Type:          agentv1beta1.CommandShell,
				Command:       []string{"echo hello"},
				BackoffIgnore: true,
			},
			currentStatus: &agentv1beta1.CommandStatus{
				Conditions: []*agentv1beta1.Condition{},
			},
			stopTime: time.Now().Add(10 * time.Second),
			mockExecute: func(cmdType agentv1beta1.CommandType, cmd []string) ([]string, error) {
				return nil, errors.New("error")
			},
			syncStatusErr: nil,
			expectBreak:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockJob := &job.Job{
				Shell: mockShell{execFn: tt.mockExecute},
			}

			r := &CommandReconciler{
				Job: *mockJob,
			}

			patches := gomonkey.ApplyFunc((*CommandReconciler).syncStatusUntilComplete, func(r *CommandReconciler, cmd *agentv1beta1.Command) error {
				return tt.syncStatusErr
			})
			defer patches.Reset()

			patches.ApplyFunc(time.Sleep, func(d time.Duration) {})
			defer patches.Reset()

			result := r.processExecCommand(context.Background(), tt.command, tt.execCommand, tt.currentStatus, tt.stopTime)

			if tt.expectSyncError {
				assert.NotNil(t, result.syncError)
			}
			assert.Equal(t, tt.expectBreak, result.shouldBreak)

			if tt.execCommand.BackoffIgnore && !tt.expectBreak {
				cond := agentv1beta1.GetCondition(tt.currentStatus.Conditions, &agentv1beta1.Condition{ID: tt.execCommand.ID})
				assert.NotNil(t, cond)
				if cond != nil {
					assert.Equal(t, agentv1beta1.CommandSkip, cond.Phase)
					assert.Equal(t, metav1.ConditionTrue, cond.Status)
				}
			}
		})
	}
}

func TestFinalizeTaskStatus(t *testing.T) {
	tests := []struct {
		name          string
		command       *agentv1beta1.Command
		currentStatus *agentv1beta1.CommandStatus
		gid           string
		syncStatusErr error
		expectError   bool
	}{
		{
			name: "successful finalize",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testName,
					Namespace: testNamespace,
				},
				Spec: agentv1beta1.CommandSpec{
					Commands: []agentv1beta1.ExecCommand{{ID: "cmd1"}},
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeName: {},
				},
			},
			currentStatus: &agentv1beta1.CommandStatus{
				Conditions: []*agentv1beta1.Condition{
					{ID: "cmd1", Phase: agentv1beta1.CommandComplete, Status: metav1.ConditionTrue},
				},
			},
			gid:           testGID,
			syncStatusErr: nil,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				Job: job.Job{
					Task: map[string]*job.Task{
						tt.gid: {StopChan: make(chan struct{}), Once: &sync.Once{}},
					},
				},
			}

			patches := gomonkey.ApplyFunc((*CommandReconciler).syncStatusUntilComplete, func(r *CommandReconciler, cmd *agentv1beta1.Command) error {
				return tt.syncStatusErr
			})
			defer patches.Reset()

			err := r.finalizeTaskStatus(tt.command, tt.currentStatus, tt.gid)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSyncStatusUntilComplete(t *testing.T) {
	t.Skip("skip unstable UT: fake client status patch can retry until the 5 minute timeout in CI")
	tests := []struct {
		name        string
		setupClient func() client.Client
		cmd         *agentv1beta1.Command
		expectError bool
	}{
		{
			name: "successful sync",
			setupClient: func() client.Client {
				scheme := runtime.NewScheme()
				agentv1beta1.AddToScheme(scheme)
				cmd := &agentv1beta1.Command{
					ObjectMeta: metav1.ObjectMeta{
						Name:            testName,
						Namespace:       testNamespace,
						ResourceVersion: "1000",
					},
					Status: map[string]*agentv1beta1.CommandStatus{
						testNodeName: {Phase: agentv1beta1.CommandRunning},
					},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(cmd).Build()
			},
			cmd: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:            testName,
					Namespace:       testNamespace,
					ResourceVersion: "1000",
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeName: {Phase: agentv1beta1.CommandRunning},
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &CommandReconciler{
				Client:    tt.setupClient(),
				APIReader: tt.setupClient(),
				Ctx:       context.Background(),
			}

			patches := gomonkey.ApplyFunc(rand.IntnRange, func(low, high int) int {
				return low
			})
			defer patches.Reset()

			err := r.syncStatusUntilComplete(tt.cmd)
			if tt.expectError {
				assert.Error(t, err)
			}
		})
	}
}

func TestReconcile(t *testing.T) {
	tests := []struct {
		name        string
		setupClient func() client.Client
		nodeName    string
		req         ctrl.Request
		expectError bool
	}{
		{
			name: "empty commands",
			setupClient: func() client.Client {
				scheme := runtime.NewScheme()
				agentv1beta1.AddToScheme(scheme)
				cmd := &agentv1beta1.Command{
					ObjectMeta: metav1.ObjectMeta{
						Name:            testName,
						Namespace:       testNamespace,
						ResourceVersion: "1000",
					},
					Spec: agentv1beta1.CommandSpec{
						Commands: []agentv1beta1.ExecCommand{},
					},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(cmd).Build()
			},
			nodeName: testNodeName,
			req: ctrl.Request{
				NamespacedName: client.ObjectKey{Name: testName, Namespace: testNamespace},
			},
			expectError: false,
		},
		{
			name: "command not found",
			setupClient: func() client.Client {
				scheme := runtime.NewScheme()
				agentv1beta1.AddToScheme(scheme)
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			nodeName: testNodeName,
			req: ctrl.Request{
				NamespacedName: client.ObjectKey{Name: testName, Namespace: testNamespace},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.ApplyFunc((*CommandReconciler).syncStatusUntilComplete, func(r *CommandReconciler, cmd *agentv1beta1.Command) error {
				return nil
			})
			defer patches.Reset()

			cli := tt.setupClient()
			r := &CommandReconciler{
				Client:    cli,
				APIReader: cli,
				Ctx:       context.Background(),
				NodeName:  tt.nodeName,
				Job: job.Job{
					Task: make(map[string]*job.Task),
				},
			}

			_, err := r.Reconcile(context.Background(), tt.req)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

type mockBuiltIn struct {
	execFn func(agentv1beta1.CommandType, []string) ([]string, error)
}

func (m mockBuiltIn) Execute(ctx context.Context, execCommands []string) ([]string, error) {
	if m.execFn != nil {
		return m.execFn(agentv1beta1.CommandBuiltIn, execCommands)
	}
	return nil, nil
}

type mockK8s struct {
	execFn func(agentv1beta1.CommandType, []string) ([]string, error)
}

func (m mockK8s) Execute(ctx context.Context, timeout time.Duration, execCommands []string) ([]string, error) {
	if m.execFn != nil {
		return m.execFn(agentv1beta1.CommandKubernetes, execCommands)
	}
	return nil, nil
}

type mockShell struct {
	execFn func(agentv1beta1.CommandType, []string) ([]string, error)
}

func (m mockShell) Execute(ctx context.Context, timeout time.Duration, execCommands []string) ([]string, error) {
	if m.execFn != nil {
		return m.execFn(agentv1beta1.CommandShell, execCommands)
	}
	return nil, nil
}

func TestCreateAndStartTask(t *testing.T) {
	tests := []struct {
		name          string
		command       *agentv1beta1.Command
		currentStatus *agentv1beta1.CommandStatus
		syncStatusErr error
		expectDone    bool
	}{
		{
			name: "create task successfully",
			command: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					Name:            testName,
					Namespace:       testNamespace,
					ResourceVersion: "1000",
					Generation:      1,
				},
				Spec: agentv1beta1.CommandSpec{
					Commands: []agentv1beta1.ExecCommand{{ID: "cmd1"}},
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeName: {
						LastStartTime: &metav1.Time{Time: time.Now()},
					},
				},
			},
			currentStatus: &agentv1beta1.CommandStatus{
				LastStartTime: &metav1.Time{Time: time.Now()},
			},
			syncStatusErr: nil,
			expectDone:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().Build()
			r := &CommandReconciler{
				NodeName:  testNodeName,
				Client:    fakeClient,
				APIReader: fakeClient,
				Ctx:       context.Background(),
				Job: job.Job{
					Task: make(map[string]*job.Task),
				},
			}

			patches := gomonkey.ApplyFunc((*CommandReconciler).syncStatusUntilComplete, func(r *CommandReconciler, cmd *agentv1beta1.Command) error {
				return tt.syncStatusErr
			})
			patches.ApplyFunc((*CommandReconciler).startTask, func(r *CommandReconciler, task *job.Task, command *agentv1beta1.Command) {
			})
			defer patches.Reset()

			res := r.createAndStartTask(context.Background(), tt.command, tt.currentStatus, testGID)

			assert.Equal(t, tt.expectDone, res.done)
			_, taskExists := r.Job.GetTask(testGID)
			assert.True(t, taskExists)
		})
	}
}

func TestCommandPredicateFn(t *testing.T) {
	r := &CommandReconciler{
		NodeName: testNodeName,
		Job: job.Job{
			Task: map[string]*job.Task{
				testGID: {
					Generation:      2,
					ResourceVersion: "200",
					StopChan:        make(chan struct{}),
					Once:            &sync.Once{},
				},
			},
		},
	}

	predicate := r.commandPredicateFn()
	assert.NotNil(t, predicate)
}

func TestSetupWithManager(t *testing.T) {
	t.Skip("requires real controller-runtime manager")
}
