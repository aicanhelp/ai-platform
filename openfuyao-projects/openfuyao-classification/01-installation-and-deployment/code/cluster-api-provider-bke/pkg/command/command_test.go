/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package command

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	numC0       = 0
	numC1       = 1
	numC2       = 2
	numC3       = 3
	testNS      = "test-ns"
	testClName  = "test-cluster"
	testCmdName = "test-command"
	testNodeIP  = "192.168.1.1"
	testNodeIP2 = "192.168.1.2"
	testHost    = "test-node"
	cmdTimeout  = 5 * time.Minute
	cmdInterval = 2 * time.Second
)

func TestGenerateBkeConfigStr(t *testing.T) {
	tests := []struct {
		name      string
		namespace string
		config    string
		expected  string
	}{
		{
			name:      "Normal case",
			namespace: testNS,
			config:    testClName,
			expected:  "bkeConfig=test-ns:test-cluster",
		},
		{
			name:      "Empty namespace",
			namespace: "",
			config:    testClName,
			expected:  "bkeConfig=:test-cluster",
		},
		{
			name:      "Empty config",
			namespace: testNS,
			config:    "",
			expected:  "bkeConfig=test-ns:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GenerateBkeConfigStr(tt.namespace, tt.config)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGenerateDefaultCommandSpec(t *testing.T) {
	spec := GenerateDefaultCommandSpec()

	assert.NotNil(t, spec)
	assert.Equal(t, DefaultBackoffLimit, spec.BackoffLimit)
	assert.Equal(t, 0, spec.ActiveDeadlineSecond)
	assert.Equal(t, DefaultTTLSecondsAfterFinished, spec.TTLSecondsAfterFinished)
	assert.NotNil(t, spec.NodeSelector)
	assert.False(t, spec.Suspend)
}

func TestValidateCommand(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *agentv1beta1.Command
		wantErr bool
	}{
		{
			name: "Valid with node name",
			cmd: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeName: testNodeIP,
				},
			},
			wantErr: false,
		},
		{
			name: "Valid with node selector",
			cmd: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"test": "value",
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateCommand(tt.cmd)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestCheckCommandStatus(t *testing.T) {
	tests := []struct {
		name           string
		cmd            *agentv1beta1.Command
		expectComplete bool
	}{
		{
			name: "Nil status",
			cmd: &agentv1beta1.Command{
				Status: nil,
			},
			expectComplete: false,
		},
		{
			name: "Suspended command",
			cmd: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					Suspend: true,
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeIP: {
						Phase: agentv1beta1.CommandRunning,
					},
				},
			},
			expectComplete: false,
		},
		{
			name: "Complete - all success",
			cmd: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							testNodeIP: testNodeIP,
						},
					},
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeIP: {
						Phase:  agentv1beta1.CommandComplete,
						Status: metav1.ConditionTrue,
					},
				},
			},
			expectComplete: true,
		},
		{
			name: "Complete - Skip counts as success",
			cmd: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							testNodeIP: testNodeIP,
						},
					},
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeIP: {
						Phase:  agentv1beta1.CommandSkip,
						Status: metav1.ConditionTrue,
					},
				},
			},
			expectComplete: true,
		},
		{
			name: "Complete - with failure",
			cmd: &agentv1beta1.Command{
				Spec: agentv1beta1.CommandSpec{
					NodeSelector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							testNodeIP:  testNodeIP,
							testNodeIP2: testNodeIP2,
						},
					},
				},
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeIP: {
						Phase:  agentv1beta1.CommandComplete,
						Status: metav1.ConditionTrue,
					},
					testNodeIP2: {
						Phase:  agentv1beta1.CommandFailed,
						Status: metav1.ConditionFalse,
					},
				},
			},
			expectComplete: true,
		},
		{
			name: "Running command",
			cmd: &agentv1beta1.Command{
				Status: map[string]*agentv1beta1.CommandStatus{
					testNodeIP: {
						Phase: agentv1beta1.CommandRunning,
					},
				},
			},
			expectComplete: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			complete, _, _ := CheckCommandStatus(tt.cmd)
			assert.Equal(t, tt.expectComplete, complete)
		})
	}
}

func TestIsOwnerRefCommand(t *testing.T) {
	tests := []struct {
		name   string
		object *metav1.ObjectMeta
		cmd    *agentv1beta1.Command
		expect bool
	}{
		{
			name: "Is owner reference",
			object: &metav1.ObjectMeta{
				Name: "owner",
				UID:  "test-uid",
			},
			cmd: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{
						{
							Name: "owner",
							UID:  "test-uid",
						},
					},
				},
			},
			expect: true,
		},
		{
			name: "Not owner reference - different UID",
			object: &metav1.ObjectMeta{
				Name: "owner",
				UID:  "test-uid",
			},
			cmd: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{
						{
							Name: "owner",
							UID:  "different-uid",
						},
					},
				},
			},
			expect: false,
		},
		{
			name: "No owner references",
			object: &metav1.ObjectMeta{
				Name: "owner",
				UID:  "test-uid",
			},
			cmd: &agentv1beta1.Command{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{},
				},
			},
			expect: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsOwnerRefCommand(tt.object, *tt.cmd)
			assert.Equal(t, tt.expect, result)
		})
	}
}

func TestGetNodeSelector(t *testing.T) {
	nodes := bkenode.Nodes{
		{IP: testNodeIP, Hostname: testHost},
		{IP: testNodeIP2, Hostname: "node2"},
	}

	selector := getNodeSelector(nodes)

	assert.NotNil(t, selector)
	assert.NotNil(t, selector.MatchLabels)
	assert.Equal(t, testNodeIP, selector.MatchLabels[testNodeIP])
	assert.Equal(t, testNodeIP2, selector.MatchLabels[testNodeIP2])
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "bootstrap-", BootstrapCommandNamePrefix)
	assert.Equal(t, "k8s-ha-deploy", HACommandName)
	assert.Equal(t, "k8s-env-init", K8sEnvCommandName)
	assert.Equal(t, "switch-cluster-", SwitchClusterCommandNamePrefix)
	assert.Equal(t, "reset-node-", ResetNodeCommandNamePrefix)
	assert.Equal(t, "upgrade-node-", UpgradeNodeCommandNamePrefix)
	assert.Equal(t, "ping-", PingCommandNamePrefix)
	assert.Equal(t, "collect-", CollectCertCommandNamePrefix)
	assert.Equal(t, "bke.bocloud.com/cluster-command", BKEClusterLabel)
	assert.Equal(t, "bke.bocloud.com/machine-command", BKEMachineLabel)
	assert.Equal(t, 3, DefaultBackoffLimit)
	assert.Equal(t, 1000, DefaultActiveDeadlineSecond)
	assert.Equal(t, 600, DefaultTTLSecondsAfterFinished)
	assert.Equal(t, cmdTimeout, DefaultWaitTimeout)
	assert.Equal(t, cmdInterval, DefaultWaitInterval)
}

func TestBootstrapConstants(t *testing.T) {
	assert.Equal(t, "bke.bocloud.com/master-init-command", MasterInitCommandLabel)
	assert.Equal(t, "bke.bocloud.com/master-join-command", MasterJoinCommandLabel)
	assert.Equal(t, "bke.bocloud.com/worker-join-command", WorkerJoinCommandLabel)
}

func TestTimeoutCaseResult(t *testing.T) {
	result := TimeoutCaseResult{
		Err:          nil,
		Complete:     true,
		SuccessNodes: []string{testNodeIP},
		FailedNodes:  []string{},
	}

	assert.True(t, result.Complete)
	assert.Nil(t, result.Err)
	assert.Equal(t, []string{testNodeIP}, result.SuccessNodes)
}

func TestCommandNodes(t *testing.T) {
	nodes := CommandNodes{
		SuccessNodes: []string{testNodeIP},
		FailedNodes:  []string{testNodeIP2},
	}

	assert.Equal(t, numC1, len(nodes.SuccessNodes))
	assert.Equal(t, numC1, len(nodes.FailedNodes))
}

func TestWaitCommandResult(t *testing.T) {
	result := WaitCommandResult{
		Err:          nil,
		Complete:     true,
		SuccessNodes: []string{testNodeIP},
		FailedNodes:  []string{},
	}

	assert.True(t, result.Complete)
	assert.Nil(t, result.Err)
}

func TestGenerateDefaultCommandSpecFields(t *testing.T) {
	spec := GenerateDefaultCommandSpec()

	assert.NotNil(t, spec)
	assert.Empty(t, spec.NodeName)
	assert.False(t, spec.Suspend)
	assert.Empty(t, spec.Commands)
	assert.Equal(t, DefaultBackoffLimit, spec.BackoffLimit)
	assert.Equal(t, 0, spec.ActiveDeadlineSecond)
	assert.Equal(t, DefaultTTLSecondsAfterFinished, spec.TTLSecondsAfterFinished)
	assert.NotNil(t, spec.NodeSelector)
}

func TestBaseCommandValidate(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *BaseCommand
		wantErr bool
	}{
		{
			name: "Valid base command",
			cmd: &BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: false,
		},
		{
			name: "Nil client",
			cmd: &BaseCommand{
				Client:    nil,
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
		{
			name: "Nil scheme",
			cmd: &BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    nil,
			},
			wantErr: true,
		},
		{
			name: "Empty namespace",
			cmd: &BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: "",
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cmd.validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateBkeCommand(t *testing.T) {
	tests := []struct {
		name    string
		nodes   bkenode.Nodes
		config  string
		cmd     *BaseCommand
		wantErr bool
	}{
		{
			name:   "Valid command",
			nodes:  bkenode.Nodes{{IP: testNodeIP, Hostname: testHost}},
			config: testBKEConfig,
			cmd: &BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: false,
		},
		{
			name:   "Empty nodes",
			nodes:  bkenode.Nodes{},
			config: testBKEConfig,
			cmd: &BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
		{
			name:   "Empty config",
			nodes:  bkenode.Nodes{{IP: testNodeIP, Hostname: testHost}},
			config: "",
			cmd: &BaseCommand{
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
		{
			name:   "Nil client",
			nodes:  bkenode.Nodes{{IP: testNodeIP, Hostname: testHost}},
			config: testBKEConfig,
			cmd: &BaseCommand{
				Client:    nil,
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateBkeCommand(tt.nodes, tt.config, tt.cmd)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBaseCommandSetCommandName(t *testing.T) {
	cmd := &BaseCommand{}
	cmd.setCommandName(testCmdName)
	assert.Equal(t, testCmdName, cmd.commandName)
}

func TestBaseCommandGetCommand(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *BaseCommand
		wantErr bool
	}{
		{
			name: "Empty command name",
			cmd: &BaseCommand{
				Ctx:         context.Background(),
				Client:      fake.NewClientBuilder().Build(),
				NameSpace:   testNS,
				Scheme:      runtime.NewScheme(),
				commandName: "",
			},
			wantErr: true,
		},
		{
			name: "Command not found",
			cmd: &BaseCommand{
				Ctx:         context.Background(),
				Client:      fake.NewClientBuilder().Build(),
				NameSpace:   testNS,
				Scheme:      runtime.NewScheme(),
				commandName: "non-existent-command",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			command, err := tt.cmd.GetCommand()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, command)
			}
		})
	}
}

func TestBaseCommandDeleteCommand(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *BaseCommand
		wantErr bool
	}{
		{
			name: "Nil command",
			cmd: &BaseCommand{
				Ctx:       context.Background(),
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
				Command:   nil,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cmd.deleteCommand()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBaseCommandWaitCommandComplete(t *testing.T) {
	cmd := &BaseCommand{
		Ctx:         context.Background(),
		Client:      fake.NewClientBuilder().Build(),
		NameSpace:   testNS,
		Scheme:      runtime.NewScheme(),
		commandName: "",
	}

	err, complete, nodes := cmd.waitCommandComplete()
	assert.Error(t, err)
	assert.False(t, complete)
	assert.Nil(t, nodes.SuccessNodes)
	assert.Nil(t, nodes.FailedNodes)
}

func TestBaseCommandWaitCommandCompleteWithStruct(t *testing.T) {
	cmd := &BaseCommand{
		Ctx:         context.Background(),
		Client:      fake.NewClientBuilder().Build(),
		NameSpace:   testNS,
		Scheme:      runtime.NewScheme(),
		commandName: "",
	}

	result := cmd.waitCommandCompleteWithStruct()
	assert.Error(t, result.Err)
	assert.False(t, result.Complete)
	assert.Nil(t, result.SuccessNodes)
	assert.Nil(t, result.FailedNodes)
}

func TestBaseCommandHandleTimeoutCase(t *testing.T) {
	cmd := &BaseCommand{
		Ctx:         context.Background(),
		Client:      fake.NewClientBuilder().Build(),
		NameSpace:   testNS,
		Scheme:      runtime.NewScheme(),
		commandName: "non-existent-command",
	}

	result := cmd.handleTimeoutCase(false, []string{}, []string{})
	assert.Error(t, result.Err)
	assert.False(t, result.Complete)
}

func TestHandleTimeoutCase_FormatMismatch(t *testing.T) {
	command := &agentv1beta1.Command{}
	command.SetName("test-cmd")
	command.SetNamespace(testNS)
	command.Spec.NodeSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{
			testNodeIP:  testNodeIP,
			testNodeIP2: testNodeIP2,
		},
	}

	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(command).Build()

	cmd := &BaseCommand{
		Ctx:         context.Background(),
		Client:      fakeClient,
		NameSpace:   testNS,
		Scheme:      scheme,
		commandName: "test-cmd",
	}

	successNodes := []string{"master-host/" + testNodeIP}
	failedNodes := []string{}

	result := cmd.handleTimeoutCase(false, successNodes, failedNodes)
	assert.NoError(t, result.Err)
	assert.False(t, result.Complete)
	assert.Contains(t, result.SuccessNodes, "master-host/"+testNodeIP)
	assert.Contains(t, result.FailedNodes, testNodeIP2)
	assert.NotContains(t, result.FailedNodes, testNodeIP)
}

func TestGetNodeIPFromCommandWaitResult(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "hostname/IP format", input: "master-node/192.168.1.1", want: "192.168.1.1"},
		{name: "bare IP", input: "192.168.1.1", want: "192.168.1.1"},
		{name: "empty string", input: "", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := utils.GetNodeIPFromCommandWaitResult(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestClusterNameLabelSelectorRequirement(t *testing.T) {
	cmd := &BaseCommand{
		ClusterName: testClName,
	}

	requirement := cmd.ClusterNameLabelSelectorRequirement()
	assert.Equal(t, utils.ClusterNameLabelKey, requirement.Key)
	assert.Equal(t, metav1.LabelSelectorOpIn, requirement.Operator)
	assert.Contains(t, requirement.Values, testClName)
}

func TestBaseCommandBuildLabels(t *testing.T) {
	cmd := &BaseCommand{
		ClusterName: testClName,
	}

	labels := cmd.buildLabels("test-label", []string{"custom-label"})
	assert.NotNil(t, labels)
	assert.Contains(t, labels, "test-label")
	assert.Contains(t, labels, clusterv1.ClusterNameLabel)
	assert.Contains(t, labels, "custom-label")
}

func TestBaseCommandBuildLabelsWithoutClusterName(t *testing.T) {
	cmd := &BaseCommand{
		ClusterName: "",
	}

	labels := cmd.buildLabels("test-label", []string{})
	assert.NotNil(t, labels)
	assert.Contains(t, labels, "test-label")
	assert.NotContains(t, labels, clusterv1.ClusterNameLabel)
}

func TestBaseCommandSetOwnerReference(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *BaseCommand
		wantErr bool
	}{
		{
			name: "Nil owner object",
			cmd: &BaseCommand{
				Scheme: runtime.NewScheme(),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			command := &agentv1beta1.Command{}
			err := tt.cmd.setOwnerReference(command)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUniqueCommandBase_IPPrefixBoundary(t *testing.T) {
	assert.Equal(t, "reset-node-122.235.189.1", uniqueCommandBase("reset-node-122.235.189.1-1784721347"))
	assert.Equal(t, "reset-node-122.235.189.16", uniqueCommandBase("reset-node-122.235.189.16-1784721348"))
	assert.Equal(t, "shutdown-bkeagent-122.235.189.16", uniqueCommandBase("shutdown-bkeagent-122.235.189.16"))
	assert.False(t, isSameUniqueCommandFamily(
		"reset-node-122.235.189.1-1784721347",
		"reset-node-122.235.189.16-1784721348",
	))
	assert.False(t, isSameUniqueCommandFamily(
		"shutdown-bkeagent-122.235.189.1",
		"shutdown-bkeagent-122.235.189.16",
	))
	assert.True(t, isSameUniqueCommandFamily(
		"reset-node-122.235.189.16-1784721347",
		"reset-node-122.235.189.16-1784721348",
	))
}

func TestBaseCommandHandleUniqueCommand(t *testing.T) {
	tests := []struct {
		name    string
		cmd     *BaseCommand
		wantErr bool
	}{
		{
			name: "Not unique command",
			cmd: &BaseCommand{
				Ctx:       context.Background(),
				Client:    fake.NewClientBuilder().Build(),
				NameSpace: testNS,
				Scheme:    runtime.NewScheme(),
				Unique:    false,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cmd.handleUniqueCommand("test-command")
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestBaseCommandWaitCommandCompleteWithDefaults(t *testing.T) {
	cmd := &BaseCommand{
		Ctx:          context.Background(),
		Client:       fake.NewClientBuilder().Build(),
		NameSpace:    testNS,
		Scheme:       runtime.NewScheme(),
		commandName:  "",
		WaitTimeout:  0,
		WaitInterval: 0,
	}

	result := cmd.waitCommandCompleteWithStruct()
	assert.Error(t, result.Err)
	assert.Equal(t, time.Duration(0), cmd.WaitInterval)
	assert.Equal(t, time.Duration(0), cmd.WaitTimeout)
}

func makeUniqueTestCommand(name string, withFinalizer bool, terminating bool) *agentv1beta1.Command {
	command := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: testNS,
		},
		Spec: agentv1beta1.CommandSpec{
			NodeSelector: &metav1.LabelSelector{},
		},
	}
	if withFinalizer {
		command.Finalizers = []string{agentv1beta1.CommandFinalizer}
	}
	if terminating {
		command.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	}
	return command
}

func TestHandleUniqueCommand_PurgesAllFamilyCommands(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	familyStale1 := makeUniqueTestCommand("upgrade-node-192.168.201.94-1789870401", true, false)
	familyStale2 := makeUniqueTestCommand("upgrade-node-192.168.201.94-1789871149", true, false)
	otherFamily := makeUniqueTestCommand("ping--1789738516", true, false)
	noTimestamp := makeUniqueTestCommand("save-file-downloader-script", true, false)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(familyStale1, familyStale2, otherFamily, noTimestamp).Build()

	cmd := &BaseCommand{
		Ctx:       context.Background(),
		Client:    fakeClient,
		NameSpace: testNS,
		Scheme:    scheme,
		Unique:    true,
	}

	assert.NoError(t, cmd.handleUniqueCommand("upgrade-node-192.168.201.94-1789871605"))

	// 同族全部消失（finalizer 被摘除、对象真正删除）
	for _, name := range []string{"upgrade-node-192.168.201.94-1789870401", "upgrade-node-192.168.201.94-1789871149"} {
		got := &agentv1beta1.Command{}
		err := fakeClient.Get(context.Background(), client.ObjectKey{Namespace: testNS, Name: name}, got)
		assert.True(t, apierrors.IsNotFound(err), "command %s should be purged", name)
	}
	// 异族保留
	assert.NoError(t, fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "ping--1789738516"}, &agentv1beta1.Command{}))
	assert.NoError(t, fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "save-file-downloader-script"}, &agentv1beta1.Command{}))
}

func TestHandleUniqueCommand_SkipsTerminatingZombies(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	// 僵尸：已 terminating 且 finalizer 未摘——不应占用清理轮次
	zombie := makeUniqueTestCommand("upgrade-node-192.168.201.94-1789738975", true, true)
	// 存活同族：应被删除并摘除 finalizer
	alive := makeUniqueTestCommand("upgrade-node-192.168.201.94-1789871605", true, false)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(zombie, alive).Build()

	cmd := &BaseCommand{
		Ctx:       context.Background(),
		Client:    fakeClient,
		NameSpace: testNS,
		Scheme:    scheme,
		Unique:    true,
	}

	assert.NoError(t, cmd.handleUniqueCommand("upgrade-node-192.168.201.94-1789874955"))

	// 僵尸保持 terminating 原状
	gotZombie := &agentv1beta1.Command{}
	assert.NoError(t, fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "upgrade-node-192.168.201.94-1789738975"}, gotZombie))
	assert.False(t, gotZombie.DeletionTimestamp.IsZero())
	assert.True(t, controllerutil.ContainsFinalizer(gotZombie, agentv1beta1.CommandFinalizer))

	// 存活同族被清掉
	gotAlive := &agentv1beta1.Command{}
	err := fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "upgrade-node-192.168.201.94-1789871605"}, gotAlive)
	assert.True(t, apierrors.IsNotFound(err), "alive family command should be purged")
}

func TestHandleUniqueCommand_UniqueDisabledKeepsCommands(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	stale := makeUniqueTestCommand("upgrade-node-192.168.201.94-1789870401", true, false)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(stale).Build()

	cmd := &BaseCommand{
		Ctx:       context.Background(),
		Client:    fakeClient,
		NameSpace: testNS,
		Scheme:    scheme,
		Unique:    false,
	}

	assert.NoError(t, cmd.handleUniqueCommand("upgrade-node-192.168.201.94-1789871605"))
	assert.NoError(t, fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "upgrade-node-192.168.201.94-1789870401"}, &agentv1beta1.Command{}))
}

func TestPurgeCommand(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	t.Run("removes finalizer so the object really disappears", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(makeUniqueTestCommand("purge-with-finalizer", true, false)).Build()

		assert.NoError(t, PurgeCommand(context.Background(), fakeClient,
			makeUniqueTestCommand("purge-with-finalizer", true, false)))
		err := fakeClient.Get(context.Background(),
			client.ObjectKey{Namespace: testNS, Name: "purge-with-finalizer"}, &agentv1beta1.Command{})
		assert.True(t, apierrors.IsNotFound(err), "command with finalizer should be purged")
	})

	t.Run("removes command without finalizer", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).
			WithObjects(makeUniqueTestCommand("purge-no-finalizer", false, false)).Build()

		assert.NoError(t, PurgeCommand(context.Background(), fakeClient,
			makeUniqueTestCommand("purge-no-finalizer", false, false)))
		err := fakeClient.Get(context.Background(),
			client.ObjectKey{Namespace: testNS, Name: "purge-no-finalizer"}, &agentv1beta1.Command{})
		assert.True(t, apierrors.IsNotFound(err))
	})

	t.Run("ignores missing command", func(t *testing.T) {
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

		assert.NoError(t, PurgeCommand(context.Background(), fakeClient,
			makeUniqueTestCommand("not-exists", true, false)))
	})
}

func TestDeleteCommand_StripsFinalizerByDefault(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	command := makeUniqueTestCommand("ping--1789738516", true, false)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(command).Build()

	cmd := &BaseCommand{
		Ctx:       context.Background(),
		Client:    fakeClient,
		NameSpace: testNS,
		Scheme:    scheme,
		Command:   makeUniqueTestCommand("ping--1789738516", true, false),
	}

	// RemoveAfterWait 路径调用 deleteCommand 时从未设置过 ForceRemove，
	// 此前该场景必然产出 terminating 僵尸；现在默认摘除 finalizer。
	assert.NoError(t, cmd.deleteCommand())
	err := fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "ping--1789738516"}, &agentv1beta1.Command{})
	assert.True(t, apierrors.IsNotFound(err), "command should be purged instead of stuck terminating")
}

func TestApplyActiveDeadlineLinkage(t *testing.T) {
	t.Run("unset links to WaitTimeout minus skew", func(t *testing.T) {
		b := &BaseCommand{WaitTimeout: 5 * time.Minute}
		spec := &agentv1beta1.CommandSpec{ActiveDeadlineSecond: 0}
		b.applyActiveDeadlineLinkage(spec)
		assert.Equal(t, int((5*time.Minute-ActiveDeadlineWaitSkew).Seconds()), spec.ActiveDeadlineSecond)
	})
	t.Run("explicit value respected", func(t *testing.T) {
		b := &BaseCommand{WaitTimeout: 5 * time.Minute}
		spec := &agentv1beta1.CommandSpec{ActiveDeadlineSecond: 900}
		b.applyActiveDeadlineLinkage(spec)
		assert.Equal(t, 900, spec.ActiveDeadlineSecond)
	})
	t.Run("buildCommandObject applies linkage", func(t *testing.T) {
		scheme := runtime.NewScheme()
		_ = agentv1beta1.AddToScheme(scheme)
		b := &BaseCommand{
			NameSpace:   testNS,
			WaitTimeout: DefaultWaitTimeout,
			Scheme:      scheme,
		}
		cmd := b.buildCommandObject("link-deadline", BKEClusterLabel, GenerateDefaultCommandSpec(), nil)
		assert.Equal(t, int((DefaultWaitTimeout-ActiveDeadlineWaitSkew).Seconds()), cmd.Spec.ActiveDeadlineSecond)
		assert.Contains(t, cmd.Finalizers, agentv1beta1.CommandFinalizer)
	})
}

func TestWaitTimeoutPurgesImmediately(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = agentv1beta1.AddToScheme(scheme)

	cmdObj := &agentv1beta1.Command{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "wait-timeout-cmd",
			Namespace:  testNS,
			UID:        "uid-wait-timeout",
			Finalizers: []string{agentv1beta1.CommandFinalizer},
		},
		Spec: agentv1beta1.CommandSpec{
			NodeSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{testNodeIP: ""},
			},
			ActiveDeadlineSecond: 240,
		},
		Status: map[string]*agentv1beta1.CommandStatus{},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cmdObj).Build()
	recorder := record.NewFakeRecorder(4)

	owner := &clusterv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{Name: "owner-cluster", Namespace: testNS},
	}

	b := &BaseCommand{
		Ctx:          context.Background(),
		Client:       fakeClient,
		NameSpace:    testNS,
		Scheme:       scheme,
		commandName:  "wait-timeout-cmd",
		WaitTimeout:  50 * time.Millisecond,
		WaitInterval: 10 * time.Millisecond,
		Recorder:     recorder,
		OwnerObj:     owner,
		Command:      cmdObj,
	}

	result := b.waitCommandCompleteWithStruct()
	assert.Equal(t, ReasonWaitTimeout, result.Reason)
	assert.Error(t, result.Err)
	assert.Contains(t, result.Err.Error(), ReasonWaitTimeout)

	err := fakeClient.Get(context.Background(),
		client.ObjectKey{Namespace: testNS, Name: "wait-timeout-cmd"}, &agentv1beta1.Command{})
	assert.True(t, apierrors.IsNotFound(err), "WaitTimeout must Purge immediately")

	select {
	case evt := <-recorder.Events:
		assert.Contains(t, evt, EventCommandWaitTimeout)
	default:
		t.Fatal("expected CommandWaitTimeout event on owner")
	}
}
