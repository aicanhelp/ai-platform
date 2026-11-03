/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package v1beta1

import (
	"encoding/json"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCommandTypeValues(t *testing.T) {
	tests := []struct {
		name     string
		cmdType  CommandType
		expected string
	}{
		{
			name:     "CommandBuiltIn",
			cmdType:  CommandBuiltIn,
			expected: "BuiltIn",
		},
		{
			name:     "CommandShell",
			cmdType:  CommandShell,
			expected: "Shell",
		},
		{
			name:     "CommandKubernetes",
			cmdType:  CommandKubernetes,
			expected: "Kubernetes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.cmdType) != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, string(tt.cmdType))
			}
		})
	}
}

func TestCommandPhaseValues(t *testing.T) {
	tests := []struct {
		name    string
		phase   CommandPhase
		isValid bool
	}{
		{
			name:    "CommandPending",
			phase:   CommandPending,
			isValid: true,
		},
		{
			name:    "CommandRunning",
			phase:   CommandRunning,
			isValid: true,
		},
		{
			name:    "CommandComplete",
			phase:   CommandComplete,
			isValid: true,
		},
		{
			name:    "CommandSuspend",
			phase:   CommandSuspend,
			isValid: true,
		},
		{
			name:    "CommandSkip",
			phase:   CommandSkip,
			isValid: true,
		},
		{
			name:    "CommandFailed",
			phase:   CommandFailed,
			isValid: true,
		},
		{
			name:    "CommandUnKnown",
			phase:   CommandUnKnown,
			isValid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.isValid && tt.phase == "" {
				t.Error("phase should not be empty")
			}
		})
	}
}

func TestDefaultActiveDeadlineSecond(t *testing.T) {
	expectedValue := 600

	if DefaultActiveDeadlineSecond != expectedValue {
		t.Errorf("expected DefaultActiveDeadlineSecond %d, got %d", expectedValue, DefaultActiveDeadlineSecond)
	}
}

func TestCommandSpecJSONMarshaling(t *testing.T) {
	spec := CommandSpec{
		NodeName: "test-node",
		Suspend:  false,
		Commands: []ExecCommand{
			{
				ID:            "cmd1",
				Command:       []string{"echo", "hello"},
				Type:          CommandShell,
				BackoffIgnore: false,
				BackoffDelay:  0,
			},
		},
		BackoffLimit:            numThree,
		ActiveDeadlineSecond:    DefaultActiveDeadlineSecond,
		TTLSecondsAfterFinished: numTwenty,
	}

	data, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("failed to marshal CommandSpec: %v", err)
	}

	var unmarshaled CommandSpec
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal CommandSpec: %v", err)
	}

	if unmarshaled.NodeName != spec.NodeName {
		t.Errorf("expected NodeName %s, got %s", spec.NodeName, unmarshaled.NodeName)
	}
	if unmarshaled.Suspend != spec.Suspend {
		t.Errorf("expected Suspend %v, got %v", spec.Suspend, unmarshaled.Suspend)
	}
	if len(unmarshaled.Commands) != len(spec.Commands) {
		t.Errorf("expected %d commands, got %d", len(spec.Commands), len(unmarshaled.Commands))
	}
}

func TestExecCommandJSONMarshaling(t *testing.T) {
	execCmd := ExecCommand{
		ID:            "test-id",
		Command:       []string{"ls", "-la"},
		Type:          CommandShell,
		BackoffIgnore: true,
		BackoffDelay:  numTen,
	}

	data, err := json.Marshal(execCmd)
	if err != nil {
		t.Fatalf("failed to marshal ExecCommand: %v", err)
	}

	var unmarshaled ExecCommand
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal ExecCommand: %v", err)
	}

	if unmarshaled.ID != execCmd.ID {
		t.Errorf("expected ID %s, got %s", execCmd.ID, unmarshaled.ID)
	}
	if unmarshaled.Type != execCmd.Type {
		t.Errorf("expected Type %s, got %s", execCmd.Type, unmarshaled.Type)
	}
	if len(unmarshaled.Command) != len(execCmd.Command) {
		t.Errorf("expected %d command elements, got %d", len(execCmd.Command), len(unmarshaled.Command))
	}
	if unmarshaled.BackoffIgnore != execCmd.BackoffIgnore {
		t.Errorf("expected BackoffIgnore %v, got %v", execCmd.BackoffIgnore, unmarshaled.BackoffIgnore)
	}
}

func TestConditionJSONMarshaling(t *testing.T) {
	now := metav1.Now()
	condition := Condition{
		ID:            "cond-1",
		Status:        metav1.ConditionTrue,
		Phase:         CommandComplete,
		LastStartTime: &now,
		StdOut:        []string{"output1", "output2"},
		StdErr:        []string{},
		Count:         numOne,
	}

	data, err := json.Marshal(condition)
	if err != nil {
		t.Fatalf("failed to marshal Condition: %v", err)
	}

	var unmarshaled Condition
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal Condition: %v", err)
	}

	if unmarshaled.ID != condition.ID {
		t.Errorf("expected ID %s, got %s", condition.ID, unmarshaled.ID)
	}
	if unmarshaled.Status != condition.Status {
		t.Errorf("expected Status %v, got %v", condition.Status, unmarshaled.Status)
	}
	if unmarshaled.Phase != condition.Phase {
		t.Errorf("expected Phase %s, got %s", condition.Phase, unmarshaled.Phase)
	}
	if unmarshaled.Count != condition.Count {
		t.Errorf("expected Count %d, got %d", condition.Count, unmarshaled.Count)
	}
}

func TestCommandStatusJSONMarshaling(t *testing.T) {
	now := metav1.Now()
	completionTime := metav1.Now()

	status := CommandStatus{
		Conditions: []*Condition{
			{
				ID:            "cond-1",
				Status:        metav1.ConditionTrue,
				Phase:         CommandComplete,
				LastStartTime: &now,
				Count:         numOne,
			},
		},
		LastStartTime:  &now,
		CompletionTime: &completionTime,
		Succeeded:      numFive,
		Failed:         numOne,
		Phase:          CommandComplete,
		Status:         metav1.ConditionTrue,
	}

	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("failed to marshal CommandStatus: %v", err)
	}

	var unmarshaled CommandStatus
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal CommandStatus: %v", err)
	}

	if len(unmarshaled.Conditions) != len(status.Conditions) {
		t.Errorf("expected %d conditions, got %d", len(status.Conditions), len(unmarshaled.Conditions))
	}
	if unmarshaled.Succeeded != status.Succeeded {
		t.Errorf("expected Succeeded %d, got %d", status.Succeeded, unmarshaled.Succeeded)
	}
	if unmarshaled.Failed != status.Failed {
		t.Errorf("expected Failed %d, got %d", status.Failed, unmarshaled.Failed)
	}
	if unmarshaled.Phase != status.Phase {
		t.Errorf("expected Phase %s, got %s", status.Phase, unmarshaled.Phase)
	}
}

func TestCommandListJSONMarshaling(t *testing.T) {
	list := CommandList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bkeagent.bocloud.com/v1beta1",
			Kind:       "CommandList",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "12345",
		},
		Items: []Command{
			{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "bkeagent.bocloud.com/v1beta1",
					Kind:       "Command",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "cmd-1",
					Namespace: "default",
				},
				Spec: CommandSpec{
					NodeName: "node-1",
				},
			},
			{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "bkeagent.bocloud.com/v1beta1",
					Kind:       "Command",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "cmd-2",
					Namespace: "default",
				},
				Spec: CommandSpec{
					NodeName: "node-2",
				},
			},
		},
	}

	data, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("failed to marshal CommandList: %v", err)
	}

	var unmarshaled CommandList
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("failed to unmarshal CommandList: %v", err)
	}

	if len(unmarshaled.Items) != len(list.Items) {
		t.Errorf("expected %d items, got %d", len(list.Items), len(unmarshaled.Items))
	}
}

func TestNodeSelectorInCommandSpec(t *testing.T) {
	spec := CommandSpec{
		NodeSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"node-role.kubernetes.io/master": "",
			},
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      "kubernetes.io/os",
					Operator: metav1.LabelSelectorOpIn,
					Values:   []string{"linux"},
				},
			},
		},
	}

	if spec.NodeSelector == nil {
		t.Error("NodeSelector should not be nil")
	}

	if len(spec.NodeSelector.MatchLabels) != numOne {
		t.Errorf("expected 1 match label, got %d", len(spec.NodeSelector.MatchLabels))
	}

	if len(spec.NodeSelector.MatchExpressions) != numOne {
		t.Errorf("expected 1 match expression, got %d", len(spec.NodeSelector.MatchExpressions))
	}
}

func TestEmptyCommandSpec(t *testing.T) {
	spec := CommandSpec{}

	if spec.NodeName != "" {
		t.Errorf("expected empty NodeName, got %s", spec.NodeName)
	}
	if spec.Suspend != false {
		t.Errorf("expected Suspend to be false, got %v", spec.Suspend)
	}
	if spec.Commands != nil {
		t.Error("expected Commands to be nil")
	}
	if spec.BackoffLimit != numZero {
		t.Errorf("expected BackoffLimit to be 0, got %d", spec.BackoffLimit)
	}
	if spec.ActiveDeadlineSecond != numZero {
		t.Errorf("expected ActiveDeadlineSecond to be 0, got %d", spec.ActiveDeadlineSecond)
	}
}

func TestMultipleExecCommands(t *testing.T) {
	spec := CommandSpec{
		Commands: []ExecCommand{
			{
				ID:      "cmd1",
				Command: []string{"echo", "1"},
				Type:    CommandShell,
			},
			{
				ID:      "cmd2",
				Command: []string{"iptables", "--list"},
				Type:    CommandShell,
			},
			{
				ID:      "cmd3",
				Command: []string{"configmap:ns/name:ro:/tmp/file"},
				Type:    CommandKubernetes,
			},
		},
	}

	if len(spec.Commands) != numThree {
		t.Errorf("expected 3 commands, got %d", len(spec.Commands))
	}

	for i, cmd := range spec.Commands {
		if cmd.ID == "" {
			t.Errorf("command at index %d has empty ID", i)
		}
		if len(cmd.Command) == numZero {
			t.Errorf("command at index %d has empty command array", i)
		}
	}
}

func TestConditionDeepCopy(t *testing.T) {
	now := metav1.Now()
	condition := &Condition{
		ID:            "test-condition",
		Status:        metav1.ConditionTrue,
		Phase:         CommandComplete,
		LastStartTime: &now,
		StdOut:        []string{"output1", "output2"},
		StdErr:        []string{"error1"},
		Count:         numFive,
	}

	deepCopy := condition.DeepCopy()

	if deepCopy.ID != condition.ID {
		t.Errorf("expected ID %s, got %s", condition.ID, deepCopy.ID)
	}
	if deepCopy.Status != condition.Status {
		t.Errorf("expected Status %v, got %v", condition.Status, deepCopy.Status)
	}
	if deepCopy.Phase != condition.Phase {
		t.Errorf("expected Phase %s, got %s", condition.Phase, deepCopy.Phase)
	}
	if deepCopy.Count != condition.Count {
		t.Errorf("expected Count %d, got %d", condition.Count, deepCopy.Count)
	}
	if len(deepCopy.StdOut) != len(condition.StdOut) {
		t.Errorf("expected %d stdOut entries, got %d", len(condition.StdOut), len(deepCopy.StdOut))
	}

	deepCopy.ID = "modified"
	deepCopy.Count = numTen
	if condition.ID != "test-condition" {
		t.Error("original condition ID should not be modified")
	}
	if condition.Count != numFive {
		t.Error("original condition Count should not be modified")
	}
}

func TestExecCommandDeepCopy(t *testing.T) {
	execCmd := &ExecCommand{
		ID:            "test-exec-cmd",
		Command:       []string{"echo", "hello"},
		Type:          CommandShell,
		BackoffIgnore: true,
		BackoffDelay:  numTen,
	}

	deepCopy := execCmd.DeepCopy()

	if deepCopy.ID != execCmd.ID {
		t.Errorf("expected ID %s, got %s", execCmd.ID, deepCopy.ID)
	}
	if deepCopy.Type != execCmd.Type {
		t.Errorf("expected Type %s, got %s", execCmd.Type, deepCopy.Type)
	}
	if len(deepCopy.Command) != len(execCmd.Command) {
		t.Errorf("expected %d command entries, got %d", len(execCmd.Command), len(deepCopy.Command))
	}
	if deepCopy.BackoffIgnore != execCmd.BackoffIgnore {
		t.Errorf("expected BackoffIgnore %v, got %v", execCmd.BackoffIgnore, deepCopy.BackoffIgnore)
	}
	if deepCopy.BackoffDelay != execCmd.BackoffDelay {
		t.Errorf("expected BackoffDelay %d, got %d", execCmd.BackoffDelay, deepCopy.BackoffDelay)
	}

	deepCopy.ID = "modified"
	deepCopy.Command[0] = "modified"
	if execCmd.ID != "test-exec-cmd" {
		t.Error("original ExecCommand ID should not be modified")
	}
	if execCmd.Command[0] != "echo" {
		t.Error("original ExecCommand Command should not be modified")
	}
}

func TestCommandSpecDeepCopy(t *testing.T) {
	spec := &CommandSpec{
		NodeName: "test-node",
		Suspend:  true,
		Commands: []ExecCommand{
			{
				ID:      "cmd1",
				Command: []string{"ls", "-la"},
				Type:    CommandShell,
			},
		},
		BackoffLimit:            numThree,
		ActiveDeadlineSecond:    DefaultActiveDeadlineSecond,
		TTLSecondsAfterFinished: numTwenty,
	}

	deepCopy := &CommandSpec{}
	spec.DeepCopyInto(deepCopy)

	if deepCopy.NodeName != spec.NodeName {
		t.Errorf("expected NodeName %s, got %s", spec.NodeName, deepCopy.NodeName)
	}
	if deepCopy.Suspend != spec.Suspend {
		t.Errorf("expected Suspend %v, got %v", spec.Suspend, deepCopy.Suspend)
	}
	if len(deepCopy.Commands) != len(spec.Commands) {
		t.Errorf("expected %d commands, got %d", len(spec.Commands), len(deepCopy.Commands))
	}
	if deepCopy.BackoffLimit != spec.BackoffLimit {
		t.Errorf("expected BackoffLimit %d, got %d", spec.BackoffLimit, deepCopy.BackoffLimit)
	}
	if deepCopy.ActiveDeadlineSecond != spec.ActiveDeadlineSecond {
		t.Errorf("expected ActiveDeadlineSecond %d, got %d", spec.ActiveDeadlineSecond, deepCopy.ActiveDeadlineSecond)
	}
	if deepCopy.TTLSecondsAfterFinished != spec.TTLSecondsAfterFinished {
		t.Errorf("expected TTLSecondsAfterFinished %d, got %d", spec.TTLSecondsAfterFinished, deepCopy.TTLSecondsAfterFinished)
	}

	deepCopy.NodeName = "modified"
	deepCopy.Commands[0].ID = "modified"
	if spec.NodeName != "test-node" {
		t.Error("original spec NodeName should not be modified")
	}
	if spec.Commands[0].ID != "cmd1" {
		t.Error("original spec Commands should not be modified")
	}
}

func TestCommandStatusDeepCopy(t *testing.T) {
	now := metav1.Now()
	status := &CommandStatus{
		Conditions: []*Condition{
			{
				ID:            "cond1",
				Status:        metav1.ConditionTrue,
				Phase:         CommandComplete,
				LastStartTime: &now,
				Count:         numOne,
			},
		},
		LastStartTime: &now,
		Succeeded:     numTen,
		Failed:        numTwo,
		Phase:         CommandComplete,
		Status:        metav1.ConditionTrue,
	}

	deepCopy := status.DeepCopy()

	if len(deepCopy.Conditions) != len(status.Conditions) {
		t.Errorf("expected %d conditions, got %d", len(status.Conditions), len(deepCopy.Conditions))
	}
	if deepCopy.Succeeded != status.Succeeded {
		t.Errorf("expected Succeeded %d, got %d", status.Succeeded, deepCopy.Succeeded)
	}
	if deepCopy.Failed != status.Failed {
		t.Errorf("expected Failed %d, got %d", status.Failed, deepCopy.Failed)
	}
	if deepCopy.Phase != status.Phase {
		t.Errorf("expected Phase %s, got %s", status.Phase, deepCopy.Phase)
	}

	deepCopy.Succeeded = numTwenty
	deepCopy.Conditions[0].ID = "modified"
	if status.Succeeded != numTen {
		t.Error("original status Succeeded should not be modified")
	}
	if status.Conditions[0].ID != "cond1" {
		t.Error("original status Conditions should not be modified")
	}
}

func TestCommandDeepCopy(t *testing.T) {
	command := &Command{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bkeagent.bocloud.com/v1beta1",
			Kind:       "Command",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-command",
			Namespace: "default",
		},
		Spec: CommandSpec{
			NodeName: "test-node",
		},
		Status: map[string]*CommandStatus{
			"node1": {
				Phase:     CommandComplete,
				Status:    metav1.ConditionTrue,
				Succeeded: numOne,
				Failed:    numZero,
			},
		},
	}

	deepCopy := command.DeepCopy()

	if deepCopy.APIVersion != command.APIVersion {
		t.Errorf("expected APIVersion %s, got %s", command.APIVersion, deepCopy.APIVersion)
	}
	if deepCopy.Kind != command.Kind {
		t.Errorf("expected Kind %s, got %s", command.Kind, deepCopy.Kind)
	}
	if deepCopy.Name != command.Name {
		t.Errorf("expected Name %s, got %s", command.Name, deepCopy.Name)
	}
	if deepCopy.Namespace != command.Namespace {
		t.Errorf("expected Namespace %s, got %s", command.Namespace, deepCopy.Namespace)
	}
	if deepCopy.Spec.NodeName != command.Spec.NodeName {
		t.Errorf("expected Spec.NodeName %s, got %s", command.Spec.NodeName, deepCopy.Spec.NodeName)
	}
	if len(deepCopy.Status) != len(command.Status) {
		t.Errorf("expected %d status entries, got %d", len(command.Status), len(deepCopy.Status))
	}

	deepCopy.Name = "modified"
	deepCopy.Spec.NodeName = "modified"
	deepCopy.Status["node1"].Succeeded = numTwenty
	if command.Name != "test-command" {
		t.Error("original command Name should not be modified")
	}
	if command.Spec.NodeName != "test-node" {
		t.Error("original command Spec should not be modified")
	}
	if command.Status["node1"].Succeeded != numOne {
		t.Error("original command Status should not be modified")
	}
}

func TestCommandListDeepCopy(t *testing.T) {
	list := &CommandList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bkeagent.bocloud.com/v1beta1",
			Kind:       "CommandList",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "v123",
		},
		Items: []Command{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "cmd1",
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "cmd2",
				},
			},
		},
	}

	deepCopy := list.DeepCopy()

	if deepCopy.APIVersion != list.APIVersion {
		t.Errorf("expected APIVersion %s, got %s", list.APIVersion, deepCopy.APIVersion)
	}
	if deepCopy.Kind != list.Kind {
		t.Errorf("expected Kind %s, got %s", list.Kind, deepCopy.Kind)
	}
	if deepCopy.ResourceVersion != list.ResourceVersion {
		t.Errorf("expected ResourceVersion %s, got %s", list.ResourceVersion, deepCopy.ResourceVersion)
	}
	if len(deepCopy.Items) != len(list.Items) {
		t.Errorf("expected %d items, got %d", len(list.Items), len(deepCopy.Items))
	}

	deepCopy.Kind = "modified"
	deepCopy.Items[0].Name = "modified"
	if list.Kind != "CommandList" {
		t.Error("original list Kind should not be modified")
	}
	if list.Items[0].Name != "cmd1" {
		t.Error("original list Items should not be modified")
	}
}

func TestCommandDeepCopyObjectCoverage(t *testing.T) {
	command := &Command{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bkeagent.bocloud.com/v1beta1",
			Kind:       "Command",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-command",
			Namespace: "default",
		},
		Spec: CommandSpec{
			NodeName: "test-node",
		},
	}

	object := command.DeepCopyObject()

	if object == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	copiedCommand, ok := object.(*Command)
	if !ok {
		t.Error("DeepCopyObject should return *Command")
	}
	if copiedCommand.APIVersion != command.APIVersion {
		t.Errorf("expected APIVersion %s, got %s", command.APIVersion, copiedCommand.APIVersion)
	}
	if copiedCommand.Kind != command.Kind {
		t.Errorf("expected Kind %s, got %s", command.Kind, copiedCommand.Kind)
	}
	if copiedCommand.Name != command.Name {
		t.Errorf("expected Name %s, got %s", command.Name, copiedCommand.Name)
	}
	if copiedCommand.Namespace != command.Namespace {
		t.Errorf("expected Namespace %s, got %s", command.Namespace, copiedCommand.Namespace)
	}
}

func TestCommandListDeepCopyObjectCoverage(t *testing.T) {
	list := &CommandList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bkeagent.bocloud.com/v1beta1",
			Kind:       "CommandList",
		},
		ListMeta: metav1.ListMeta{
			ResourceVersion: "v123",
		},
		Items: []Command{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "cmd1",
				},
			},
		},
	}

	object := list.DeepCopyObject()

	if object == nil {
		t.Error("DeepCopyObject should not return nil")
	}

	copiedList, ok := object.(*CommandList)
	if !ok {
		t.Error("DeepCopyObject should return *CommandList")
	}
	if copiedList.APIVersion != list.APIVersion {
		t.Errorf("expected APIVersion %s, got %s", list.APIVersion, copiedList.APIVersion)
	}
	if copiedList.Kind != list.Kind {
		t.Errorf("expected Kind %s, got %s", list.Kind, copiedList.Kind)
	}
	if copiedList.ResourceVersion != list.ResourceVersion {
		t.Errorf("expected ResourceVersion %s, got %s", list.ResourceVersion, copiedList.ResourceVersion)
	}
	if len(copiedList.Items) != len(list.Items) {
		t.Errorf("expected %d items, got %d", len(list.Items), len(copiedList.Items))
	}
}

func TestCommandSpecDeepCopyCoverage(t *testing.T) {
	spec := &CommandSpec{
		NodeName:             "test-node",
		Suspend:              true,
		Commands:             []ExecCommand{{ID: "cmd1", Command: []string{"echo hello"}, Type: CommandShell}},
		BackoffLimit:         numFive,
		ActiveDeadlineSecond: numSixHundred,
		NodeSelector:         &metav1.LabelSelector{MatchLabels: map[string]string{"app": "test"}},
	}

	deepCopy := spec.DeepCopy()

	if deepCopy == nil {
		t.Error("DeepCopy should not return nil")
	}
	if deepCopy.NodeName != spec.NodeName {
		t.Errorf("expected NodeName %s, got %s", spec.NodeName, deepCopy.NodeName)
	}
	if deepCopy.Suspend != spec.Suspend {
		t.Errorf("expected Suspend %v, got %v", spec.Suspend, deepCopy.Suspend)
	}
	if deepCopy.BackoffLimit != spec.BackoffLimit {
		t.Errorf("expected BackoffLimit %d, got %d", spec.BackoffLimit, deepCopy.BackoffLimit)
	}
	if deepCopy.ActiveDeadlineSecond != spec.ActiveDeadlineSecond {
		t.Errorf("expected ActiveDeadlineSecond %d, got %d", spec.ActiveDeadlineSecond, deepCopy.ActiveDeadlineSecond)
	}
	if len(deepCopy.Commands) != len(spec.Commands) {
		t.Errorf("expected %d commands, got %d", len(spec.Commands), len(deepCopy.Commands))
	}
	if deepCopy.Commands[numZero].ID != spec.Commands[numZero].ID {
		t.Errorf("expected Commands[0].ID %s, got %s", spec.Commands[numZero].ID, deepCopy.Commands[numZero].ID)
	}
	if deepCopy.NodeSelector == nil {
		t.Error("NodeSelector should not be nil")
	}

	deepCopy.NodeName = "modified-node"
	deepCopy.Suspend = false
	deepCopy.Commands[numZero].ID = "modified-cmd"

	if spec.NodeName != "test-node" {
		t.Error("original spec NodeName should not be modified")
	}
	if spec.Suspend != true {
		t.Error("original spec Suspend should not be modified")
	}
	if spec.Commands[numZero].ID != "cmd1" {
		t.Error("original spec Commands should not be modified")
	}
}
