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

package v1beta1

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestConditionCountResultDeepCopyNil(t *testing.T) {
	var result *ConditionCountResult
	deepCopy := result.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestConditionCountResultDeepCopy(t *testing.T) {
	result := &ConditionCountResult{
		Succeeded: numFive,
		Failed:    numTwo,
		Status:    metav1.ConditionTrue,
		Phase:     CommandComplete,
	}
	deepCopy := result.DeepCopy()
	if deepCopy == nil {
		t.Error("DeepCopy should not return nil")
	}
	if deepCopy.Succeeded != result.Succeeded {
		t.Errorf("expected Succeeded %d, got %d", result.Succeeded, deepCopy.Succeeded)
	}
	if deepCopy.Failed != result.Failed {
		t.Errorf("expected Failed %d, got %d", result.Failed, deepCopy.Failed)
	}
	if deepCopy.Status != result.Status {
		t.Errorf("expected Status %v, got %v", result.Status, deepCopy.Status)
	}
	if deepCopy.Phase != result.Phase {
		t.Errorf("expected Phase %s, got %s", result.Phase, deepCopy.Phase)
	}
	deepCopy.Succeeded = numTen
	if result.Succeeded != numFive {
		t.Error("original Succeeded should not be modified")
	}
}

func TestConditionCountResultDeepCopyInto(t *testing.T) {
	result := &ConditionCountResult{
		Succeeded: numThree,
		Failed:    numOne,
		Status:    metav1.ConditionFalse,
		Phase:     CommandFailed,
	}
	out := &ConditionCountResult{}
	result.DeepCopyInto(out)
	if out.Succeeded != result.Succeeded {
		t.Errorf("expected Succeeded %d, got %d", result.Succeeded, out.Succeeded)
	}
	if out.Failed != result.Failed {
		t.Errorf("expected Failed %d, got %d", result.Failed, out.Failed)
	}
	if out.Status != result.Status {
		t.Errorf("expected Status %v, got %v", result.Status, out.Status)
	}
	if out.Phase != result.Phase {
		t.Errorf("expected Phase %s, got %s", result.Phase, out.Phase)
	}
	result.Succeeded = numFive
	if out.Succeeded != numThree {
		t.Error("out Succeeded should not be modified after DeepCopyInto")
	}
}

func TestCommandDeepCopyNil(t *testing.T) {
	var cmd *Command
	deepCopy := cmd.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestCommandDeepCopyIntoNil(t *testing.T) {
	cmd := &Command{}
	into := &Command{}
	cmd.DeepCopyInto(into)
}

func TestCommandListDeepCopyNil(t *testing.T) {
	var list *CommandList
	deepCopy := list.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestCommandSpecDeepCopyNil(t *testing.T) {
	var spec *CommandSpec
	deepCopy := spec.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestCommandStatusDeepCopyNil(t *testing.T) {
	var status *CommandStatus
	deepCopy := status.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestConditionDeepCopyNil(t *testing.T) {
	var condition *Condition
	deepCopy := condition.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestExecCommandDeepCopyNil(t *testing.T) {
	var exec *ExecCommand
	deepCopy := exec.DeepCopy()
	if deepCopy != nil {
		t.Error("DeepCopy on nil should return nil")
	}
}

func TestCommandDeepCopyStatusNil(t *testing.T) {
	cmd := &Command{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-cmd",
		},
		Spec: CommandSpec{
			NodeName: "test-node",
		},
		Status: nil,
	}
	deepCopy := cmd.DeepCopy()
	if deepCopy.Status != nil {
		t.Error("Status should be nil in deep copy")
	}
}

func TestCommandDeepCopyStatusWithNilValue(t *testing.T) {
	cmd := &Command{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-cmd",
		},
		Spec: CommandSpec{
			NodeName: "test-node",
		},
		Status: map[string]*CommandStatus{
			"node1": nil,
			"node2": {
				Phase:  CommandComplete,
				Status: metav1.ConditionTrue,
			},
		},
	}
	deepCopy := cmd.DeepCopy()
	if deepCopy.Status["node1"] != nil {
		t.Error("node1 should be nil in deep copy")
	}
	if deepCopy.Status["node2"] == nil {
		t.Error("node2 should not be nil in deep copy")
	}
	deepCopy.Status["node2"].Phase = CommandFailed
	if cmd.Status["node2"].Phase == CommandFailed {
		t.Error("original status should not be modified")
	}
}

func TestCommandStatusDeepCopyWithTimes(t *testing.T) {
	now := metav1.Now()
	completion := metav1.Now()
	status := &CommandStatus{
		LastStartTime:  &now,
		CompletionTime: &completion,
		Succeeded:      numFive,
		Failed:         numTwo,
		Phase:          CommandComplete,
		Status:         metav1.ConditionTrue,
	}
	deepCopy := status.DeepCopy()
	if deepCopy.LastStartTime == nil {
		t.Error("LastStartTime should not be nil")
	}
	if deepCopy.CompletionTime == nil {
		t.Error("CompletionTime should not be nil")
	}
	deepCopy.Succeeded = numTen
	if status.Succeeded != numFive {
		t.Error("original Succeeded should not be modified")
	}
}

func TestCommandStatusDeepCopyConditionsNilElement(t *testing.T) {
	status := &CommandStatus{
		Conditions: []*Condition{
			nil,
			{
				ID:     "cond1",
				Status: metav1.ConditionTrue,
				Phase:  CommandComplete,
			},
		},
	}
	deepCopy := status.DeepCopy()
	if deepCopy.Conditions[numZero] != nil {
		t.Error("first condition should be nil")
	}
	if deepCopy.Conditions[numOne] == nil {
		t.Error("second condition should not be nil")
	}
	deepCopy.Conditions[numOne].ID = "modified"
	if status.Conditions[numOne].ID == "modified" {
		t.Error("original condition should not be modified")
	}
}

func TestConditionDeepCopyWithStrings(t *testing.T) {
	now := metav1.Now()
	condition := &Condition{
		ID:            "test-condition",
		Status:        metav1.ConditionTrue,
		Phase:         CommandComplete,
		LastStartTime: &now,
		StdOut:        []string{"output1", "output2", "output3"},
		StdErr:        []string{"error1", "error2"},
		Count:         numFive,
	}
	deepCopy := condition.DeepCopy()
	if len(deepCopy.StdOut) != numThree {
		t.Errorf("expected %d stdOut entries, got %d", numThree, len(deepCopy.StdOut))
	}
	if len(deepCopy.StdErr) != numTwo {
		t.Errorf("expected %d stdErr entries, got %d", numTwo, len(deepCopy.StdErr))
	}
	deepCopy.StdOut[numZero] = "modified"
	deepCopy.StdErr[numZero] = "modified"
	if condition.StdOut[numZero] == "modified" {
		t.Error("original StdOut should not be modified")
	}
	if condition.StdErr[numZero] == "modified" {
		t.Error("original StdErr should not be modified")
	}
}

func TestExecCommandDeepCopyInto(t *testing.T) {
	exec := &ExecCommand{
		ID:            "test-exec",
		Command:       []string{"echo", "hello"},
		Type:          CommandShell,
		BackoffIgnore: true,
		BackoffDelay:  numTen,
	}
	into := &ExecCommand{}
	exec.DeepCopyInto(into)
	if into.ID != exec.ID {
		t.Errorf("expected ID %s, got %s", exec.ID, into.ID)
	}
	if len(into.Command) != len(exec.Command) {
		t.Errorf("expected %d commands, got %d", len(exec.Command), len(into.Command))
	}
	into.Command[numZero] = "modified"
	if exec.Command[numZero] == "modified" {
		t.Error("original Command should not be modified")
	}
}

func TestCommandListDeepCopyInto(t *testing.T) {
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
	into := &CommandList{}
	list.DeepCopyInto(into)
	if into.APIVersion != list.APIVersion {
		t.Errorf("expected APIVersion %s, got %s", list.APIVersion, into.APIVersion)
	}
	if len(into.Items) != len(list.Items) {
		t.Errorf("expected %d items, got %d", len(list.Items), len(into.Items))
	}
	into.Items[numZero].Name = "modified"
	if list.Items[numZero].Name == "modified" {
		t.Error("original item should not be modified")
	}
}

func TestCommandSpecDeepCopyWithNodeSelectorNil(t *testing.T) {
	spec := &CommandSpec{
		NodeName:             "test-node",
		Commands:             []ExecCommand{},
		NodeSelector:         nil,
		BackoffLimit:         numFive,
		ActiveDeadlineSecond: numTen,
	}
	deepCopy := spec.DeepCopy()
	if deepCopy.NodeSelector != nil {
		t.Error("NodeSelector should be nil")
	}
}

func TestCommandSpecDeepCopyInto(t *testing.T) {
	spec := &CommandSpec{
		NodeName:                "test-node",
		Commands:                []ExecCommand{{ID: "cmd1", Type: CommandShell}},
		NodeSelector:            &metav1.LabelSelector{MatchLabels: map[string]string{"app": "test"}},
		BackoffLimit:            numThree,
		ActiveDeadlineSecond:    numTen,
		TTLSecondsAfterFinished: numFive,
	}
	into := &CommandSpec{}
	spec.DeepCopyInto(into)
	if into.NodeName != spec.NodeName {
		t.Errorf("expected NodeName %s, got %s", spec.NodeName, into.NodeName)
	}
	if into.NodeSelector == nil {
		t.Error("NodeSelector should not be nil")
	}
	into.NodeSelector.MatchLabels["new"] = "label"
	if spec.NodeSelector.MatchLabels["new"] == "label" {
		t.Error("original NodeSelector should not be modified")
	}
}

func TestCommandDeepCopyObjectNil(t *testing.T) {
	var cmd *Command
	obj := cmd.DeepCopyObject()
	if obj != nil {
		t.Error("DeepCopyObject on nil should return nil")
	}
}

func TestCommandListDeepCopyObjectNil(t *testing.T) {
	var list *CommandList
	obj := list.DeepCopyObject()
	if obj != nil {
		t.Error("DeepCopyObject on nil should return nil")
	}
}
