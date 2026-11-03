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
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	numZero         = 0
	numOne          = 1
	numTwo          = 2
	numThree        = 3
	numFour         = 4
	numFive         = 5
	numSix          = 6
	numSeven        = 7
	numTen          = 10
	numFifteen      = 15
	numTwenty       = 20
	numThreeHundred = 300
	numSixHundred   = 600
	testTimeout     = 100 * time.Millisecond
)

func TestRemoveCondition(t *testing.T) {
	tests := []struct {
		name        string
		conditions  []*Condition
		target      *Condition
		expectedLen int
	}{
		{
			name:        "nilTarget",
			conditions:  []*Condition{{ID: "test1"}},
			target:      nil,
			expectedLen: numOne,
		},
		{
			name:        "targetNotFound",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}},
			target:      &Condition{ID: "test3"},
			expectedLen: numTwo,
		},
		{
			name:        "removeFirst",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}, {ID: "test3"}},
			target:      &Condition{ID: "test1"},
			expectedLen: numTwo,
		},
		{
			name:        "removeMiddle",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}, {ID: "test3"}},
			target:      &Condition{ID: "test2"},
			expectedLen: numTwo,
		},
		{
			name:        "removeLast",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}, {ID: "test3"}},
			target:      &Condition{ID: "test3"},
			expectedLen: numTwo,
		},
		{
			name:        "emptyConditions",
			conditions:  []*Condition{},
			target:      &Condition{ID: "test1"},
			expectedLen: numZero,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RemoveCondition(tt.conditions, tt.target)
			if len(result) != tt.expectedLen {
				t.Errorf("expected length %d, got %d", tt.expectedLen, len(result))
			}
		})
	}
}

func TestGetCondition(t *testing.T) {
	tests := []struct {
		name       string
		conditions []*Condition
		target     *Condition
		found      bool
		expectedID string
	}{
		{
			name:       "nilTarget",
			conditions: []*Condition{{ID: "test1"}},
			target:     nil,
			found:      false,
			expectedID: "",
		},
		{
			name:       "emptyConditions",
			conditions: []*Condition{},
			target:     &Condition{ID: "test1"},
			found:      false,
			expectedID: "",
		},
		{
			name:       "findFirst",
			conditions: []*Condition{{ID: "test1"}, {ID: "test2"}},
			target:     &Condition{ID: "test1"},
			found:      true,
			expectedID: "test1",
		},
		{
			name:       "findMiddle",
			conditions: []*Condition{{ID: "test1"}, {ID: "test2"}, {ID: "test3"}},
			target:     &Condition{ID: "test2"},
			found:      true,
			expectedID: "test2",
		},
		{
			name:       "findLast",
			conditions: []*Condition{{ID: "test1"}, {ID: "test2"}, {ID: "test3"}},
			target:     &Condition{ID: "test3"},
			found:      true,
			expectedID: "test3",
		},
		{
			name:       "notFound",
			conditions: []*Condition{{ID: "test1"}, {ID: "test2"}},
			target:     &Condition{ID: "test3"},
			found:      false,
			expectedID: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetCondition(tt.conditions, tt.target)
			if tt.found {
				if result == nil {
					t.Errorf("expected to find condition, got nil")
				} else if result.ID != tt.expectedID {
					t.Errorf("expected ID %s, got %s", tt.expectedID, result.ID)
				}
			} else {
				if result != nil {
					t.Errorf("expected nil, got %v", result)
				}
			}
		})
	}
}

func TestReplaceCondition(t *testing.T) {
	tests := []struct {
		name        string
		conditions  []*Condition
		target      *Condition
		expectedLen int
		expectNew   bool
	}{
		{
			name:        "nilTarget",
			conditions:  []*Condition{{ID: "test1"}},
			target:      nil,
			expectedLen: numOne,
			expectNew:   false,
		},
		{
			name:        "appendNewCondition",
			conditions:  []*Condition{{ID: "test1"}},
			target:      &Condition{ID: "test2"},
			expectedLen: numTwo,
			expectNew:   true,
		},
		{
			name:        "replaceFirst",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}},
			target:      &Condition{ID: "test1", Status: metav1.ConditionTrue},
			expectedLen: numTwo,
			expectNew:   false,
		},
		{
			name:        "replaceMiddle",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}, {ID: "test3"}},
			target:      &Condition{ID: "test2", Status: metav1.ConditionTrue},
			expectedLen: numThree,
			expectNew:   false,
		},
		{
			name:        "replaceLast",
			conditions:  []*Condition{{ID: "test1"}, {ID: "test2"}},
			target:      &Condition{ID: "test2", Status: metav1.ConditionTrue},
			expectedLen: numTwo,
			expectNew:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ReplaceCondition(tt.conditions, tt.target)
			if len(result) != tt.expectedLen {
				t.Errorf("expected length %d, got %d", tt.expectedLen, len(result))
			}
			if tt.expectNew {
				found := false
				for _, c := range result {
					if c.ID == tt.target.ID {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected new condition to be added")
				}
			}
		})
	}
}

func TestConditionCount(t *testing.T) {
	baseTime := metav1.Now()

	tests := []struct {
		name              string
		conditions        []*Condition
		commandCount      int
		expectedSucceeded int
		expectedFailed    int
		expectedStatus    metav1.ConditionStatus
		expectedPhase     CommandPhase
	}{
		{
			name:              "emptyConditions",
			conditions:        []*Condition{},
			commandCount:      numThree,
			expectedSucceeded: numZero,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionUnknown,
			expectedPhase:     CommandRunning,
		},
		{
			name: "allSucceeded",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test3", Status: metav1.ConditionTrue, Phase: CommandComplete},
			},
			commandCount:      numThree,
			expectedSucceeded: numThree,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name: "allFailed",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionFalse, Phase: CommandFailed},
				{ID: "test2", Status: metav1.ConditionFalse, Phase: CommandFailed},
			},
			commandCount:      numTwo,
			expectedSucceeded: numZero,
			expectedFailed:    numTwo,
			expectedStatus:    metav1.ConditionFalse,
			expectedPhase:     CommandFailed,
		},
		{
			name: "mixedSuccessAndFailure",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionFalse, Phase: CommandFailed},
				{ID: "test3", Status: metav1.ConditionTrue, Phase: CommandComplete},
			},
			commandCount:      numThree,
			expectedSucceeded: numTwo,
			expectedFailed:    numOne,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name: "incompleteWithLessConditions",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandRunning},
			},
			commandCount:      numFive,
			expectedSucceeded: numTwo,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionUnknown,
			expectedPhase:     CommandRunning,
		},
		{
			name: "incompleteAtCountWithRunning",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandRunning},
			},
			commandCount:      numTwo,
			expectedSucceeded: numTwo,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandRunning,
		},
		{
			name: "incompleteAtCountWithSkip",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandSkip},
			},
			commandCount:      numTwo,
			expectedSucceeded: numTwo,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name: "failedPhaseOverridesStatus",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandFailed},
			},
			commandCount:      numTwo,
			expectedSucceeded: numTwo,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionFalse,
			expectedPhase:     CommandFailed,
		},
		{
			name: "singleConditionTrue",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete, LastStartTime: &baseTime},
			},
			commandCount:      numOne,
			expectedSucceeded: numOne,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name: "singleConditionFalse",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionFalse, Phase: CommandFailed, LastStartTime: &baseTime},
			},
			commandCount:      numOne,
			expectedSucceeded: numZero,
			expectedFailed:    numOne,
			expectedStatus:    metav1.ConditionFalse,
			expectedPhase:     CommandFailed,
		},
		{
			name: "fiveConditionsMixed",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test3", Status: metav1.ConditionFalse, Phase: CommandFailed},
				{ID: "test4", Status: metav1.ConditionTrue, Phase: CommandComplete},
				{ID: "test5", Status: metav1.ConditionTrue, Phase: CommandComplete},
			},
			commandCount:      numFive,
			expectedSucceeded: numFour,
			expectedFailed:    numOne,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name: "allSkipPhase",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandSkip},
				{ID: "test2", Status: metav1.ConditionTrue, Phase: CommandSkip},
			},
			commandCount:      numTwo,
			expectedSucceeded: numTwo,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name: "unknownConditionStatus",
			conditions: []*Condition{
				{ID: "test1", Status: metav1.ConditionUnknown, Phase: CommandPending},
			},
			commandCount:      numOne,
			expectedSucceeded: numOne,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionUnknown,
			expectedPhase:     CommandUnKnown,
		},
		{
			name:              "tenConditionsAllSucceeded",
			conditions:        generateConditions(numTen, metav1.ConditionTrue, CommandComplete),
			commandCount:      numTen,
			expectedSucceeded: numTen,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name:              "sevenConditionsWithThreeFailed",
			conditions:        generateMixedConditions(numSeven),
			commandCount:      numSeven,
			expectedSucceeded: numFive,
			expectedFailed:    numTwo,
			expectedStatus:    metav1.ConditionTrue,
			expectedPhase:     CommandComplete,
		},
		{
			name:              "commandCountGreaterThanConditions",
			conditions:        []*Condition{{ID: "test1", Status: metav1.ConditionTrue, Phase: CommandComplete}},
			commandCount:      numFifteen,
			expectedSucceeded: numOne,
			expectedFailed:    numZero,
			expectedStatus:    metav1.ConditionUnknown,
			expectedPhase:     CommandRunning,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConditionCount(tt.conditions, tt.commandCount)
			if result.Succeeded != tt.expectedSucceeded {
				t.Errorf("expected succeeded %d, got %d", tt.expectedSucceeded, result.Succeeded)
			}
			if result.Failed != tt.expectedFailed {
				t.Errorf("expected failed %d, got %d", tt.expectedFailed, result.Failed)
			}
			if result.Status != tt.expectedStatus {
				t.Errorf("expected status %v, got %v", tt.expectedStatus, result.Status)
			}
			if result.Phase != tt.expectedPhase {
				t.Errorf("expected phase %s, got %s", tt.expectedPhase, result.Phase)
			}
		})
	}
}

func generateConditions(count int, status metav1.ConditionStatus, phase CommandPhase) []*Condition {
	conditions := make([]*Condition, count)
	for i := 0; i < count; i++ {
		conditions[i] = &Condition{
			ID:     fmt.Sprintf("test%d", i+1),
			Status: status,
			Phase:  phase,
		}
	}
	return conditions
}

func generateMixedConditions(count int) []*Condition {
	conditions := make([]*Condition, count)
	for i := 0; i < count; i++ {
		cStatus := metav1.ConditionTrue
		cPhase := CommandComplete
		if i%numThree == numTwo {
			cStatus = metav1.ConditionFalse
			cPhase = CommandFailed
		} else if i%numThree == numOne {
			cStatus = metav1.ConditionTrue
			cPhase = CommandRunning
		}
		conditions[i] = &Condition{
			ID:     fmt.Sprintf("test%d", i+1),
			Status: cStatus,
			Phase:  cPhase,
		}
	}
	return conditions
}
