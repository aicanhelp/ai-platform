// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func Test_setConditionsError(t *testing.T) {
	const (
		testReason = componentNotReadyReason
		testMsg    = "device-plugin not ready"
	)

	initialStatus := v1.NPUClusterPolicyStatus{
		Phase: v1.PolicyReady,
		Conditions: []metav1.Condition{
			{
				Type:               readyCondition,
				Status:             metav1.ConditionTrue,
				Reason:             reconciledReason,
				Message:            "previous success",
				LastTransitionTime: metav1.NewTime(time.Now()),
			},
		},
	}

	callback := func(s *v1.NPUClusterPolicyStatus) bool {
		changed := s.Phase != v1.PolicyNotReady
		if changed {
			s.Phase = v1.PolicyNotReady
		}
		condChanged := s.UpdateConditions(readyCondition, errorCondition, testReason, testMsg)
		return condChanged || changed
	}

	modifiedStatus := initialStatus
	callback(&modifiedStatus)

	assert.Equal(t, v1.PolicyNotReady, modifiedStatus.Phase,
		"Phase should switch from PolicyReady to PolicyNotReady")

	var readyCond, errorCond metav1.Condition
	for _, cond := range modifiedStatus.Conditions {
		switch cond.Type {
		case readyCondition:
			readyCond = cond
		case errorCondition:
			errorCond = cond
		default:
		}
	}

	assert.NotEmpty(t, errorCond.Type, "Error condition should be added")
	assert.Equal(t, metav1.ConditionTrue, errorCond.Status, "Error condition status must be True")
	assert.Equal(t, testReason, errorCond.Reason, "Error condition reason mismatch")
	assert.Equal(t, testMsg, errorCond.Message, "Error condition message mismatch")

	assert.Equal(t, metav1.ConditionFalse, readyCond.Status, "Ready condition status must be False")
}
