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

package predicates

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestNodeNotReadyPredicate_Update(t *testing.T) {
	pred := NodeNotReadyPredicate()
	oldNode := &corev1.Node{
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{
				{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
			},
		},
	}
	newNode := &corev1.Node{
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{
				{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
			},
		},
	}
	e := event.UpdateEvent{ObjectOld: oldNode, ObjectNew: newNode}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestNodeNotReadyPredicate_Create(t *testing.T) {
	pred := NodeNotReadyPredicate()
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{
				{Type: corev1.NodeReady, Status: corev1.ConditionFalse},
			},
		},
	}
	e := event.CreateEvent{Object: node}
	result := pred.Create(e)
	assert.True(t, result)
}

func TestNodeNotReadyPredicate_Delete(t *testing.T) {
	pred := NodeNotReadyPredicate()
	e := event.DeleteEvent{}
	result := pred.Delete(e)
	assert.False(t, result)
}

func TestGetNodeCondition(t *testing.T) {
	node := &corev1.Node{
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{
				{Type: corev1.NodeReady, Status: corev1.ConditionTrue},
			},
		},
	}
	condition := getNodeCondition(node, corev1.NodeReady)
	assert.NotNil(t, condition)
	assert.Equal(t, corev1.ConditionTrue, condition.Status)
}
