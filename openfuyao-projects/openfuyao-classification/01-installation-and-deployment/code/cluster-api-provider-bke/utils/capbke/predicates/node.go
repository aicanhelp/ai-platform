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
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

func NodeNotReadyPredicate() predicate.Funcs {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldObj, ok := e.ObjectOld.(*corev1.Node)
			if !ok {
				return false
			}
			newObj, ok := e.ObjectNew.(*corev1.Node)
			if !ok {
				return false
			}
			oldCondition := getNodeCondition(oldObj, corev1.NodeReady)
			newCondition := getNodeCondition(newObj, corev1.NodeReady)

			if oldCondition == nil || newCondition == nil {
				return false
			}
			if oldCondition.Status == newCondition.Status {
				return false
			}
			return true
		},

		CreateFunc: func(e event.CreateEvent) bool {
			obj, ok := e.Object.(*corev1.Node)
			if !ok {
				return false
			}
			if obj != nil {
				condition := getNodeCondition(obj, corev1.NodeReady)
				if condition != nil {
					if condition.Status == corev1.ConditionFalse {
						return true
					}
				}
			}
			return false
		},
		DeleteFunc:  func(event.DeleteEvent) bool { return false },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}

func getNodeCondition(node *corev1.Node, conditionType corev1.NodeConditionType) *corev1.NodeCondition {
	if node.Status.Conditions == nil {
		return nil
	}
	for _, condition := range node.Status.Conditions {
		if condition.Type == conditionType {
			return &condition
		}
	}
	return nil
}
