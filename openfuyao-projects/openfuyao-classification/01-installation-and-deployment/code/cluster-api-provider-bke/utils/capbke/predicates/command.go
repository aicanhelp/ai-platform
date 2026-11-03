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
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
)

// CommandUpdateCompleted is a predicate that returns true if the command changes from other phase to completed phase
func CommandUpdateCompleted() predicate.Funcs {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldCommand, ok := e.ObjectOld.(*agentv1beta1.Command)
			if !ok {
				return false
			}
			if oldCommand != nil {
				newCommand, newOk := e.ObjectNew.(*agentv1beta1.Command)
				if !newOk || newCommand == nil {
					return false
				}
				// For command, Reconciler expects that all nodes specified by command have been executed and have a result
				switch {
				case newCommand == nil:
					return false
				case len(newCommand.Status) < len(oldCommand.Status):
					return false
				case newCommand.Spec.NodeName != "" && len(newCommand.Status) != 1:
					return false
				case newCommand.Spec.NodeSelector == nil && len(newCommand.Status) == newCommand.Spec.NodeSelector.Size():
					return false
				default:
					// If none of the above conditions are met, the command update is considered completed
					return true
				}
			}
			return false
		},
		CreateFunc:  func(event.CreateEvent) bool { return false },
		DeleteFunc:  func(event.DeleteEvent) bool { return false },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}
