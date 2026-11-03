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
	"sigs.k8s.io/cluster-api/util/conditions"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func BKEMachineConditionUpdate() predicate.Funcs {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			newObj, newObjOk := e.ObjectNew.(*bkev1beta1.BKEMachine)
			if !newObjOk {
				return false
			}
			if newObj != nil {
				oldObj, oldObjOk := e.ObjectOld.(*bkev1beta1.BKEMachine)
				if !oldObjOk || oldObj == nil {
					return false
				}
				oldCondition := conditions.Get(oldObj, bkev1beta1.BootstrapSucceededCondition)
				newCondition := conditions.Get(newObj, bkev1beta1.BootstrapSucceededCondition)
				if oldCondition == nil && newCondition != nil {
					return true
				}
				if oldCondition == nil || newCondition == nil {
					return false
				}
				oldConditionTime := oldCondition.LastTransitionTime
				newConditionTime := newCondition.LastTransitionTime
				return oldConditionTime != newConditionTime
			}
			return false
		},
		CreateFunc:  func(event.CreateEvent) bool { return false },
		DeleteFunc:  func(event.DeleteEvent) bool { return false },
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}
