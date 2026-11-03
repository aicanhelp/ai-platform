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

package condition

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// ConditionMark add or update condition to BKECluster
func ConditionMark(bkeCluster *bkev1beta1.BKECluster, conditionType confv1beta1.ClusterConditionType, conditionStatus confv1beta1.ConditionStatus, reason, message string) {
	AddonConditionMark(MarkOptions{
		BKECluster: bkeCluster, Type: conditionType, Status: conditionStatus,
		Reason: reason, Message: message,
	})
}

// MarkOptions contains the values used to create or update a condition.
type MarkOptions struct {
	// BKECluster is the cluster whose status condition is updated.
	BKECluster *bkev1beta1.BKECluster
	// Type identifies the condition being updated.
	Type confv1beta1.ClusterConditionType
	// Status is the new condition status.
	Status confv1beta1.ConditionStatus
	// Reason is the machine-readable reason for the update.
	Reason string
	// Message is the human-readable detail for the update.
	Message string
	// AddonName identifies an addon condition; it is empty for cluster conditions.
	AddonName string
}

// AddonConditionMark adds or updates a cluster or addon condition.
func AddonConditionMark(options MarkOptions) {
	bkeCluster := options.BKECluster
	if bkeCluster.Status.Conditions == nil {
		bkeCluster.Status.Conditions = []confv1beta1.ClusterCondition{}
	}
	now := metav1.Now()

	if updateExistingAddonCondition(bkeCluster, options, &now) {
		return
	}
	if updateExistingCondition(bkeCluster, options, &now) {
		return
	}

	condition := confv1beta1.ClusterCondition{
		Type:               options.Type,
		Status:             options.Status,
		Reason:             options.Reason,
		Message:            options.Message,
		AddonName:          options.AddonName,
		LastTransitionTime: &now,
	}
	bkeCluster.Status.Conditions = append(bkeCluster.Status.Conditions, condition)
}

func updateExistingAddonCondition(
	bkeCluster *bkev1beta1.BKECluster,
	options MarkOptions,
	now *metav1.Time,
) bool {
	if options.AddonName == "" || options.Type != bkev1beta1.ClusterAddonCondition {
		return false
	}
	condition, ok := HasAddonCondition(bkeCluster, options.AddonName)
	if !ok {
		return false
	}
	condition.Status = options.Status
	condition.Reason = options.Reason
	condition.Message = options.Message
	condition.LastTransitionTime = now
	replaceConditionByType(bkeCluster, options.Type, condition)
	return true
}

func updateExistingCondition(
	bkeCluster *bkev1beta1.BKECluster,
	options MarkOptions,
	now *metav1.Time,
) bool {
	condition, ok := HasCondition(options.Type, bkeCluster)
	if !ok {
		return false
	}
	condition.LastTransitionTime = now
	condition.Reason = options.Reason
	condition.Message = options.Message
	condition.Status = options.Status
	condition.AddonName = options.AddonName
	replaceConditionByType(bkeCluster, options.Type, condition)
	return true
}

func replaceConditionByType(
	bkeCluster *bkev1beta1.BKECluster,
	conditionType confv1beta1.ClusterConditionType,
	condition *confv1beta1.ClusterCondition,
) {
	for i := range bkeCluster.Status.Conditions {
		if bkeCluster.Status.Conditions[i].Type == conditionType {
			bkeCluster.Status.Conditions[i] = *condition
		}
	}
}

func HasCondition(conditionType confv1beta1.ClusterConditionType, bkeCluster *bkev1beta1.BKECluster) (*confv1beta1.ClusterCondition, bool) {
	if bkeCluster.Status.Conditions == nil {
		return nil, false
	}

	for _, condition := range bkeCluster.Status.Conditions {
		if condition.Type == conditionType {
			return &condition, true
		}
	}

	return nil, false
}

func HasAddonCondition(bkeCluster *bkev1beta1.BKECluster, addonName string) (*confv1beta1.ClusterCondition, bool) {
	if bkeCluster.Status.Conditions == nil {
		return nil, false
	}
	for _, condition := range bkeCluster.Status.Conditions {
		if condition.Type == bkev1beta1.ClusterAddonCondition && condition.AddonName == addonName {
			return &condition, true
		}
	}
	return nil, false
}

func HasConditionStatus(conditionType confv1beta1.ClusterConditionType, bkeCluster *bkev1beta1.BKECluster, conditionStatus confv1beta1.ConditionStatus) bool {
	condition, ok := HasCondition(conditionType, bkeCluster)
	if !ok {
		return false
	}
	return condition.Status == conditionStatus
}

func RemoveCondition(conditionType confv1beta1.ClusterConditionType, bkeCluster *bkev1beta1.BKECluster) {
	conditions := bkeCluster.Status.Conditions
	if conditions == nil {
		return
	}

	indexToRemove := -1

	for i, condition := range conditions {
		if condition.Type == conditionType {
			indexToRemove = i
			break
		}
	}

	if indexToRemove != -1 {
		conditions = append(conditions[:indexToRemove], conditions[indexToRemove+1:]...)
		bkeCluster.Status.Conditions = conditions
	}
}
