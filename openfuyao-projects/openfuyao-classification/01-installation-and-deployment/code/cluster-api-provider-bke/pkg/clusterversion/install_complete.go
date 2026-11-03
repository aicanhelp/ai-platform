/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package clusterversion

import (
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

const (
	// ClusterVersionConditionReady is the standard readiness condition type on ClusterVersion.
	ClusterVersionConditionReady = "Ready"
)

// NeedsInstallCompletion reports whether BKECluster install completion should patch CV status.
func NeedsInstallCompletion(cv *cvv1alpha1.ClusterVersion) bool {
	if cv == nil {
		return false
	}
	desired := strings.TrimSpace(cv.Spec.DesiredVersion)
	if desired == "" {
		return false
	}
	current := strings.TrimSpace(cv.Status.CurrentVersion)
	phase := cv.Status.Phase

	// Pending version hop: install completion must not advance CV currentVersion.
	if desired != "" && current != "" && current != desired {
		return false
	}
	if phase != cvv1alpha1.ClusterVersionPhaseUpgrading {
		return true
	}
	return current == ""
}

// ApplyInstallCompleteStatus updates cv status in memory after cluster install succeeds.
func ApplyInstallCompleteStatus(cv *cvv1alpha1.ClusterVersion, installedVersion string) {
	if cv == nil {
		return
	}
	installedVersion = strings.TrimSpace(installedVersion)
	msg := "cluster install complete"

	cv.Status.CurrentVersion = installedVersion
	cv.Status.Phase = cvv1alpha1.ClusterVersionPhaseReady
	now := metav1.NewTime(time.Now())
	cv.Status.Conditions = upsertClusterVersionCondition(cv.Status.Conditions, cvv1alpha1.ClusterVersionCondition{
		Type:               ClusterVersionConditionReady,
		Status:             metav1.ConditionTrue,
		Reason:             "InstallComplete",
		Message:            msg,
		LastTransitionTime: now,
	})
}

func upsertClusterVersionCondition(
	conditions []cvv1alpha1.ClusterVersionCondition,
	condition cvv1alpha1.ClusterVersionCondition,
) []cvv1alpha1.ClusterVersionCondition {
	for i := range conditions {
		if conditions[i].Type == condition.Type {
			conditions[i] = condition
			return conditions
		}
	}
	return append(conditions, condition)
}
