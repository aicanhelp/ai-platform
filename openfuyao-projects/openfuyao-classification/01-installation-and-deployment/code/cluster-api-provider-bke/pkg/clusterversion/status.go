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
	"context"
	"fmt"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
)

// ShouldSyncClusterVersionInstallStatus reports whether EnsureCluster may patch CV install status.
func ShouldSyncClusterVersionInstallStatus(
	ctx context.Context,
	c client.Client,
	bkeCluster *bkev1beta1.BKECluster,
) (bool, error) {
	if c == nil || bkeCluster == nil {
		return false, nil
	}
	if upgradeReady, _ := bkeCluster.Annotations[annotation.CVOUpgradeReadyAnnotationKey]; strings.TrimSpace(upgradeReady) != "" {
		return false, nil
	}
	cv, err := GetClusterVersionForBKECluster(ctx, c, bkeCluster)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	if !NeedsInstallCompletion(cv) {
		return false, nil
	}
	desired := strings.TrimSpace(cv.Spec.DesiredVersion)
	current := strings.TrimSpace(cv.Status.CurrentVersion)
	if desired != "" && current != "" && current != desired {
		return false, nil
	}
	installed := OpenFuyaoVersionForBKECluster(bkeCluster)
	if desired == "" || installed == "" || installed != desired {
		return false, nil
	}
	if cv.Status.Phase == cvv1alpha1.ClusterVersionPhaseReady && current == desired {
		return false, nil
	}
	return true, nil
}

// SyncClusterVersionInstallStatus patches ClusterVersion.status when cluster install completes.
// It must not advance currentVersion to spec.desiredVersion during an in-flight upgrade hop.
func SyncClusterVersionInstallStatus(
	ctx context.Context,
	c client.Client,
	bkeCluster *bkev1beta1.BKECluster,
) error {
	ok, err := ShouldSyncClusterVersionInstallStatus(ctx, c, bkeCluster)
	if err != nil || !ok {
		return err
	}
	cv, err := GetClusterVersionForBKECluster(ctx, c, bkeCluster)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	installed := OpenFuyaoVersionForBKECluster(bkeCluster)
	if installed == "" {
		return nil
	}

	orig := cv.DeepCopy()
	ApplyInstallCompleteStatus(cv, installed)
	return c.Status().Patch(ctx, cv, client.MergeFrom(orig))
}

// CompleteUpgradeHop patches ClusterVersion.status after one declarative upgrade hop.
func CompleteUpgradeHop(
	ctx context.Context,
	c client.Client,
	cv *cvv1alpha1.ClusterVersion,
	hopVersion string,
) error {
	if c == nil || cv == nil {
		return fmt.Errorf("client or cluster version is nil")
	}
	hopVersion = strings.TrimSpace(hopVersion)
	if hopVersion == "" {
		return fmt.Errorf("hop version is empty")
	}

	orig := cv.DeepCopy()
	fromVersion := strings.TrimSpace(cv.Status.CurrentVersion)
	cv.Status.CurrentVersion = hopVersion
	if hopVersion == cv.Spec.DesiredVersion {
		cv.Status.Phase = cvv1alpha1.ClusterVersionPhaseReady
	} else {
		cv.Status.Phase = cvv1alpha1.ClusterVersionPhaseUpgrading
	}
	if fromVersion != "" && hopVersion != fromVersion {
		now := metav1.Now()
		cv.Status.UpgradeHistory = append(cv.Status.UpgradeHistory, cvv1alpha1.ClusterUpgradeRecord{
			From:        fromVersion,
			To:          hopVersion,
			CompletedAt: &now,
			Status:      cvv1alpha1.ClusterUpgradeRecordStatusSucceeded,
		})
	}
	return c.Status().Patch(ctx, cv, client.MergeFrom(orig))
}

// FailUpgradeHop patches ClusterVersion.status when a declarative upgrade hop fails terminally.
func FailUpgradeHop(
	ctx context.Context,
	c client.Client,
	cv *cvv1alpha1.ClusterVersion,
	hopTarget string,
	cause error,
) error {
	if c == nil || cv == nil {
		return fmt.Errorf("client or cluster version is nil")
	}
	hopTarget = strings.TrimSpace(hopTarget)
	msg := "declarative upgrade failed"
	if cause != nil {
		msg = cause.Error()
	}

	orig := cv.DeepCopy()
	fromVersion := strings.TrimSpace(cv.Status.CurrentVersion)
	cv.Status.Phase = cvv1alpha1.ClusterVersionPhaseFailed
	cv.Status.Conditions = upgradeFailureCondition(msg)
	if hopTarget != "" && fromVersion != "" && hopTarget != fromVersion {
		now := metav1.Now()
		cv.Status.UpgradeHistory = append(cv.Status.UpgradeHistory, cvv1alpha1.ClusterUpgradeRecord{
			From:        fromVersion,
			To:          hopTarget,
			CompletedAt: &now,
			Status:      cvv1alpha1.ClusterUpgradeRecordStatusFailed,
		})
	}
	return c.Status().Patch(ctx, cv, client.MergeFrom(orig))
}

func upgradeFailureCondition(message string) []cvv1alpha1.ClusterVersionCondition {
	return []cvv1alpha1.ClusterVersionCondition{{
		Type:               "Ready",
		Status:             metav1.ConditionFalse,
		Reason:             "UpgradeHopFailed",
		Message:            message,
		LastTransitionTime: metav1.Now(),
	}}
}

// HasUpgradeRecord reports whether an upgrade history entry exists for from→to.
func HasUpgradeRecord(records []cvv1alpha1.ClusterUpgradeRecord, from, to string) bool {
	for _, record := range records {
		if record.From == from && record.To == to {
			return true
		}
	}
	return false
}

// Now returns the current time as metav1.Time.
func Now() *metav1.Time {
	t := metav1.NewTime(time.Now())
	return &t
}
