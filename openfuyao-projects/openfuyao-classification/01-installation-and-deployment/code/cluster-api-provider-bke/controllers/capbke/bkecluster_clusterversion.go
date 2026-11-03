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

package capbke

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/clusterversion"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/featuregate"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

// ensureClusterVersionOnInstall ensures a ClusterVersion exists for this BKECluster during install/reconcile.
// Per design, BKECluster creates CV; ClusterVersion reconciler then pulls ri and creates ReleaseImage.
func (r *BKEClusterReconciler) ensureClusterVersionOnInstall(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	bkeLogger *bkev1beta1.BKELogger,
) (ctrl.Result, error) {
	if bkeCluster == nil || bkeCluster.Spec.ClusterConfig == nil {
		return ctrl.Result{}, nil
	}
	if clusterversion.OpenFuyaoVersionForBKECluster(bkeCluster) == "" {
		return ctrl.Result{}, nil
	}

	created, err := clusterversion.EnsureClusterVersionForBKECluster(ctx, r.Client, r.Scheme, bkeCluster)
	if err != nil {
		bkeLogger.Error(constant.ReconcileErrorReason, "ensure ClusterVersion failed: %v", err)
		return ctrl.Result{}, err
	}
	if created {
		msg := fmt.Sprintf("created ClusterVersion %s/%s for BKECluster install",
			bkeCluster.Namespace, bkeCluster.Name)
		bkeLogger.Info("ClusterVersionCreated", msg)
		r.Recorder.Event(bkeCluster, corev1.EventTypeNormal, "ClusterVersionCreated", msg)
	}
	return ctrl.Result{}, nil
}

// completeClusterVersionInstall patches ClusterVersion.status when BKECluster install finishes.
// Per design, only BKECluster Reconciler writes CV status on install completion.
func (r *BKEClusterReconciler) completeClusterVersionInstall(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	bkeLogger *bkev1beta1.BKELogger,
) error {
	if bkeCluster == nil || !bkeCluster.DeletionTimestamp.IsZero() {
		return nil
	}
	if bkeCluster.Status.ClusterStatus != bkev1beta1.ClusterStatusReady {
		return nil
	}
	if _, ok := featuregate.UpgradeReady(bkeCluster); ok {
		return nil
	}
	installed := clusterversion.OpenFuyaoVersionForBKECluster(bkeCluster)
	if installed == "" {
		return nil
	}

	cv, err := clusterversion.GetClusterVersionForBKECluster(ctx, r.Client, bkeCluster)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !clusterversion.NeedsInstallCompletion(cv) {
		return nil
	}
	desired := strings.TrimSpace(cv.Spec.DesiredVersion)
	if desired != "" && desired != installed {
		return nil
	}

	cvOrig := cv.DeepCopy()
	clusterversion.ApplyInstallCompleteStatus(cv, installed)
	if err := r.Status().Patch(ctx, cv, client.MergeFrom(cvOrig)); err != nil {
		bkeLogger.Error(constant.ReconcileErrorReason, "patch ClusterVersion install status failed: %v", err)
		return err
	}
	msg := fmt.Sprintf("ClusterVersion %s/%s install complete at %s", cv.Namespace, cv.Name, installed)
	bkeLogger.Info("ClusterVersionInstallComplete", msg)
	r.Recorder.Event(bkeCluster, corev1.EventTypeNormal, "ClusterVersionInstallComplete", msg)
	return nil
}
