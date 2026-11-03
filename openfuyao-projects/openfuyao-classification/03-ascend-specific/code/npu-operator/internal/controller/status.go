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
	"context"
	"fmt"

	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const (
	readyCondition = "Ready"
	errorCondition = "Error"
)

const (
	reconciledReason         = "Reconciled"
	reconcileFailedReason    = "ReconcileFailed"
	reconcilingReason        = "Reconciling"
	noNFDLabelsReason        = "NoNFDLabels"
	noNPUNodesReason         = "NoNPUNodes"
	componentConflictReason  = "ComponentConflict"
	componentNotReadyReason  = "ComponentNotReady"
	componentUnmanagedReason = "ComponentUnmanaged"
)

func (r *NPUClusterPolicyReconciler) setConditionsReady(ctx context.Context, reason, message string) (err error) {
	err = r.updateStatus(ctx, func(s *v1.NPUClusterPolicyStatus) bool {
		changed := s.Phase != v1.PolicyReady
		if changed {
			s.Phase = v1.PolicyReady
		}
		return s.UpdateConditions(errorCondition, readyCondition, reason, message) || changed
	})
	if err != nil {
		log.FromContext(ctx).V(1).Error(nil, err.Error())
	}
	return
}

func (r *NPUClusterPolicyReconciler) setConditionsError(ctx context.Context, reason, message string) (err error) {
	err = r.updateStatus(ctx, func(s *v1.NPUClusterPolicyStatus) bool {
		changed := s.Phase != v1.PolicyNotReady
		if changed {
			s.Phase = v1.PolicyNotReady
		}
		return s.UpdateConditions(readyCondition, errorCondition, reason, message) || changed
	})
	if err != nil {
		log.FromContext(ctx).V(1).Error(nil, err.Error())
	}
	return
}

type updateStatusCallback func(*v1.NPUClusterPolicyStatus) bool

func (r *NPUClusterPolicyReconciler) updateStatus(ctx context.Context, cb updateStatusCallback) error {
	if err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := r.getInstance(ctx, r.instance.Name); err != nil {
			return err
		}
		if !cb(&r.instance.Status) {
			return nil
		}
		r.instance.Status.Namespace = r.namespace
		return r.Status().Update(ctx, r.instance)
	}); err != nil {
		return fmt.Errorf("Failed to update status: %w", err)
	}
	return nil
}
