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

package phases

import (
	"fmt"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	v "gopkg.openfuyao.cn/cluster-api-provider-bke/version"
)

const (
	EnsureFinalizerName confv1beta1.BKEClusterPhase = "EnsureFinalizer"
)

type EnsureFinalizer struct {
	phaseframe.BasePhase
}

func NewEnsureFinalizer(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureFinalizerName)
	return &EnsureFinalizer{
		BasePhase: base,
	}
}

func (e *EnsureFinalizer) Execute() (ctrl.Result, error) {
	controllerutil.AddFinalizer(e.Ctx.BKECluster, bkev1beta1.ClusterFinalizer)
	e.Ctx.Log.Info("VERSION", "-----------------Start Reconcile BKECluster-----------------------")
	e.Ctx.Log.Info("VERSION", fmt.Sprintf("BKE Version     : %s", v.Version))
	e.Ctx.Log.Info("VERSION", fmt.Sprintf("BKE GitCommitId : %s", v.GitCommitID))
	e.Ctx.Log.Info("VERSION", fmt.Sprintf("BKE Architecture: %s", v.Architecture))
	e.Ctx.Log.Info("VERSION", fmt.Sprintf("BKE BuildTime   : %s", v.BuildTime))
	e.Ctx.Log.Info("VERSION", "------------------------------------------------------------------")
	return ctrl.Result{}, nil
}

func (e *EnsureFinalizer) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if controllerutil.ContainsFinalizer(new, bkev1beta1.ClusterFinalizer) {
		e.SetStatus(bkev1beta1.PhaseSkipped)
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}
