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
	"errors"

	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

// TODO: new BKEPhase steps
// 1, replace XXX to your custom phase name
// 2, add phase cName to list.PhaseNameCNMap map
// 3. write NeedExecute function code
// 4. write Execute function code
// 5. register your phase in list.CommonPhases or list.DeployPhases or list.PostDeployPhases
//	  the phase is sequential execution

const (
	XXXName confv1beta1.BKEClusterPhase = "XXX"
)

type XXX struct {
	phaseframe.BasePhase
}

func NewXXX(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, XXXName)
	return &XXX{BasePhase: base}
}

func (e *XXX) Execute() (ctrl.Result, error) {
	if err := e.reconcileXXX(); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

func (e *XXX) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	// true means need execute and false means not
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	// TODO: your logic here, sure DefaultNeedExecute will be called at first
	// ...
	return false
}

func (e *XXX) reconcileXXX() error {
	// Return an error instead of panicking
	return errors.New("reconcileXXX not implemented")
}
