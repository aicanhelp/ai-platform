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

package phaseframe

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// Phase is the interface of a phase
type Phase interface {
	// Name returns the name of the phase
	Name() confv1beta1.BKEClusterPhase

	// Execute executes the phase
	Execute() (ctrl.Result, error)

	// ExecutePreHook executes the pre-hook of the phase
	ExecutePreHook() error

	// ExecutePostHook executes the post-hook of the phase
	ExecutePostHook(err error) error

	// NeedExecute returns whether the phase needs to be executed, run in webhook
	NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) (needExecute bool)

	//RegisterPreHooks is used to register a custom pre hook function
	RegisterPreHooks(hooks ...func(p Phase) error)

	// RegisterPostHooks is used to register a custom post hook function
	RegisterPostHooks(hook ...func(p Phase, err error) error)

	// Report reports the phase status to BKECluster.Status
	Report(msg string, onlyRecord bool) error

	SetCName(name string)

	SetStatus(status confv1beta1.BKEClusterPhaseStatus)

	GetStatus() confv1beta1.BKEClusterPhaseStatus

	SetStartTime(t metav1.Time)

	GetStartTime() metav1.Time

	GetPhaseContext() *PhaseContext

	SetPhaseContext(ctx *PhaseContext)
}
