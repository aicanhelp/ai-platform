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

package componentfactory

import (
	"fmt"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
)

// PhaseRunner executes inline upgrade phases via ComponentFactory.
type PhaseRunner struct {
	Factory *ComponentFactory
}

// Execute resolves and runs an inline handler when NeedExecute is true.
func (r *PhaseRunner) Execute(
	phaseCtx *phaseframe.PhaseContext,
	oldCluster, newCluster *bkev1beta1.BKECluster,
	handler, version string,
) error {
	if r == nil || r.Factory == nil {
		return fmt.Errorf("phase runner or factory is nil")
	}
	phase, err := ResolveInlineUpgrade(r.Factory, handler, version, phaseCtx)
	if err != nil {
		return err
	}
	if !phase.NeedExecute(oldCluster, newCluster) {
		return nil
	}
	if err := phase.ExecutePreHook(); err != nil {
		return err
	}
	result, err := phase.Execute()
	if postErr := phase.ExecutePostHook(err); postErr != nil {
		return postErr
	}
	if err != nil {
		return err
	}
	if result.Requeue || result.RequeueAfter > 0 {
		return fmt.Errorf("inline phase %s requested requeue", handler)
	}
	return nil
}
