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

package phaseframe

import (
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

// GetVersionContext returns the version context attached to the phase context.
func (b *BasePhase) GetVersionContext() *upgrade.VersionContext {
	if b.Ctx == nil {
		return nil
	}
	return b.Ctx.VersionContext
}

// ComponentVersionDecision uses VersionContext when present.
// When decided is true, needUpgrade indicates whether the component version differs.
func (b *BasePhase) ComponentVersionDecision(component string) (bool, bool) {
	vc := b.GetVersionContext()
	if vc == nil {
		return false, false
	}
	if vc.HasTarget(component) {
		return true, vc.NeedsUpgrade(component)
	}
	return false, false
}

// NeedExecuteWithVersionContext runs version-context logic when available, otherwise legacy.
func (b *BasePhase) NeedExecuteWithVersionContext(
	component string,
	old, new *bkev1beta1.BKECluster,
	legacy func(old, new *bkev1beta1.BKECluster) bool,
) bool {
	if decided, need := b.ComponentVersionDecision(component); decided {
		return need
	}
	if legacy == nil {
		return false
	}
	return legacy(old, new)
}
