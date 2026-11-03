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

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phases"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func registerInlineHandler(f *ComponentFactory, handler, version string) error {
	switch handler {
	case upgrade.InlineHandlerEtcdUpgrade:
		f.Register(handler, version, phases.NewEnsureEtcdUpgrade)
	case upgrade.InlineHandlerMasterUpgrade:
		f.Register(handler, version, phases.NewEnsureMasterUpgrade)
	case upgrade.InlineHandlerWorkerUpgrade:
		f.Register(handler, version, phases.NewEnsureWorkerUpgrade)
	case upgrade.InlineHandlerContainerdUpgrade:
		f.Register(handler, version, phases.NewEnsureContainerdUpgrade)
	case upgrade.InlineHandlerNginxUpgrade:
		f.Register(handler, version, phases.NewEnsureNginxUpgrade)
	case upgrade.InlineHandlerAgentUpgrade:
		f.Register(handler, version, phases.NewEnsureAgentUpgrade)
	default:
		return fmt.Errorf("unknown inline handler %q", handler)
	}
	return nil
}

func registerInlineComponent(
	f *ComponentFactory,
	handler, version string,
	componentVersion *cvv1alpha1.ComponentVersion,
) error {
	if handler != upgrade.InlineHandlerPreUpgradeResources {
		return registerInlineHandler(f, handler, version)
	}
	if componentVersion == nil {
		return fmt.Errorf("component version is required for inline handler %q", handler)
	}
	f.Register(handler, version, func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
		return phases.NewEnsurePreUpgradeResourcesWithComponentVersion(ctx, *componentVersion.DeepCopy())
	})
	return nil
}

// ResolveInlineUpgrade resolves an inline handler from the catalog and prepares version context on ctx.
func ResolveInlineUpgrade(f *ComponentFactory, handler, version string, ctx *phaseframe.PhaseContext) (phaseframe.Phase, error) {
	if f == nil {
		return nil, fmt.Errorf("component factory is nil")
	}
	if ctx != nil && ctx.VersionContext == nil && ctx.BKECluster != nil {
		ctx.BuildAndSetVersionContext()
	}
	return f.Resolve(handler, version, ctx)
}
