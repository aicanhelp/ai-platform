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
	"testing"

	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

type stubPhase struct {
	phaseframe.BasePhase
}

func (s *stubPhase) Execute() (ctrl.Result, error) {
	return ctrl.Result{}, nil
}

func TestComponentFactory_RegisterResolve(t *testing.T) {
	f := NewComponentFactory()
	f.Register("EnsureEtcdUpgrade", upgrade.InlineHandlerVersion, func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
		return &stubPhase{BasePhase: phaseframe.NewBasePhase(ctx, confv1beta1.BKEClusterPhase("EnsureEtcdUpgrade"))}
	})

	ctx := phaseframe.NewReconcilePhaseCtx(t.Context())
	phase, err := f.Resolve("EnsureEtcdUpgrade", upgrade.InlineHandlerVersion, ctx)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if phase.Name() != "EnsureEtcdUpgrade" {
		t.Fatalf("unexpected phase name: %s", phase.Name())
	}
}

func TestComponentFactory_ResolveNotFound(t *testing.T) {
	f := NewComponentFactory()
	_, err := f.Resolve("missing", upgrade.InlineHandlerVersion, phaseframe.NewReconcilePhaseCtx(t.Context()))
	if err == nil {
		t.Fatal("expected error for missing component")
	}
}

func TestComponentFactory_Lookup(t *testing.T) {
	f := NewComponentFactory()
	f.Register("EnsureEtcdUpgrade", upgrade.InlineHandlerVersion, func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
		return &stubPhase{BasePhase: phaseframe.NewBasePhase(ctx, "EnsureEtcdUpgrade")}
	})
	inst, err := f.Lookup("EnsureEtcdUpgrade", upgrade.InlineHandlerVersion)
	if err != nil || inst.Name != "EnsureEtcdUpgrade" || inst.Version != upgrade.InlineHandlerVersion {
		t.Fatalf("lookup: inst=%+v err=%v", inst, err)
	}
}
