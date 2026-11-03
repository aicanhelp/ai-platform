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

package dagexec

import (
	"context"
	"testing"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

type gateStubExecutor struct {
	typ ComponentType
}

func (e gateStubExecutor) GetComponentType() ComponentType { return e.typ }
func (e gateStubExecutor) ExecuteComponent(context.Context, *topology.ComponentNode, *ExecutionContext) error {
	return nil
}

func TestWithFeatureGateExecutors_OffClears(t *testing.T) {
	cfg := WithFeatureGateExecutors(Config{}, false,
		gateStubExecutor{typ: ComponentTypeYAML},
		gateStubExecutor{typ: ComponentTypeHelm},
	)
	if cfg.YamlExecutor != nil || cfg.HelmExecutor != nil {
		t.Fatal("gate off must clear executors")
	}
	sched := NewScheduler(cfg)
	if sched.Registry.Has(ComponentTypeYAML) || sched.Registry.Has(ComponentTypeHelm) {
		t.Fatal("gate off must not register yaml/helm")
	}
}

func TestWithFeatureGateExecutors_OnInjects(t *testing.T) {
	cfg := WithFeatureGateExecutors(Config{}, true,
		gateStubExecutor{typ: ComponentTypeYAML},
		gateStubExecutor{typ: ComponentTypeHelm},
	)
	sched := NewScheduler(cfg)
	if !sched.Registry.Has(ComponentTypeYAML) || !sched.Registry.Has(ComponentTypeHelm) {
		t.Fatal("gate on should register provided executors")
	}
}

func TestWithFeatureGateExecutors_OnNilStaysLegacy(t *testing.T) {
	cfg := WithFeatureGateExecutors(Config{}, true, nil, nil)
	sched := NewScheduler(cfg)
	if sched.Registry.Has(ComponentTypeYAML) || sched.Registry.Has(ComponentTypeHelm) {
		t.Fatal("nil executors must not register")
	}
}
