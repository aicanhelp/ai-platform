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

type stubExecutor struct {
	typ      ComponentType
	executed int
}

func (e *stubExecutor) GetComponentType() ComponentType { return e.typ }

func (e *stubExecutor) ExecuteComponent(
	_ context.Context,
	_ *topology.ComponentNode,
	_ *ExecutionContext,
) error {
	e.executed++
	return nil
}

func TestExecutorRegistry_RegisterGetHas(t *testing.T) {
	reg := NewExecutorRegistry()
	inline := &stubExecutor{typ: ComponentTypeInline}
	yaml := &stubExecutor{typ: ComponentTypeYAML}

	if err := reg.Register(inline); err != nil {
		t.Fatalf("register inline: %v", err)
	}
	if err := reg.Register(yaml); err != nil {
		t.Fatalf("register yaml: %v", err)
	}

	if !reg.Has(ComponentTypeInline) || !reg.Has(ComponentTypeYAML) {
		t.Fatalf("expected inline and yaml to be registered")
	}
	if reg.Has(ComponentTypeHelm) {
		t.Fatalf("helm must not be registered")
	}

	got, ok := reg.Get(ComponentTypeInline)
	if !ok || got != inline {
		t.Fatalf("Get(inline) miss: ok=%v got=%v", ok, got)
	}
	got, ok = reg.Get(ComponentTypeYAML)
	if !ok || got != yaml {
		t.Fatalf("Get(yaml) miss: ok=%v got=%v", ok, got)
	}
	if _, ok := reg.Get(ComponentTypeHelm); ok {
		t.Fatalf("Get(helm) should miss")
	}
}

func TestExecutorRegistry_RegisterOverwrite(t *testing.T) {
	reg := NewExecutorRegistry()
	first := &stubExecutor{typ: ComponentTypeInline}
	second := &stubExecutor{typ: ComponentTypeInline}
	if err := reg.Register(first); err != nil {
		t.Fatal(err)
	}
	if err := reg.Register(second); err != nil {
		t.Fatal(err)
	}
	got, ok := reg.Get(ComponentTypeInline)
	if !ok || got != second {
		t.Fatalf("expected overwrite with second executor")
	}
}

func TestExecutorRegistry_RegisterErrors(t *testing.T) {
	var nilReg *ExecutorRegistry
	if err := nilReg.Register(&stubExecutor{typ: ComponentTypeInline}); err == nil {
		t.Fatalf("expected error registering on nil registry")
	}

	reg := NewExecutorRegistry()
	if err := reg.Register(nil); err == nil {
		t.Fatalf("expected error registering nil executor")
	}
	if err := reg.Register(&stubExecutor{typ: ""}); err == nil {
		t.Fatalf("expected error registering empty type")
	}
}

func TestExecutorRegistry_NilSafeLookup(t *testing.T) {
	var nilReg *ExecutorRegistry
	if nilReg.Has(ComponentTypeInline) {
		t.Fatalf("nil registry Has should be false")
	}
	if _, ok := nilReg.Get(ComponentTypeInline); ok {
		t.Fatalf("nil registry Get should miss")
	}
}
