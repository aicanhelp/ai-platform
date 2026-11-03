/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package alwaysdisagg

import (
	"context"
	"encoding/json"
	"testing"
)

func TestFactoryAndTypedName(t *testing.T) {
	plugin, err := Factory("always", json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	decider, ok := plugin.(*Decider)
	if !ok {
		t.Fatalf("unexpected type %T", plugin)
	}
	if got := decider.TypedName(); got.Type != PluginType || got.Name != "always" {
		t.Fatalf("TypedName=%+v, want type=%q name=always", got, PluginType)
	}
	if _, err := Factory("always", json.RawMessage(`{"unexpected":1}`), nil); err == nil {
		t.Fatal("expected unknown-field error")
	}
}

func TestDecideAlwaysRunsPrefill(t *testing.T) {
	decider := New("always")
	shouldRun, err := decider.Decide(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected always-disagg decider to run Prefill")
	}
}
