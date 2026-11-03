/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package pdgroup

import (
	"encoding/json"
	"testing"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

	"hermes-router/pkg/epp/internal/testutil"
)

func TestFactoryAcceptsParameterlessConfiguration(t *testing.T) {
	for _, raw := range []json.RawMessage{nil, []byte(`{}`)} {
		plugin, err := Factory("pd-filter", raw, testutil.NewHandle())
		if err != nil {
			t.Fatalf("factory with raw %s: %v", raw, err)
		}
		filter, ok := plugin.(*Filter)
		if !ok {
			t.Fatalf("expected *Filter, got %T", plugin)
		}
		typedName := filter.TypedName()
		if typedName.Name != "pd-filter" || typedName.Type != PluginType {
			t.Fatalf("unexpected typed name %+v", typedName)
		}
	}
}

func TestFactoryRejectsUnknownParameters(t *testing.T) {
	if _, err := Factory("pd-filter", []byte(`{"unexpected":true}`), testutil.NewHandle()); err == nil {
		t.Fatal("expected unknown parameters to be rejected")
	}
}

func TestFactoryRejectsInvalidJSON(t *testing.T) {
	if _, err := Factory("pd-filter", []byte(`{invalid`), testutil.NewHandle()); err == nil {
		t.Fatal("expected invalid JSON to be rejected")
	}
}

func TestNewSetsTypedName(t *testing.T) {
	typedName := New("leader-filter").TypedName()
	if typedName.Name != "leader-filter" {
		t.Fatalf("expected plugin name leader-filter, got %q", typedName.Name)
	}
	if typedName.Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, typedName.Type)
	}
}

var _ fwkplugin.Plugin = (*Filter)(nil)
