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

package autokvaware

import (
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestScorerIdentityAccessors(t *testing.T) {
	scorer := New("auto-kv", Config{})

	typedName := scorer.TypedName()
	if typedName.Name != "auto-kv" {
		t.Fatalf("expected plugin name auto-kv, got %q", typedName.Name)
	}
	if typedName.Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, typedName.Type)
	}
	if scorer.Category() != fwkscheduling.Balance {
		t.Fatalf("expected balance category, got %q", scorer.Category())
	}
}
