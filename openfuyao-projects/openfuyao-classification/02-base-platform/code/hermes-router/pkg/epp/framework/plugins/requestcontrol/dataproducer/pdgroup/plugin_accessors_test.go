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
	"testing"
)

func TestProducerIdentityAccessors(t *testing.T) {
	producer := New("producer", Config{
		RoleLabelName:  "role",
		GroupLabelName: "group",
		PrefillValue:   "prefill",
		DecodeValue:    "decode",
		LeaderValue:    "leader",
	})

	typedName := producer.TypedName()
	if typedName.Name != "producer" {
		t.Fatalf("expected plugin name producer, got %q", typedName.Name)
	}
	if typedName.Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, typedName.Type)
	}
	if produced := producer.Produces(); produced == nil {
		t.Fatal("expected non-nil produced keys")
	}
	if consumed := producer.Consumes(); len(consumed) != 0 {
		t.Fatalf("expected no consumed keys, got %+v", consumed)
	}
}
