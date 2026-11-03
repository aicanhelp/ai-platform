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

package inflight

import (
	"testing"

	"hermes-router/pkg/epp/internal/inflight"
)

func TestLoadProducerTypedNameAccessor(t *testing.T) {
	producer := NewLoadProducer("producer", inflight.StoreForName("accessor-test"), inflight.DefaultAttributeKey)

	typedName := producer.TypedName()
	if typedName.Name != "producer" {
		t.Fatalf("expected plugin name producer, got %q", typedName.Name)
	}
	if typedName.Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, typedName.Type)
	}
}
