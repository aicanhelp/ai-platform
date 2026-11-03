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

package prefixcache

import (
	"testing"
	"time"
)

func TestProducerTypedNameAccessor(t *testing.T) {
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
	})

	typedName := producer.TypedName()
	if typedName.Name != "producer" {
		t.Fatalf("expected plugin name producer, got %q", typedName.Name)
	}
	if typedName.Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, typedName.Type)
	}
}
