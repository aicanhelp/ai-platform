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

package testutil

import "testing"

func TestKeyValuesMapIgnoresInvalidAndUnpairedKeys(t *testing.T) {
	got := keyValuesMap([]any{
		"model", "qwen",
		404, "ignored",
		"latency", 12,
		"requestID",
	})

	if got["model"] != "qwen" {
		t.Fatalf("expected model key to be recorded, got %+v", got)
	}
	if got["latency"] != 12 {
		t.Fatalf("expected latency key to be recorded, got %+v", got)
	}
	if _, ok := got["requestID"]; ok {
		t.Fatalf("expected unpaired key to be ignored, got %+v", got)
	}
	if len(got) != 2 {
		t.Fatalf("expected exactly 2 key-value pairs, got %+v", got)
	}
}
