/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package client

import (
	"sync"
	"testing"

	"k8s.io/client-go/rest"
)

func TestInitKubeClient_SetsSingleton(t *testing.T) {
	// Reset singleton state between tests.
	kubeClientInstance = nil
	once = sync.Once{}

	InitKubeClient(&rest.Config{Host: "https://example"})

	if got := GetKubeClient(); got == nil {
		t.Fatalf("Expected non-nil client")
	}
}

func TestInitKubeClient_OnceOnly(t *testing.T) {
	kubeClientInstance = nil
	once = sync.Once{}

	InitKubeClient(&rest.Config{Host: "https://example"})
	first := GetKubeClient()
	InitKubeClient(&rest.Config{Host: "https://example"})
	second := GetKubeClient()

	if first == nil || second == nil {
		t.Fatalf("Expected non-nil clients")
	}
	if first != second {
		t.Fatalf("Expected singleton instance reused")
	}
}
