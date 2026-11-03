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

package prerequest

import (
	"testing"

	"hermes-router/pkg/epp/internal/testutil"
)

func TestHeaderHandlerFactoryCreatesNamedPlugin(t *testing.T) {
	plugin, err := HeaderHandlerFactory("pd-headers", nil, testutil.NewHandle())
	if err != nil {
		t.Fatalf("header handler factory: %v", err)
	}
	handler, ok := plugin.(*HeaderHandler)
	if !ok {
		t.Fatalf("expected *HeaderHandler, got %T", plugin)
	}
	typedName := handler.TypedName()
	if typedName.Name != "pd-headers" {
		t.Fatalf("expected plugin name pd-headers, got %q", typedName.Name)
	}
	if typedName.Type != HeaderHandlerType {
		t.Fatalf("expected plugin type %q, got %q", HeaderHandlerType, typedName.Type)
	}
}

func TestNewHeaderHandlerDefaults(t *testing.T) {
	handler := NewHeaderHandler()
	if handler.TypedName().Name != "" {
		t.Fatalf("expected empty default name, got %q", handler.TypedName().Name)
	}
	if got := handler.WithName("named").TypedName().Name; got != "named" {
		t.Fatalf("expected name to be set via WithName, got %q", got)
	}
}
