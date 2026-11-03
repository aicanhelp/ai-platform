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

package v1beta1

import (
	"testing"

	"github.com/emicklei/go-restful/v3"
)

func TestNewHandler(t *testing.T) {
	handler := NewHandler()
	if handler == nil {
		t.Fatal("NewHandler() returned nil")
	}
}

func TestBindUtilityWebService(t *testing.T) {
	ws := new(restful.WebService)
	BindUtilityWebService(ws)
	if len(ws.Routes()) != 1 {
		t.Errorf("BindUtilityWebService() routes = %d, want 1", len(ws.Routes()))
	}
	if ws.Routes()[0].Method != "GET" {
		t.Errorf("BindUtilityWebService() method = %q, want GET", ws.Routes()[0].Method)
	}
}
