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

package auth

import (
	"testing"

	"github.com/gorilla/websocket"
)

func TestWebSocket(t *testing.T) {
	conn := &websocket.Conn{}
	AddWebSocketConnection("123", conn)
	_, exists := GetWebSocketConnection("321")
	if exists {
		t.Errorf("Expect ws 321 exists = false, got true")
	}
	_, exists = GetWebSocketConnection("123")
	if !exists {
		t.Errorf("Expect ws 123 exists = true, got false")
	}
	RemoveWebSocketConnection("123")
}
