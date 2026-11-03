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
	"sync"

	"github.com/gorilla/websocket"

	"console-service/pkg/zlog"
)

// WebSocketConnection 连接结构体
type WebSocketConnection struct {
	Conn *websocket.Conn
}

// 用于存储 WebSocket 连接的 map
var wsStore map[string]*WebSocketConnection

// 用来确保 wsStore 是单例模式的
var once sync.Once

// GetWsStore 获取单例的 wsStore
func GetWsStore() map[string]*WebSocketConnection {
	once.Do(func() {
		// 初始化 wsStore 只会执行一次
		wsStore = make(map[string]*WebSocketConnection)
	})
	return wsStore
}

// AddWebSocketConnection 向 wsStore 添加 WebSocket 连接
func AddWebSocketConnection(wsID string, conn *websocket.Conn) {
	wsStore := GetWsStore()
	if wsStore == nil {
		zlog.Errorf("wsStore is nil before assigning value")
		return
	}
	wsStore[wsID] = &WebSocketConnection{
		Conn: conn,
	}
}

// RemoveWebSocketConnection 从 wsStore 删除 WebSocket 连接
func RemoveWebSocketConnection(wsID string) {
	wsStore := GetWsStore()
	delete(wsStore, wsID)
}

// GetWebSocketConnection 获取 WebSocket 连接
func GetWebSocketConnection(wsID string) (*WebSocketConnection, bool) {
	wsStore := GetWsStore()
	conn, exists := wsStore[wsID]
	if !exists {
		return nil, false
	}
	return conn, exists
}
