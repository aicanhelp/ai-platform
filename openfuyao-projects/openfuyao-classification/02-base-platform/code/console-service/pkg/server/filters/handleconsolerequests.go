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

package filters

import (
	"net/http"
	"path"
	"strings"

	"console-service/pkg/constant"
	"console-service/pkg/server/runtime"
	"console-service/pkg/zlog"
)

type consoleRequestHandler struct {
	consoleHandler    http.Handler
	nonConsoleHandler http.Handler
}

// HandleConsoleRequests checks whether the request is a console request,
// If so, skip the chain and handle it with server container
func HandleConsoleRequests(console, nonConsole http.Handler) http.Handler {
	return &consoleRequestHandler{
		consoleHandler:    console,
		nonConsoleHandler: nonConsole,
	}
}

func isConsoleRequest(pathname string) bool {
	appApiPath := path.Join(runtime.RestRootPath, constant.ConsoleServiceDefaultHost)
	consolePluginPrefix := path.Join(
		runtime.RestRootPath, constant.ConsolePluginDefaultHost, constant.ConsoleServiceDefaultAPIVersion,
		"consoleplugins")
	return strings.HasPrefix(pathname, appApiPath) && !strings.HasPrefix(pathname, consolePluginPrefix)
}

// ServeHTTP handles request to HandleConsoleRequests
func (ah consoleRequestHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if isConsoleRequest(req.URL.Path) {
		zlog.Infof("console request %s", req.URL.Path)
		ah.consoleHandler.ServeHTTP(w, req)
	} else {
		ah.nonConsoleHandler.ServeHTTP(w, req)
	}
}
