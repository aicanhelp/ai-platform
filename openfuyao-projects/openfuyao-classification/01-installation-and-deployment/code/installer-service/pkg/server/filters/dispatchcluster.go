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
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"

	"installer-service/pkg/server/request"
)

type clusterDispatcher struct {
	nextHandler http.Handler
}

// DispatchCluster 处理多集群分发，根据集群名称进行路由分发
func DispatchCluster(handler http.Handler) http.Handler {
	return &clusterDispatcher{
		nextHandler: handler,
	}
}

// ServeHTTP handles request to DispatchCluster
func (m *clusterDispatcher) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	info, ok := request.RequestInfoFrom(req.Context())
	if !ok {
		responsewriters.InternalError(w, req, fmt.Errorf("no RequestInfo found in the context"))
		return
	}
	if info.Cluster == "" {
		m.nextHandler.ServeHTTP(w, req)
		return
	}

	m.nextHandler.ServeHTTP(w, req)
}
