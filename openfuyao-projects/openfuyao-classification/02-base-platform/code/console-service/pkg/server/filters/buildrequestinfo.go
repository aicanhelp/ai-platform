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
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"console-service/pkg/server/request"
	"console-service/pkg/zlog"
)

type requestInfoBuilder struct {
	nextHandler http.Handler
	resolver    request.RequestInfoResolver
	k8sClient   *kubernetes.Clientset
}

// BuildRequestInfo builds new requestinfo
func BuildRequestInfo(handler http.Handler, requestInfoResolver request.RequestInfoResolver,
	config *rest.Config) http.Handler {
	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		zlog.Error("Fail to initialize client set")
		return &requestInfoBuilder{
			nextHandler: handler,
			resolver:    requestInfoResolver,
			k8sClient:   nil,
		}
	}
	return &requestInfoBuilder{
		nextHandler: handler,
		resolver:    requestInfoResolver,
		k8sClient:   clientSet,
	}
}

// ServeHTTP handles the HTTP request by adding request information to the context.
// 到这一步 /clusters/{cluster}/api/kubernetes 或者 /clusters/{cluster}/rest 两种前缀
func (r *requestInfoBuilder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	infoCtx := req.Context()

	// create request info
	requestInfo, err := r.resolver.NewRequestInfo(req, r.k8sClient)
	if err != nil {
		responsewriters.InternalError(w, req, fmt.Errorf("creating RequestInfo failed: %v", err))
		return
	}

	// create http request
	req = req.WithContext(request.WithRequestInfo(infoCtx, requestInfo))

	// go to the next handler
	r.nextHandler.ServeHTTP(w, req)
}
