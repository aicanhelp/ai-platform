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

// Package filters handle proxy
package filters

import (
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"

	"installer-service/pkg/server/request"
)

type requestInfoBuilder struct {
	nextHandler http.Handler
	resolver    request.RequestInfoResolver
}

// BuildRequestInfo build request structural
func BuildRequestInfo(handler http.Handler, requestInfoResolver request.RequestInfoResolver) http.Handler {
	return &requestInfoBuilder{
		nextHandler: handler,
		resolver:    requestInfoResolver,
	}
}

// ServeHTTP handles the HTTP request by adding request information to the context.
func (r *requestInfoBuilder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	infoCtx := req.Context()

	// create request info
	requestInfo, err := r.resolver.NewRequestInfo(req)
	if err != nil {
		errorMessage := fmt.Sprintf("RequestInfo create failed: %v", err)
		responsewriters.InternalError(w, req, fmt.Errorf("%s", errorMessage))
		return
	}

	// create http request
	req = req.WithContext(request.WithRequestInfo(infoCtx, requestInfo))

	// go to the next handler
	r.nextHandler.ServeHTTP(w, req)
}
