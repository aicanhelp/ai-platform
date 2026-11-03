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
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/client-go/rest"

	"installer-service/pkg/server/request"
	"installer-service/pkg/zlog"
)

type apiServerProxy struct {
	nextHandler  http.Handler
	kubeUrl      *url.URL
	roundTripper http.RoundTripper
}

// ProxyAPIServer proxies requests to kubernetes api server
func ProxyAPIServer(handler http.Handler, config *rest.Config) http.Handler {
	kubeUrl, err := url.Parse(config.Host)
	if err != nil {
		zlog.Errorf("Failed to parse kubenetes host url: %v", err)
		return handler
	}
	roundTripper, err := rest.TransportFor(config)
	if err != nil {
		zlog.Errorf("Failed to create transport for configuration: %v", err)
		return handler
	}
	return &apiServerProxy{
		nextHandler:  handler,
		kubeUrl:      kubeUrl,
		roundTripper: roundTripper,
	}
}

// ServeHTTP handles request to ProxyAPIServer
func (k apiServerProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	info, exist := request.RequestInfoFrom(req.Context())
	if !exist {
		http.Error(w, "RequestInfo not founded in request context", http.StatusInternalServerError)
		k.nextHandler.ServeHTTP(w, req)
		return
	}

	if info.IsK8sRequest {
		zlog.Infof("API Server request %s requested", req.URL.Path)
		req.URL.Path = strings.TrimPrefix(req.URL.Path, "/api/kubernetes")
		req.URL.Scheme = k.kubeUrl.Scheme
		req.URL.Host = k.kubeUrl.Host
		req.Header.Del("Authorization")
		apiProxy := proxy.NewUpgradeAwareHandler(req.URL, k.roundTripper, true, false, &responder{})
		apiProxy.UpgradeTransport = proxy.NewUpgradeRequestRoundTripper(k.roundTripper, k.roundTripper)
		apiProxy.ServeHTTP(w, req)
		return
	}

	k.nextHandler.ServeHTTP(w, req)
}
