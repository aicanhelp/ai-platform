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
	"net/http/httputil"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	httputil2 "console-service/pkg/utils/httputil"
	"console-service/pkg/utils/util"
	"console-service/pkg/zlog"
)

type staticProxy struct {
	proxy *httputil.ReverseProxy
}

const (
	defaultConsoleWebsiteHost = "http://console-website.openfuyao-system.svc.cluster.local"
)

// ProxyStatic proxies static resource requests to console-website,
// this is the last part of handler chain, hence has no nextHandler
func ProxyStatic(handler http.Handler, config *rest.Config) http.Handler {
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		zlog.Errorf("error creating client set, err: %v", err)
		return handler
	}

	var consoleWebsiteHost string
	csConfig, err := util.GetConsoleServiceConfig(clientset)
	insecureSkipVerify := true
	if err != nil {
		zlog.Warnf("read console-service-config config map failed, reading default set")
		consoleWebsiteHost = defaultConsoleWebsiteHost
	} else {
		consoleWebsiteHost = csConfig.ConsoleWebsiteHost
		insecureSkipVerify = csConfig.InsecureSkipVerify == "true"
	}

	target := parseHost(consoleWebsiteHost)
	proxy := httputil.NewSingleHostReverseProxy(target)
	transport, err := httputil2.GetHttpTransport(!insecureSkipVerify)
	proxy.Transport = transport
	if err != nil {
		zlog.Warn("Fail to add https transport for static resource proxy, use http")
	}
	return &staticProxy{
		proxy: proxy,
	}
}

// ServeHTTP handles request to ProxyOAuthRequest
func (sp staticProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	zlog.Infof("Static proxy %s requested", req.URL.Path)
	sp.proxy.ServeHTTP(w, req)
}
