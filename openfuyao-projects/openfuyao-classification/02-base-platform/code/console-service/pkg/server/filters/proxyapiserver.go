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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"time"

	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/client-go/kubernetes"

	"console-service/pkg/client/k8s"
	"console-service/pkg/server/request"
	"console-service/pkg/utils/httputil"
	"console-service/pkg/utils/iputil"
	"console-service/pkg/utils/multiclusterutil"
	"console-service/pkg/utils/util"
	"console-service/pkg/zlog"
)

type apiServerProxy struct {
	nextHandler              http.Handler
	kubeUrl                  *url.URL
	roundTripper             http.RoundTripper
	multiClusterRoundTripper http.RoundTripper
}

// ProxyAPIServer proxies requests to kubernetes api server
func ProxyAPIServer(handler http.Handler, client k8s.Client) http.Handler {
	kubeUrl, err := url.Parse(client.Config().Host)
	if err != nil {
		zlog.Fatalf("Failed to parse kubenetes host url: %v", err)
	}
	roundTripper, err := getApiServerTransport(client.Kubernetes(), false)
	if err != nil {
		zlog.Fatalf("Failed to get single cluster api server transport: %v", err)
	}
	multiClusterRoundTripper, err := getApiServerTransport(client.Kubernetes(), true)
	if err != nil {
		zlog.Fatalf("Failed to get multi cluster api server transport: %v", err)
	}

	return &apiServerProxy{
		nextHandler:              handler,
		kubeUrl:                  kubeUrl,
		roundTripper:             roundTripper,
		multiClusterRoundTripper: multiClusterRoundTripper,
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
		if req.Method == http.MethodDelete || req.Method == http.MethodPut || req.Method == http.MethodGet {
			zlog.Infof(`%s - - [%s] %dms "%s %s"`,
				iputil.RemoteIp(req),
				time.Now().Format("02/Jan/2006:15:04:05 -0700"),
				time.Since(time.Now()).Milliseconds(),
				req.Method,
				req.RequestURI,
			)
		}
		// check if it's a multi-cluster request
		if multiclusterutil.IsMultiClusterRequest(info) {
			k.proxyMultiCluster(w, req, info)
		} else {
			k.proxySingleCluster(w, req)
		}
		return
	}

	k.nextHandler.ServeHTTP(w, req)
}

func (k apiServerProxy) proxySingleCluster(w http.ResponseWriter, req *http.Request) {
	req.URL.Scheme = k.kubeUrl.Scheme
	req.URL.Host = k.kubeUrl.Host
	apiProxy := proxy.NewUpgradeAwareHandler(req.URL, k.roundTripper, true, false, &responder{})
	apiProxy.UpgradeTransport = proxy.NewUpgradeRequestRoundTripper(k.roundTripper, k.roundTripper)
	apiProxy.ServeHTTP(w, req)
}

func (k apiServerProxy) proxyMultiCluster(w http.ResponseWriter, req *http.Request, info *request.RequestInfo) {
	req.URL.Scheme = info.ClusterProxyScheme
	req.URL.Host = info.ClusterProxyHost
	req.URL.Path = path.Join(info.ClusterProxyURL, req.URL.Path)
	apiProxy := proxy.NewUpgradeAwareHandler(req.URL, k.multiClusterRoundTripper, true, false, &responder{})
	apiProxy.UpgradeTransport = proxy.NewUpgradeRequestRoundTripper(k.multiClusterRoundTripper,
		k.multiClusterRoundTripper)
	apiProxy.ServeHTTP(w, req)
}

func getApiServerTransport(client kubernetes.Interface, isMultiCluster bool) (http.RoundTripper, error) {
	csConfig, err := util.GetConsoleServiceConfig(client)

	insecureSkipVerify := true
	if err != nil {
		zlog.Warnf("read console-service-config config map failed, reading default set")
	} else {
		insecureSkipVerify = csConfig.InsecureSkipVerify == "true"
	}

	if !insecureSkipVerify && !isMultiCluster {
		k8sRootCAData, err := httputil.GetK8sRootCA(client)
		if err != nil {
			zlog.Error("Failed to get Kubernetes CA cert")
			return nil, err
		}

		CACertPool := x509.NewCertPool()
		if !CACertPool.AppendCertsFromPEM(k8sRootCAData) {
			zlog.Error("Failed to append Kubernetes CA cert to CA CertPool")
			return nil, fmt.Errorf("failed to append Kubernetes CA cert to CA CertPool")
		}

		roundTripper := &http.Transport{TLSClientConfig: &tls.Config{
			InsecureSkipVerify: false,
			MinVersion:         tls.VersionTLS12,
			MaxVersion:         tls.VersionTLS13,
			RootCAs:            CACertPool,
		}}

		return roundTripper, nil
	}

	return &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
			MinVersion:         tls.VersionTLS12,
			MaxVersion:         tls.VersionTLS13,
		},
	}, nil
}
