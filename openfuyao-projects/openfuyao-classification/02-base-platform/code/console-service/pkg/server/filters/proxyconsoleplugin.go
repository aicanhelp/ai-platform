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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"path"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"console-service/pkg/constant"
	"console-service/pkg/plugin"
	"console-service/pkg/server/request"
	httputil2 "console-service/pkg/utils/httputil"
	"console-service/pkg/utils/multiclusterutil"
	"console-service/pkg/zlog"
)

type consolePluginProxy struct {
	config            *rest.Config
	dynamicClient     *dynamic.DynamicClient
	k8sClient         kubernetes.Interface
	nextHandler       http.Handler
	multiClusterProxy *httputil.ReverseProxy
}

const (
	proxyMinLength        = 2
	consolePluginUrl      = "/apis/console.openfuyao.com/v1beta1/consoleplugins"
	consolePluginResource = "consoleplugin"

	listClustersUrl = "/rest/multicluster/v1beta1/resources/clusters"

	isPluginMjs     = 0
	isPluginBackend = 1
	notPlugin       = 2
)

// ProxyConsolePlugin proxies requests for console plugin resources
func ProxyConsolePlugin(handler http.Handler, config *rest.Config) http.Handler {
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		zlog.Error("Fail to start dynamic client")
		return handler
	}

	k8sClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		zlog.Error("Fail to start k8s client")
		return handler
	}

	pluginProxy := &consolePluginProxy{
		config:        config,
		k8sClient:     k8sClient,
		dynamicClient: dynamicClient,
		nextHandler:   handler,
	}

	// proxy multi cluster requests
	multiClusterHost := fmt.Sprintf("%s://%s", constant.MultiClusterProxyScheme, constant.MultiClusterProxyHost)
	pluginProxy.multiClusterProxy = httputil.NewSingleHostReverseProxy(parseHost(multiClusterHost))
	pluginProxy.multiClusterProxy.Transport = &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}

	return pluginProxy
}

func (cp *consolePluginProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	info, ok := request.RequestInfoFrom(req.Context())
	if !ok {
		responsewriters.InternalError(w, req, fmt.Errorf("no RequestInfo found in the context"))
		return
	}

	status := checkConsolePluginType(req.URL.Path)
	if status == notPlugin {
		cp.nextHandler.ServeHTTP(w, req)
		return
	}

	pathParts := splitPath(req.URL.Path)
	if len(pathParts) < proxyMinLength {
		zlog.Error("Plugin name not specified")
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	// in isPluginMjs and isPluginBackend pluginName both resides at the 1st place
	pluginName := pathParts[1]
	zlog.Infof("ConsolePlugin Proxy: %s", pluginName)
	// for multicluster plugin we only proxy to current cluster
	if pluginName == constant.MultiClusterPluginName {
		info.SetSingleCluster()
	}
	requestUrl := preparePluginRequestURLByCluster(info, consolePluginUrl, pluginName)
	consolePlugin, err := getConsolePlugin(req, requestUrl, cp.k8sClient)
	if err != nil {
		// here err != nil means the multi-cluster plugin is not installed, need to return dummy "host" cluster
		if pluginName == constant.MultiClusterPluginName && req.URL.Path == listClustersUrl {
			returnDummyHostCluster(w)
			return
		}
		zlog.Errorf("Fail to get %s", pluginName)
		w.WriteHeader(http.StatusNotFound)
		return
	}

	if status == isPluginMjs {
		req.URL.Path = strings.TrimPrefix(req.URL.Path, "/proxy/"+pluginName)
	}
	if multiclusterutil.IsMultiClusterRequest(info) {
		req.URL.Path = path.Join(info.ClusterProxyURL,
			getPluginApiServerProxyPathPrefix(consolePlugin), req.URL.Path)
		cp.multiClusterProxy.ServeHTTP(w, req)
	} else {
		proxyUrl := getSingleClusterConsolePluginProxyUrl(consolePlugin)
		proxySingleClusterConsolePluginRequest(proxyUrl, consolePlugin, w, req)
	}
	return
}

func returnDummyHostCluster(w http.ResponseWriter) {
	dummyClusterList := multiclusterutil.ReturnDummyHostCluster()
	w.WriteHeader(http.StatusOK)
	err := json.NewEncoder(w).Encode(dummyClusterList)
	if err != nil {
		zlog.Errorf("cannot return dummy cluster obj, err: %v", err)
	}
	return
}

func addAuthorizationHeader(req *http.Request, extReq *http.Request) {
	// add authorization header
	authInfo := req.Header.Get("Authorization")
	if authInfo != "" {
		zlog.Info("Successfully retrieve token from request")
	}
	extReq.Header.Set("Authorization", authInfo)

	authInfo = req.Header.Get(constant.OpenFuyaoAuthHeader)
	if authInfo != "" {
		zlog.Info("Successfully retrieve openfuyao token from request")
	}
	extReq.Header.Set(constant.OpenFuyaoAuthHeader, authInfo)
}

func getConsolePlugin(oriReq *http.Request, url string, client kubernetes.Interface) (*plugin.ConsolePlugin, error) {
	// Create a new HTTP request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		zlog.Errorf("Error creating request: %v", err)
		return nil, nil
	}

	// Add the Authorization header
	addAuthorizationHeader(oriReq, req)

	resp, err := requestConsolePlugin(req, client)
	if err != nil {
		zlog.Errorf("Error sending request: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	// Read and print the response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		zlog.Errorf("Error reading response: %v", err)
		return nil, err
	}

	// Unmarshal the response into an unstructured.Unstructured object
	var obj unstructured.Unstructured
	err = json.Unmarshal(body, &obj.Object)
	if err != nil {
		zlog.Errorf("Error unmarshaling response into Unstructured: %v", err)
		return nil, err
	}
	var consolePlugin plugin.ConsolePlugin
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &consolePlugin)
	if err != nil {
		return nil, err
	}

	return &consolePlugin, nil
}

func requestConsolePlugin(req *http.Request, client kubernetes.Interface) (*http.Response, error) {
	// Create a custom HTTP client with TLS config to skip certificate verification
	// 如果是单集群request，根据是否skipVerification选择证书，多集群当前必须跳过证书
	httpClient := &http.Client{}
	transport, err := getApiServerTransport(client,
		multiclusterutil.IsMultiClusterRequestURL(req.URL.Host))
	if err != nil {
		return nil, err
	}
	httpClient = &http.Client{
		Transport: transport,
	}

	// Send the HTTP request
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func checkConsolePluginType(path string) int {
	if strings.HasPrefix(path, "/proxy") {
		return isPluginMjs
	} else if (strings.HasPrefix(path, "/rest") || strings.HasPrefix(path, "/ws")) &&
		!strings.HasPrefix(path, "/rest/console") {
		return isPluginBackend
	} else {
		return notPlugin
	}
}

func preparePluginRequestURLByCluster(info *request.RequestInfo, urlPath string, pluginName string) string {
	url := ""
	if multiclusterutil.IsMultiClusterRequest(info) && pluginName != constant.MultiClusterPluginName {
		urlPath = path.Join(info.ClusterProxyURL, urlPath)
		scheme := info.ClusterProxyScheme
		if scheme == "" {
			scheme = constant.MultiClusterProxyScheme
		}
		url = fmt.Sprintf("%s://%s%s/%s", scheme, info.ClusterProxyHost, urlPath, pluginName)
	} else {
		url = fmt.Sprintf("%s://%s%s/%s", constant.SingleClusterProxyScheme, constant.SingleClusterProxyHost,
			urlPath, pluginName)
	}
	return url
}

func splitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}

func getPluginApiServerProxyPathPrefix(cp *plugin.ConsolePlugin) string {
	cpBackend := cp.Spec.Backend

	if cpBackend.Type == plugin.ServiceBackendType {
		service := cpBackend.Service
		name := service.Name
		namespace := service.Namespace
		basePath := service.BasePath
		scheme := service.Scheme
		if scheme == "" {
			scheme = "http"
		}
		port := service.Port
		if port == 0 {
			if scheme == "http" {
				port = constant.DefaultHttpPort
			} else {
				port = constant.DefaultHttpsPort
			}
		}
		proxyPath := constant.ServiceProxyURL
		proxyPath = strings.Replace(proxyPath, "{namespace}", namespace, 1)
		if scheme == "https" {
			proxyPath = strings.Replace(proxyPath, "{service}", scheme+":"+name, 1)
		} else {
			proxyPath = strings.Replace(proxyPath, "{service}", name, 1)
		}
		proxyPath = strings.Replace(proxyPath, "{port}", strconv.Itoa(int(port)), 1)
		return path.Join(proxyPath, basePath)
	}

	return ""
}

func getSingleClusterConsolePluginProxyUrl(cp *plugin.ConsolePlugin) string {
	if cp.Spec.Backend.Type == plugin.ServiceBackendType {
		svc := cp.Spec.Backend.Service
		scheme := svc.Scheme
		if scheme == "" {
			scheme = "http"
		}
		port := svc.Port
		if port == 0 {
			if scheme == "http" {
				port = constant.DefaultHttpPort
			} else {
				port = constant.DefaultHttpsPort
			}
		}
		return fmt.Sprintf("%s://%s.%s.svc:%d%s", scheme, svc.Name, svc.Namespace, port, svc.BasePath)
	}

	return ""
}

func proxySingleClusterConsolePluginRequest(url string, cp *plugin.ConsolePlugin,
	w http.ResponseWriter, r *http.Request,
) {
	// proxy single cluster component requests
	if cp.Spec.Backend.Type != plugin.ServiceBackendType {
		zlog.Errorf("only %s type of ConsolePlugin is supported", plugin.ServiceBackendType)
		w.WriteHeader(http.StatusBadRequest)
		_, err := w.Write([]byte(`{"reason": "unsupported ConsolePlugin backend type"}`))
		if err != nil {
			zlog.Errorf("Failed to write response: %v", err)
		}
		return
	}
	svc := cp.Spec.Backend.Service
	reverseProxy := httputil.NewSingleHostReverseProxy(parseHost(url))
	scheme := svc.Scheme
	if scheme == "" {
		scheme = "http"
	}
	skipVerification := svc.InsecureSkipVerify || scheme == "http"
	transport, err := httputil2.GetCustomizedHttpTransportByRaw(skipVerification, nil, nil, svc.CABundle)
	if err != nil {
		zlog.Errorf("Fail to add console plugin transport for url: %s, err: %v", url, err)
		w.WriteHeader(http.StatusInternalServerError)
		_, err = w.Write([]byte(fmt.Sprintf(`{"reason": "%v"}`, err)))
		if err != nil {
			zlog.Errorf("Failed to write response: %v", err)
		}
		return
	}
	reverseProxy.Transport = transport
	reverseProxy.ServeHTTP(w, r)
}
