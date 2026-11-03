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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/golang-jwt/jwt/v4"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"console-service/pkg/auth"
	"console-service/pkg/client/k8s"
	"console-service/pkg/constant"
	"console-service/pkg/plugin"
	"console-service/pkg/server/request"
	httputil2 "console-service/pkg/utils/httputil"
	"console-service/pkg/utils/multiclusterutil"
	"console-service/pkg/utils/util"
	"console-service/pkg/zlog"
)

type componentProxy struct {
	kubeConfig         *rest.Config
	clientset          kubernetes.Interface
	oauthProxy         *httputil.ReverseProxy
	multiClusterProxy  *httputil.ReverseProxy
	consolePluginProxy *httputil.ReverseProxy

	insecureSkipVerify      bool
	alertInsecureSkipVerify bool

	componentHost         map[string]string
	singleClusterProxyMap map[string]*httputil.ReverseProxy

	nextHandler    http.Handler
	consoleHandler http.Handler
}

// AlertSilenceRequest structure
type AlertSilenceRequest struct {
	StartsAt  string    `json:"startsAt,omitempty"`
	EndsAt    string    `json:"endsAt,omitempty"`
	CreatedBy string    `json:"createdBy,omitempty"`
	Comment   string    `json:"comment,omitempty"`
	Matchers  []Matcher `json:"matchers,omitempty"`
}

// Matcher structure
type Matcher struct {
	Name    string `json:"name,omitempty"`
	Value   string `json:"value,omitempty"`
	IsRegex bool   `json:"isRegex,omitempty"`
	IsEqual bool   `json:"isEqual,omitempty"`
}

// JWTAccessClaims structure
type JWTAccessClaims struct {
	jwt.StandardClaims
}

const (
	// api prefix
	alertApiPrefix                = "/rest/alert"
	monitoringApiPrefix           = "/rest/monitoring/"
	webTerminalApiPrefix          = "/rest/webterminal/"
	pluginApiPrefix               = "/rest/plugin-management/"
	applicationApiPrefix          = "/rest/application-management/"
	marketApiPrefix               = "/rest/marketplace/"
	userManagementApiPrefix       = "/rest/user/"
	webPasswordPrefix             = "/password"
	oauthServerPrefix             = "/auth"
	oauthPasswordIdentityProvider = "fuyaoPasswordProvider"
)

// ProxyComponentRequest proxies requests to openFuyao components other than oauth and APIServer,
// otherwise passes the request to the next handler
func ProxyComponentRequest(consoleHandler http.Handler, nextHandler http.Handler, client k8s.Client) http.Handler {
	proxy := &componentProxy{
		kubeConfig:         client.Config(),
		clientset:          client.Kubernetes(),
		insecureSkipVerify: true,

		componentHost:         make(map[string]string),
		singleClusterProxyMap: make(map[string]*httputil.ReverseProxy),
		nextHandler:           nextHandler,
		consoleHandler:        consoleHandler,
	}
	proxy.setComponentProxyHosts()
	proxy.setSingleClusterProxy()
	multiClusterHost := proxy.setMultiClusterProxy(true)
	proxy.setConsolePluginProxy(multiClusterHost, true)

	// oauthProxy and staticProxy only proxy to the current cluster service and without kube-apiserver proxy
	oauthProxy := httputil.NewSingleHostReverseProxy(parseHost(proxy.componentHost["oauth"]))
	transport, err := httputil2.GetHttpTransport(!proxy.insecureSkipVerify)
	if err != nil {
		zlog.Warn("Fail to add https transport for oauth proxy, use http")
	}
	oauthProxy.Transport = transport
	proxy.oauthProxy = oauthProxy

	proxy.singleClusterProxyMap["oauth"] = oauthProxy

	return proxy
}

func (sp *componentProxy) setComponentProxyHosts() {
	csConfig, err := util.GetConsoleServiceConfig(sp.clientset)

	if err != nil {
		zlog.Warnf("read console-service-config config map failed, reading default set")
	} else {
		sp.insecureSkipVerify = csConfig.InsecureSkipVerify == "true"
		hosts := map[string]string{
			"alert":          csConfig.AlertHost,
			"monitoring":     csConfig.MonitoringHost,
			"webTerminal":    csConfig.WebTerminalHost,
			"oauth":          csConfig.OAuthServerHost,
			"application":    csConfig.ApplicationHost,
			"plugin":         csConfig.PluginHost,
			"marketPlace":    csConfig.MarketPlaceHost,
			"userManagement": csConfig.UserManagementHost,
		}

		for key, host := range hosts {
			sp.componentHost[key] = host
		}
	}
	return
}

func (sp *componentProxy) setConsolePluginProxy(multiClusterHost string, insecureSkipVerify bool) {
	// proxy console-plugin requests, using the same host as multiClusterHost
	sp.consolePluginProxy = httputil.NewSingleHostReverseProxy(parseHost(multiClusterHost))
	transport, err := httputil2.GetHttpTransport(!insecureSkipVerify)
	if err != nil {
		zlog.Warn("Fail to add https transport for oauth proxy, use http")
	}
	sp.consolePluginProxy.Transport = transport
}

func (sp *componentProxy) setMultiClusterProxy(insecureSkipVerify bool) string {
	// proxy multi cluster requests
	multiClusterHost := fmt.Sprintf("%s://%s", constant.MultiClusterProxyScheme, constant.MultiClusterProxyHost)
	sp.multiClusterProxy = httputil.NewSingleHostReverseProxy(parseHost(multiClusterHost))
	transport, err := httputil2.GetHttpTransport(!insecureSkipVerify)
	if err != nil {
		zlog.Warn("Fail to add https transport for multi-cluster proxy, use http")
	}
	sp.multiClusterProxy.Transport = transport
	return multiClusterHost
}

func (sp *componentProxy) setSingleClusterProxy() {
	// proxy single cluster requests
	insecureSkipVerify := sp.insecureSkipVerify
	for key, host := range sp.componentHost {
		sp.setSingleClusterComponentProxy(key, host, insecureSkipVerify)
	}
}

func (sp *componentProxy) setSingleClusterComponentProxy(component, host string, insecureSkipVerify bool) {
	// proxy single cluster component requests
	sp.singleClusterProxyMap[host] = httputil.NewSingleHostReverseProxy(parseHost(host))
	transport, err := httputil2.GetHttpTransport(!insecureSkipVerify)
	if component == "alert" {
		transport, err = httputil2.GetCustomizedHttpTransportByPath(
			constant.AlertTLSCertPath, constant.AlertTLSKeyPath, constant.AlertCAPath)
	}
	if err != nil {
		zlog.Warn("Fail to add https transport for single-cluster proxy, use http")
	}
	sp.singleClusterProxyMap[host].Transport = transport
}

func parseHost(host string) *url.URL {
	if len(host) == 0 {
		zlog.Errorf("Configured host is empty")
		return nil
	}
	target, err := url.Parse(host)
	if err != nil {
		zlog.Errorf("Failed to parse host %s", host)
	}
	return target
}

// ServeHTTP handles request to ProxyComponentRequest
func (sp *componentProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// fetch requestinfo for multi-cluster info
	info, ok := request.RequestInfoFrom(req.Context())
	if !ok {
		responsewriters.InternalError(w, req, fmt.Errorf("no RequestInfo found in the context"))
		return
	}

	if isUserManagementRequest(req.URL.Path) {
		sp.proxyByCluster(w, req, info, sp.componentHost["userManagement"])
		return
	}

	if isApplicationRequest(req.URL.Path) {
		sp.proxyByCluster(w, req, info, sp.componentHost["application"])
		return
	}

	if isPluginRequest(req.URL.Path) {
		sp.proxyConsolePluginManagementByCluster(w, req, info, sp.componentHost["plugin"])
		return
	}

	if isMarketPlaceRequest(req.URL.Path) {
		sp.proxyByCluster(w, req, info, sp.componentHost["marketPlace"])
		return
	}

	if isAlertRequest(req.URL.Path) {
		req.URL.Path = strings.TrimPrefix(req.URL.Path, alertApiPrefix)
		ok := sp.checkCreatedBySessionUser(req)
		if !ok {
			http.Error(w, "CreatedBy user is not valid", http.StatusBadRequest)
			return
		}
		sp.proxyByCluster(w, req, info, sp.componentHost["alert"])
		return
	}

	if isMonitoringRequest(req.URL.Path) {
		sp.proxyByCluster(w, req, info, sp.componentHost["monitoring"])
		return
	}

	if isWebTerminalRequest(req.URL.Path) {
		sp.proxyByCluster(w, req, info, sp.componentHost["webTerminal"])
		return
	}

	if isOAuthRequest(req.URL.Path) {
		sp.modifyOAuthRequestResponse(w, req)
		sp.singleClusterProxyMap["oauth"].ServeHTTP(w, req)
		return
	}

	sp.nextHandler.ServeHTTP(w, req)
}

func (sp *componentProxy) modifyOAuthRequestResponse(w http.ResponseWriter, req *http.Request) {
	req.URL.Path = path.Join("/oauth2"+oauthServerPrefix, req.URL.Path, oauthPasswordIdentityProvider)
	sp.singleClusterProxyMap["oauth"].ModifyResponse = func(response *http.Response) error {
		authHandler, err := auth.NewHandler(sp.kubeConfig)
		if err != nil {
			zlog.Errorf("failed to initialize auth handler %v", err)
			return err
		}
		if response.StatusCode == http.StatusOK {
			zlog.Info("Password reset succeed, proxy to logout handler")
			authHandler.LogoutCore(req, w)
		}
		if response.StatusCode == http.StatusFound {
			zlog.Error("Password reset failed, maximum time reached")
			authHandler.LogoutCore(req, w)
		}
		return nil
	}
}

func (sp *componentProxy) proxyByCluster(w http.ResponseWriter, req *http.Request,
	info *request.RequestInfo, host string,
) {
	if multiclusterutil.IsMultiClusterRequest(info) {
		req.URL.Path = path.Join(sp.convertToServiceProxyPath(host), req.URL.Path)
		req.URL.Path = path.Join(info.ClusterProxyURL, req.URL.Path)
		sp.multiClusterProxy.ServeHTTP(w, req)
	} else {
		proxy, ok := sp.singleClusterProxyMap[host]
		if !ok {
			zlog.Errorf("failed to find proxy for host %s", host)
			return
		}
		proxy.ServeHTTP(w, req)
	}
}

func (sp *componentProxy) proxyConsolePluginManagementByCluster(w http.ResponseWriter, req *http.Request,
	info *request.RequestInfo, host string,
) {
	// proxy consoleplugin management requests to specific cluster
	// 如果是单集群的，直接调用当前的authhandler
	if !multiclusterutil.IsMultiClusterRequest(info) {
		zlog.Infof("single cluster consoleplugin request: %s", req.URL.Path)
		sp.singleClusterProxyMap[host].ServeHTTP(w, req)
		return
	}
	// 如果是 multiclusterrequest, 正常proxy；
	req.URL.Path = path.Join(info.ClusterProxyURL, sp.convertToServiceProxyPath(host), req.URL.Path)
	zlog.Infof("multi-cluster consoleplugin request: %s", req.URL.Path)

	sp.modifyConsolePluginProxyResponse(w, req, info)
	sp.consolePluginProxy.ServeHTTP(w, req)
}

func (sp *componentProxy) modifyConsolePluginProxyResponse(w http.ResponseWriter, req *http.Request,
	info *request.RequestInfo,
) {
	sp.consolePluginProxy.ModifyResponse = func(response *http.Response) error {
		// only check request ending with /consoleplugins
		if !strings.HasSuffix(strings.TrimSuffix(req.URL.Path, "/"), "/consoleplugins") {
			zlog.Infof("request path is %s, not list consoleplugins, passthrough", req.URL.Path)
			return nil
		}

		// Fetch all ConsolePlugin CRDs and build trimmed list with full i18n map
		allPlugins, err := sp.listAllConsolePlugins(req, info)
		if err != nil {
			zlog.Errorf("failed to list all console plugins: %v", err)
			return err
		}

		// For multi-cluster requests from non-host clusters, also get the multicluster plugin from the host cluster
		if multiclusterutil.IsMultiClusterRequest(info) &&
			response.StatusCode != http.StatusForbidden &&
			info.ClusterName != "host" {
			hostPlugin, err := sp.getHostPlugin(req, info,
				constant.MultiClusterPluginName)
			if err != nil {
				zlog.Errorf("failed to get host multicluster plugin: %v", err)
			}
			if hostPlugin != nil {
				allPlugins = append(allPlugins, *hostPlugin)
			}
		}

		respJson := &httputil2.ResponseJson{
			Code: 200,
			Msg:  "success",
			Data: allPlugins,
		}

		newBody, err := json.Marshal(respJson)
		if err != nil {
			zlog.Errorf("failed to marshal updated response: %v", err)
			return err
		}

		// Replace the response body with our modified JSON
		response.Body = io.NopCloser(bytes.NewBuffer(newBody))
		response.ContentLength = int64(len(newBody))
		response.Header.Set("Content-Length", strconv.Itoa(len(newBody)))
		return nil
	}
}

func parseResponse(response *http.Response) (*httputil2.ResponseJson, error) {
	body, err := io.ReadAll(response.Body)
	if err != nil {
		zlog.Errorf("failed to read response body: %v", err)
		return nil, err
	}

	var respJson httputil2.ResponseJson

	zlog.Warnf("mult-cluster cosoleplugin response: %s", string(body))
	err = json.Unmarshal(body, &respJson)
	if err != nil {
		zlog.Errorf("failed to unmarshal response body: %v", err)
		return nil, err
	}
	return &respJson, nil
}

// listAllConsolePlugins fetches all ConsolePlugin CRDs from the target cluster.
func (sp *componentProxy) listAllConsolePlugins(req *http.Request, info *request.RequestInfo) (
	[]plugin.ConsolePluginTrimmed, error,
) {
	listURL := strings.TrimSuffix(preparePluginRequestURLByCluster(info, consolePluginUrl, ""), "/")

	listReq, err := http.NewRequest("GET", listURL, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating list request: %v", err)
	}
	addAuthorizationHeader(req, listReq)

	resp, err := requestConsolePlugin(listReq, sp.clientset)
	if err != nil {
		return nil, fmt.Errorf("error listing consoleplugins: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %v", err)
	}

	var list plugin.ConsolePluginList
	if err := json.Unmarshal(body, &list); err != nil {
		return nil, fmt.Errorf("error unmarshaling list: %v", err)
	}

	result := make([]plugin.ConsolePluginTrimmed, 0, len(list.Items))
	for i := range list.Items {
		cp := &list.Items[i]
		if !cp.Spec.Enabled {
			continue
		}
		trimmed := trimConsolePlugin(cp)
		result = append(result, trimmed)
	}
	return result, nil
}

func (sp *componentProxy) getHostPlugin(req *http.Request, info *request.RequestInfo,
	pluginName string,
) (*plugin.ConsolePluginTrimmed, error) {
	requestUrl := preparePluginRequestURLByCluster(info, consolePluginUrl, pluginName)
	cp, err := getConsolePlugin(req, requestUrl, sp.clientset)
	if err != nil {
		zlog.Info("the host cluster does not install multicluster plugin")
		return nil, nil
	}
	consolePluginTrimmed := trimConsolePlugin(cp)
	return &consolePluginTrimmed, nil
}

func trimConsolePlugin(cp *plugin.ConsolePlugin) plugin.ConsolePluginTrimmed {
	consolePluginTrimmed := plugin.ConsolePluginTrimmed{
		DisplayName:     cp.Spec.DisplayName,
		DisplayNameI18n: cp.Spec.DisplayNameI18n,
		PluginName:      cp.Spec.PluginName,
		Order:           formatOrder(cp.Spec.Order),
		SubPages:        cp.Spec.SubPages,
		Entrypoint:      string(cp.Spec.Entrypoint),
		URL:             cp.Status.Link,
		Enabled:         cp.Spec.Enabled,
	}
	if releaseName, ok := cp.ObjectMeta.Annotations["meta.helm.sh/release-name"]; ok {
		consolePluginTrimmed.Release = releaseName
	}
	return consolePluginTrimmed
}

func isAlertRequest(pathname string) bool {
	return strings.HasPrefix(pathname, alertApiPrefix+"/")
}

func isMonitoringRequest(pathname string) bool {
	return strings.HasPrefix(pathname, monitoringApiPrefix)
}

func isWebTerminalRequest(pathname string) bool {
	return strings.HasPrefix(pathname, webTerminalApiPrefix)
}

func isOAuthRequest(pathname string) bool {
	return strings.HasPrefix(pathname, webPasswordPrefix)
}

func isMarketPlaceRequest(pathname string) bool {
	return strings.HasPrefix(pathname, marketApiPrefix)
}

func isApplicationRequest(pathname string) bool {
	return strings.HasPrefix(pathname, applicationApiPrefix)
}

func isUserManagementRequest(pathname string) bool {
	return strings.HasPrefix(pathname, userManagementApiPrefix)
}

func isPluginRequest(pathname string) bool {
	return strings.HasPrefix(pathname, pluginApiPrefix)
}

func (sp *componentProxy) checkCreatedBySessionUser(req *http.Request) bool {
	if req.Method != "POST" {
		return true
	}
	if req.URL.Path != "/api/v2/silences" {
		return true
	}

	// get user info from silence request
	var alert AlertSilenceRequest
	bodyBytes, err := io.ReadAll(req.Body)
	if err != nil {
		return false
	}
	req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	err = json.NewDecoder(bytes.NewReader(bodyBytes)).Decode(&alert)
	if err != nil {
		zlog.Error("error to read body bytes from Alert Request: %v", err)
		return false
	}
	createdBy := alert.CreatedBy

	// get user info from session
	sessionID, err := req.Cookie(cookieNameSessionID)
	if err != nil {
		zlog.Error("Fail to get cookieNameSessionID from cookie: %v", err)
		return false
	}
	token, _, err := auth.GetTokenFromSessionID(sp.clientset, sessionID.Value)
	if err != nil && err.Error() != constant.OnlyAccessTokenExpiredErrorStr {
		zlog.Errorf("Fail to get token from cookie: %v", err)
		return false
	}

	// parse JWT
	claims := JWTAccessClaims{
		StandardClaims: jwt.StandardClaims{},
	}
	_, _, err = jwt.NewParser().ParseUnverified(token, &claims)
	if err != nil {
		zlog.Errorf("Fail to parse tokenJWT: %v", err)
		return false
	}

	// check user info from token
	if claims.Subject != createdBy {
		zlog.Error("Unauthorized user changes.")
		return false
	}

	return true
}

const (
	protocolIndex  = 1
	serviceIndex   = 2
	namespaceIndex = 3
	portIndex      = 4
)

func (sp *componentProxy) convertToServiceProxyPath(host string) string {
	// Prefer cluster DNS FQDNs (AAAA-capable on IPv6 planes), e.g.:
	// host: http(s)://my-service.my-namespace.svc.cluster.local:80
	// Literal IPs are not converted to ServiceProxy paths; configure DNS hosts instead.
	re := regexp.MustCompile(`^(https?):\/\/([a-zA-Z0-9-]+)\.([a-zA-Z0-9-]+)(?:\.svc\.cluster\.local)?(?::(\d+))?$`)
	matches := re.FindStringSubmatch(host)
	if len(matches) <= 0 {
		zlog.Error("Invalid k8s service format.")
		return ""
	}

	// fetch each component
	protocol := matches[protocolIndex]
	svc := matches[serviceIndex]
	ns := matches[namespaceIndex]
	port := matches[portIndex]

	proxyPath := constant.ServiceProxyURL
	proxyPath = strings.Replace(proxyPath, "{namespace}", ns, 1)
	if protocol == "https" {
		proxyPath = strings.Replace(proxyPath, "{service}", protocol+":"+svc, 1)
		if port == "" {
			port = "443"
		}
	} else {
		proxyPath = strings.Replace(proxyPath, "{service}", svc, 1)
		if port == "" {
			port = "80"
		}
	}
	proxyPath = strings.Replace(proxyPath, "{port}", port, 1)

	return proxyPath
}

func formatOrder(order *int64) *string {
	if order != nil {
		formattedOrder := strconv.FormatInt(*order, constant.BaseTen)
		return &formattedOrder
	}
	return nil
}
