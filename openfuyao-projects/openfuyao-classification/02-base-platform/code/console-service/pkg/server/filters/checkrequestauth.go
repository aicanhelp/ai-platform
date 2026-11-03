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
	"path"
	"strings"

	"golang.org/x/oauth2"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"console-service/pkg/auth"
	"console-service/pkg/constant"
	"console-service/pkg/server/runtime"
	"console-service/pkg/utils/authutil"
	"console-service/pkg/utils/util"
	"console-service/pkg/zlog"
)

const (
	webAuthPrefix       = "/auth"
	cookieNameSessionID = "sessionID"
	kubeApiPrefix       = "/api/kubernetes"
	clusterPrefix       = "/clusters"
	userPrefix          = "/user"
	multiClusterPrefix  = "/multicluster"
)

type requestAuthChecker struct {
	authHttpHandler    http.Handler
	nonAuthHttpHandler http.Handler
	clientSet          *kubernetes.Clientset
	authHandler        *auth.Handler
	serverName         string
}

// CheckRequestAuth direct auth requests & non-authorized requests to server handler;
// otherwise continue the chain
func CheckRequestAuth(authHttpHandler, nonAuthHttpHandler http.Handler, config *rest.Config) http.Handler {
	authHandler, err1 := auth.NewHandler(config)
	k8sClient, err2 := kubernetes.NewForConfig(config)
	if err1 != nil || err2 != nil {
		zlog.Error("Fail to initialize client set")
		return &requestAuthChecker{
			authHttpHandler:    nonAuthHttpHandler,
			nonAuthHttpHandler: nonAuthHttpHandler,
			clientSet:          k8sClient,
			authHandler:        authHandler,
			serverName:         "",
		}
	}

	var serverName string
	csConfig, err := util.GetConsoleServiceConfig(k8sClient)
	if err == nil {
		serverName = csConfig.ServerName
	}

	return &requestAuthChecker{
		authHttpHandler:    authHttpHandler,
		nonAuthHttpHandler: nonAuthHttpHandler,
		clientSet:          k8sClient,
		authHandler:        authHandler,
		serverName:         serverName,
	}
}

func (rc *requestAuthChecker) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// direct auth requests to server handler
	if isRESTAuthRequest(req.URL.Path) {
		zlog.Infof("Auth request: %s", req.URL.Path)
		rc.authHttpHandler.ServeHTTP(w, req)
		return
	}

	if strings.HasPrefix(req.URL.Path, "/favicon") {
		rc.nonAuthHttpHandler.ServeHTTP(w, req)
		return
	}

	token, err := rc.getTokenFromOpenFuyaoAuthHeader(req)
	var refreshToken string
	if err != nil {
		token, refreshToken, err = rc.getTokenIfSessionValid(req)
	}
	// access-token expired but refresh-token valid
	if err != nil && err.Error() == constant.OnlyAccessTokenExpiredErrorStr {
		zlog.Info("rotating access-token")
		token, err = rc.rotateAccessToken(w, req, refreshToken)
		if err != nil {
			zlog.Errorf("rotate access-token failed, err: %v", err)
			return
		}
	}
	if err != nil {
		// Save original page URL before login redirect or websocket logout.
		auth.SaveReferrerCookieIfNeeded(w, req)
		if isFuyaoApi(req) {
			rc.authHandler.TriggerWSLogout(req, w)
		} else {
			http.Redirect(w, req, path.Join(runtime.RestRootPath, "/auth/login"), http.StatusSeeOther)
		}
		return
	}

	// set Authorize header for authorized requests, then proceed
	zlog.Info("Set authorization token in request header, proceeding")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set(constant.OpenFuyaoAuthHeader, "Bearer "+token)

	if isWSAuthRequest(req.URL.Path) {
		zlog.Infof("Auth websocket request: %s", req.URL.Path)
		rc.authHttpHandler.ServeHTTP(w, req)
		return
	}
	rc.nonAuthHttpHandler.ServeHTTP(w, req)
}

func (rc *requestAuthChecker) rotateAccessToken(w http.ResponseWriter, req *http.Request, refreshToken string) (
	string, error) {
	// 1. exchange refresh-token for new access & refresh-token
	// 2. update session secret (expiry, access/refresh-token)
	sessionID, err := req.Cookie(cookieNameSessionID)
	if err != nil {
		zlog.Errorf("Fail to get cookieNameSessionID from cookie")
		return "", err
	}

	token := &oauth2.Token{RefreshToken: refreshToken}
	newToken, err := rc.authHandler.RotateTokenHandler(w, sessionID.Value, token)
	if err != nil {
		return "", err
	}

	return newToken.AccessToken, nil
}

func (rc *requestAuthChecker) getTokenIfSessionValid(req *http.Request) (string, string, error) {
	sessionID, err := req.Cookie(cookieNameSessionID)
	if err != nil {
		return "", "", err
	}

	return auth.GetTokenFromSessionID(rc.clientSet, sessionID.Value)
}

func (rc *requestAuthChecker) getTokenFromOpenFuyaoAuthHeader(req *http.Request) (string, error) {
	token := req.Header.Get(constant.OpenFuyaoAuthHeader)
	if !strings.HasPrefix(token, "Bearer ") {
		return "", fmt.Errorf("cannot fetch token from openfuyao authentication header")
	}
	token = strings.TrimPrefix(token, "Bearer ")

	_, err := authutil.ExtractUserFromJWT(token)
	if err != nil {
		zlog.Errorf("openfuyao authentication token is invalid, cannot extract user")
		return "", err
	}

	return token, nil
}

func isFuyaoApi(req *http.Request) bool {
	consoleOrMonitor := isConsoleRequest(req.URL.Path) || isMonitoringRequest(req.URL.Path)
	oauthAlertKubeapi := isOAuthRequest(req.URL.Path) || isAlertRequest(req.URL.Path) || isKubeApiRequest(req.URL.Path)
	userOrMultiCluster := isUserRequest(req.URL.Path) || isMultiClusterRequest(req.URL.Path)
	clusterApi := isMultiClusterProxyRequest(req.URL.Path)
	return consoleOrMonitor || oauthAlertKubeapi || clusterApi || userOrMultiCluster
}

func isRESTAuthRequest(pathname string) bool {
	return strings.HasPrefix(pathname, path.Join(runtime.RestRootPath, webAuthPrefix))
}

func isWSAuthRequest(pathname string) bool {
	return strings.HasPrefix(pathname, path.Join(runtime.WsRootPath, webAuthPrefix))
}

func isKubeApiRequest(path string) bool {
	return strings.HasPrefix(path, kubeApiPrefix)
}

func isMultiClusterProxyRequest(path string) bool {
	return strings.HasPrefix(path, clusterPrefix)
}

func isUserRequest(pathname string) bool {
	return strings.HasPrefix(pathname, path.Join(runtime.RestRootPath, userPrefix))
}

func isMultiClusterRequest(pathname string) bool {
	return strings.HasPrefix(pathname, path.Join(runtime.RestRootPath, multiClusterPrefix))
}
