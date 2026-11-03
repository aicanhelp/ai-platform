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

package auth

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"golang.org/x/oauth2"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"console-service/pkg/constant"
	"console-service/pkg/i18n"
	"console-service/pkg/server/runtime"
	"console-service/pkg/utils/authutil"
	"console-service/pkg/utils/httputil"
	"console-service/pkg/utils/util"
	"console-service/pkg/zlog"
)

const (
	// cookie字段名称 & default value
	cookieNameLoginState = "state"
	cookieNameSessionID  = "sessionID"
	cookieNameWsID       = "wsID"
	cookieNameIdpLogin   = "idpLogin"
	defaultExpireTime    = 7200
	// cookie字段长度
	loginStateByteLength = 4
	// 请求参数名称
	queryAuthCode        = "code"
	queryLoginState      = "state"
	queryErrorCode       = "error"
	queryErrorDesc       = "error_description"
	pathIdentityProvider = "identity_provider"
	refreshTokenExpiry   = "refresh_token_expires_in"
	// Host&Path
	oAuthAuthorizeEndpoint  = "/oauth/authorize"
	oAuthTokenEndpoint      = "/oauth/token"
	oAuthLogoutEndpoint     = "/auth/logout"
	consoleLoginEndpoint    = "/auth/login"
	consoleCallbackEndpoint = "/auth/callback"
	consoleRootPage         = "/"
	// 请求参数的值
	clientIDEnv              = "OAUTH_CLIENT_ID"
	clientSecretEnv          = "OAUTH_CLIENT_SECRET"
	identityProviderPassword = "fuyaoPasswordProvider"
	// 心跳间隔时间
	heartbeatInterval = 1 * time.Minute
)

// OAuth2Config defines oauth2 config
type OAuth2Config interface {
	AuthCodeURL(state string, opts ...oauth2.AuthCodeOption) string
	Exchange(ctx context.Context, code string, opts ...oauth2.AuthCodeOption) (*oauth2.Token, error)
	TokenSource(ctx context.Context, t *oauth2.Token) oauth2.TokenSource
}

// Handler is an http request handler struct
type Handler struct {
	kubeConfig  *rest.Config
	clientset   kubernetes.Interface
	oauthConfig OAuth2Config
	tokenCache  map[string]*AccessRefreshToken
	mu          sync.Mutex
}

var (
	handlerInstance *Handler
	handlerOnce     sync.Once
)

// NewHandler inits the oauth handler, singleton pattern
func NewHandler(kubeConfig *rest.Config) (*Handler, error) {
	var initErr error
	handlerOnce.Do(func() {
		clientset, err := kubernetes.NewForConfig(kubeConfig)
		if err != nil {
			zlog.Errorf("error creating client set, err: %v", err)
			initErr = err
			return
		}

		csConfig, err := util.GetConsoleServiceConfig(clientset)
		var oauthServerHost string
		if err != nil {
			zlog.Warnf("read console-service-config config map failed, reading default set")
		} else {
			oauthServerHost = csConfig.OAuthServerHost
		}

		// load client-id and client-secret
		clientID := os.Getenv(clientIDEnv)
		clientSecret := os.Getenv(clientSecretEnv)
		if clientID == "" || clientSecret == "" {
			zlog.Errorf("cannot load correct client-id or client-secret")
			initErr = errors.New("cannot load correct client-id or client-secret")
			return
		}

		oauthConfig := &oauth2.Config{
			ClientID:     clientID,
			ClientSecret: clientSecret,
			Endpoint: oauth2.Endpoint{
				AuthURL:  "/oauth2" + oAuthAuthorizeEndpoint,
				TokenURL: oauthServerHost + "/oauth2" + oAuthTokenEndpoint,
			},
			RedirectURL: runtime.RestRootPath + consoleCallbackEndpoint,
		}

		handlerInstance = &Handler{
			kubeConfig:  kubeConfig,
			clientset:   clientset,
			oauthConfig: oauthConfig,
			tokenCache:  make(map[string]*AccessRefreshToken),
		}
	})

	if handlerInstance == nil {
		if initErr != nil {
			return nil, initErr
		}
		return nil, errors.New("auth handler initialization failed")
	}

	return handlerInstance, nil
}

// loginHandler handles login request:
// 1. generate random login-state
// 2. set login-state in cookie
// 3. redirect to oauth-server
func (h *Handler) loginHandler(req *restful.Request, resp *restful.Response) {
	zlog.Infof("Receive login request")
	sessionIDCookie, err := req.Request.Cookie(cookieNameSessionID)
	if err == nil {
		if h.checkSessionCookie(sessionIDCookie) {
			zlog.Info("Valid sessionID found in cookie")
			http.Redirect(resp.ResponseWriter, req.Request, consoleRootPage, http.StatusFound)
			return
		}
		zlog.Error("Invalid sessionID found in cookie")
		clearCookie(cookieNameSessionID, resp)
	}

	loginStateStr, err := createLoginState(loginStateByteLength)
	if err != nil {
		zlog.Error("Login state generation failed")
		_ = resp.WriteError(http.StatusInternalServerError, err)
		return
	}

	expiry := time.Now().Add(defaultExpireTime * time.Second)
	setCookie(cookieNameLoginState, loginStateStr, expiry, resp)

	redirectURI := h.generateRedirectURI(loginStateStr)
	http.Redirect(resp.ResponseWriter, req.Request, redirectURI, http.StatusFound)
}

func (h *Handler) generateRedirectURI(loginStateStr string) string {
	passwordProvider := oauth2.SetAuthURLParam(pathIdentityProvider, identityProviderPassword)
	redirectURI := h.oauthConfig.AuthCodeURL(loginStateStr, passwordProvider)
	return redirectURI
}

func createLoginState(length int) (string, error) {
	loginStateBytes := make([]byte, length)
	_, err := io.ReadFull(rand.Reader, loginStateBytes[:])
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(loginStateBytes[:]), nil
}

// checkSessionCookie returns true if sessionID in the cookie is valid
func (h *Handler) checkSessionCookie(cookie *http.Cookie) bool {
	_, _, err := GetTokenFromSessionID(h.clientset, cookie.Value)
	return err == nil || err.Error() == constant.OnlyAccessTokenExpiredErrorStr
}

// callbackHandler handles oauth callback request, to which the FE is redirected by oauth server
// after the authorization request:
// 1. parse query param and check for errors
// 2. exchange auth code for token
// 3. create session, store in persistence and set in cookie
// 4. redirect to console root page
func (h *Handler) callbackHandler(req *restful.Request, resp *restful.Response) {
	zlog.Info("Receive OAuth callback request")

	locale := i18n.GetLocaleFromContext(req.Request.Context())
	responseText, err := generateErrorRespText(req, locale)
	if err != nil {
		resp.Header().Set("Content-Type", "text/html")
		resp.Write([]byte(responseText))
		return
	}

	authCode, loginState, err := parseCallbackQuery(req)
	if err != nil {
		_ = resp.WriteError(http.StatusUnauthorized, err)
		return
	}
	if err := checkLoginState(req, loginState); err != nil {
		http.Redirect(resp.ResponseWriter, req.Request, consoleRootPage, http.StatusSeeOther)
		return
	}
	token, err := h.exchangeCodeForToken(authCode)
	if err != nil {
		_ = resp.WriteError(http.StatusInternalServerError, err)
		return
	}
	// create SessionStore for the token with a random sessionID
	session, err := NewStoreSession(token)
	if err != nil {
		zlog.Error("Fail to generate session")
		_ = resp.WriteError(http.StatusInternalServerError, err)
		return
	}

	err = StoreSession(h.clientset, session, false)
	if err != nil {
		zlog.Error("Fail to store session to Persistence", resp)
		_ = resp.WriteError(http.StatusInternalServerError, err)
		return
	}

	clearCookie(cookieNameLoginState, resp)
	setCookie(cookieNameSessionID, session.GetSessionID(), token.RefreshTokenExpiry, resp)
	wsId := uuid.New().String()
	setCookie(cookieNameWsID, wsId, time.Time{}, resp)

	// read referrer URL from cookie and redirect to original page
	redirectURL := ResolvePostLoginRedirectURL(req.Request, resp, consoleRootPage)
	http.Redirect(resp.ResponseWriter, req.Request, redirectURL, http.StatusFound)
}

func generateErrorRespText(req *restful.Request, locale string) (string, error) {
	// 错误代码或错误描述存在，则报错
	errorCode := req.QueryParameter(queryErrorCode)
	errorDesc := req.QueryParameter(queryErrorDesc)
	if errorCode != "" || errorDesc != "" {
		zlog.Errorf("Error %s: %s", errorCode, errorDesc)
		responseText := renderHTML(locale, html.EscapeString(errorCode), html.EscapeString(errorDesc))
		return responseText, errors.New(errorCode)
	}
	return "", nil
}

const responseTemplate = `
	<!DOCTYPE html>
	<html lang="%s">
	<head>
		<meta charset="UTF-8">
		<title>openFuyao</title>
	</head>
	<body>
		<h1>%s</h1>
		<p>%s</p>
	</body>
	</html>
`

func renderHTML(locale, header, text string) string {
	responseText := fmt.Sprintf(responseTemplate, locale, header, text)
	return responseText
}

func parseCallbackQuery(req *restful.Request) (string, string, error) {
	// 检查code参数
	authCode := req.QueryParameter(queryAuthCode)
	if authCode == "" {
		zlog.Errorf("Missing required query parameter %s", queryAuthCode)
		return "", "", errors.New("missing query param: " + queryAuthCode)
	}
	// 检查state参数
	loginState := req.QueryParameter(queryLoginState)
	if loginState == "" {
		zlog.Errorf("Missing required query parameter %s", queryLoginState)
		return "", "", errors.New("missing query param: " + queryLoginState)
	}
	return authCode, loginState, nil
}

// checkLoginState checks whether login state in cookie and query param match with each other
func checkLoginState(req *restful.Request, loginState string) error {
	cookieState, err := req.Request.Cookie(cookieNameLoginState)
	// 未能获取cookie中的login-state
	if err != nil {
		zlog.Errorf("Fail to get cookieNameLoginState from cookie")
		return errors.New("no login state in cookie")
	}
	// 路径参数state与cookie.login-state中的不符
	if cookieState.Value != loginState {
		zlog.Errorf("Query parameter queryLoginState does not matching cookie cookieNameLoginState")
		return errors.New("login state in cookie and query not matching")
	}
	return nil
}

func prepareOAuth2Context() (context.Context, error) {
	enableTLS, err := httputil.IsHttpsEnabled()
	if err != nil {
		return nil, err
	}
	config, err := httputil.GetHttpConfig(enableTLS)
	if err != nil {
		zlog.Error("Fail to get https config")
		return nil, err
	}
	oauthCtx := context.WithValue(context.Background(), oauth2.HTTPClient, &http.Client{Transport: &http.Transport{
		TLSClientConfig: config,
		Proxy:           http.ProxyFromEnvironment,
	}})
	return oauthCtx, nil
}

// exchangeCodeForToken communicate with oauth-server, exchanging code for token
func (h *Handler) exchangeCodeForToken(authCode string) (*AccessRefreshToken, error) {
	oauthCtx, err := prepareOAuth2Context()
	if err != nil {
		return nil, err
	}

	token, err := h.oauthConfig.Exchange(oauthCtx, authCode)
	if err != nil {
		zlog.Error("Fail to exchange authCode for token")
		return nil, err
	}

	refreshActualExpiry, err := retrieveRefreshTokenExpiry(token)
	if err != nil {
		return nil, err
	}

	return &AccessRefreshToken{
		AccessToken:        token.AccessToken,
		AccessTokenExpiry:  token.Expiry,
		RefreshToken:       token.RefreshToken,
		RefreshTokenExpiry: refreshActualExpiry,
	}, nil
}

func retrieveRefreshTokenExpiry(token *oauth2.Token) (time.Time, error) {
	refreshExpiry, ok := token.Extra(refreshTokenExpiry).(float64)
	if !ok {
		zlog.Error("Fail to retrieve refresh token")
		return time.Time{}, errors.New("fail to retrieve refresh token")
	}

	return time.Now().Add(time.Duration(refreshExpiry) * time.Second), nil
}

// RotateTokenHandler exchanges refresh-token for new-token and extends the original session expiry
func (h *Handler) RotateTokenHandler(w http.ResponseWriter, sessionID string, token *oauth2.Token) (
	*AccessRefreshToken, error,
) {
	newToken, cached, err := h.exchangeRefreshTokenForNewToken(token)
	if err != nil {
		return nil, err
	}

	// return immediately if using cached token, since this token has been rotated
	if cached {
		return newToken, nil
	}

	// create SessionStore for the token with a random sessionID
	ss, err := UpdateSession(h.clientset, sessionID, newToken)
	if err != nil {
		zlog.Errorf("Fail to update session to persistence, %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		_, err = w.Write([]byte(err.Error()))
		if err != nil {
			zlog.Errorf("http write failed: %v", err)
		}
		return nil, err
	}
	if err = StoreSession(h.clientset, ss, true); err != nil {
		zlog.Errorf("Fail to restore updated session to persistence, %v", err)
		return nil, err
	}

	// update sessionID cookie
	setCookie(cookieNameSessionID, sessionID, newToken.RefreshTokenExpiry, w)

	return newToken, nil
}

// exchangeRefreshTokenForNewToken communicate with oauth-server, use refresh-token to update tokens
func (h *Handler) exchangeRefreshTokenForNewToken(token *oauth2.Token) (*AccessRefreshToken, bool, error) {
	oauthCtx, err := prepareOAuth2Context()
	if err != nil {
		return nil, false, err
	}

	// add lock to avoid exchanging multiple times
	h.mu.Lock()
	defer h.mu.Unlock()

	if cache, exists := h.tokenCache[token.RefreshToken]; exists {
		zlog.Infof("tokenCache hits ------ refreshToken: %s", token.RefreshToken)
		return cache, true, nil
	}

	newToken, err := h.oauthConfig.TokenSource(oauthCtx, token).Token()
	if err != nil {
		zlog.Errorf("Fail to exchange refresh-token for access-token, err: %v", err)
		return nil, false, err
	}

	refreshActualExpiry, err := retrieveRefreshTokenExpiry(newToken)
	if err != nil {
		return nil, false, err
	}

	// update cache
	accessRefreshToken := &AccessRefreshToken{
		AccessToken:        newToken.AccessToken,
		AccessTokenExpiry:  newToken.Expiry,
		RefreshToken:       newToken.RefreshToken,
		RefreshTokenExpiry: refreshActualExpiry,
	}
	h.tokenCache[token.RefreshToken] = accessRefreshToken

	return accessRefreshToken, false, nil
}

// logoutHandler handles logout request:
// 1. get sessionID from cookie
// 2. delete session from persistent storage
// 3. redirect to login page
func (h *Handler) logoutHandler(req *restful.Request, resp *restful.Response) {
	h.LogoutCore(req.Request, resp.ResponseWriter)
	resp.WriteHeader(http.StatusNoContent)
}

// LogoutCore deletes session when:
// 1. logout requests
// 2. password modification (if needed)
func (h *Handler) LogoutCore(req *http.Request, w http.ResponseWriter) {
	zlog.Info("Receive logout request")
	ssID, err := req.Cookie(cookieNameSessionID)
	clearCookie(cookieNameIdpLogin, w)
	if err != nil {
		zlog.Errorf("Fail to get cookieNameSessionID from cookie")
		return
	}
	clearCookie(cookieNameSessionID, w)
	err = DeleteSession(h.clientset, ssID.Value)
	if err != nil {
		zlog.Error("Fail to delete session")
	} else {
		zlog.Info("Logout succeed")
	}
}

func setCookie(name, value string, expires time.Time, resp http.ResponseWriter) {
	cookie := http.Cookie{
		Name:     name,
		Value:    value,
		Path:     "/",
		HttpOnly: true,
		Secure:   true,
		SameSite: http.SameSiteLaxMode,
	}
	if !expires.IsZero() {
		cookie.Expires = expires
	}
	http.SetCookie(resp, &cookie)
}

func clearCookie(name string, resp http.ResponseWriter) {
	cookie := http.Cookie{
		Name:     name,
		Value:    "",
		Expires:  time.Unix(1, 0),
		Path:     "/",
		HttpOnly: true,
		Secure:   true,
		SameSite: http.SameSiteLaxMode,
	}
	http.SetCookie(resp, &cookie)
}

// getCurrentUserHandler retrieves the username from token and returns it to conosle-website
func (h *Handler) getCurrentUserHandler(req *restful.Request, resp *restful.Response) {
	// get token from openfuyaoauthheader
	accessToken, err := GetTokenFromOpenFuyaoAuthHeader(req.Request)
	if err != nil {
		// get token from sessionid
		sessionID, err := req.Request.Cookie(cookieNameSessionID)
		if err != nil {
			errResponse := httputil.GetResponseJson(http.StatusUnauthorized, "cookie not found", nil)
			// no sessionid-cookie found, return 401 directly
			_ = resp.WriteHeaderAndEntity(http.StatusUnauthorized, errResponse)
			return
		}

		accessToken, _, err = GetTokenFromSessionID(h.clientset, sessionID.Value)
		if err != nil && err.Error() != constant.OnlyAccessTokenExpiredErrorStr {
			// refresh-token expires, return 401 directly
			zlog.Errorf("Cannot get token from sessionID")
			errResponse := httputil.GetResponseJson(http.StatusUnauthorized, "cannot get token from sessionID", nil)
			_ = resp.WriteHeaderAndEntity(http.StatusUnauthorized, errResponse)
			return
		}
	}
	// extract username
	userinfo, err := authutil.ExtractUserFromJWT(accessToken)
	if err != nil {
		zlog.Errorf("Cannot extract user from token")
		errResponse := httputil.GetResponseJson(http.StatusInternalServerError, "cannot extract user from token", nil)
		_ = resp.WriteHeaderAndEntity(http.StatusInternalServerError, errResponse)
		return
	}

	response := httputil.GetResponseJson(http.StatusOK, "get current user succeed", userinfo.GetName())
	_ = resp.WriteHeaderAndEntity(http.StatusOK, response)
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
	Error: func(w http.ResponseWriter, r *http.Request, status int, reason error) {
		zlog.Errorf("Upgrade failed: %v", reason)
		response := httputil.GetResponseJson(int32(status), reason.Error(), nil)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		err := json.NewEncoder(w).Encode(response)
		if err != nil {
			zlog.Errorf("Failed to encode response: %v", err)
		}
	},
}

// loginStatusHandler starts the websocket connection to track login status
func (h *Handler) loginStatusHandler(req *restful.Request, resp *restful.Response) {
	// fetch wsID cookie
	wsUuid, err := req.Request.Cookie(cookieNameWsID)
	if err != nil {
		zlog.Errorf("Fail to get cookieNameWsID from cookie")
		response := httputil.GetResponseJson(http.StatusBadRequest,
			"websocket connection request failed, fail to get cookieNameWsID from cookie", nil)
		_ = resp.WriteHeaderAndEntity(http.StatusBadRequest, response)
		return
	}

	// upgrade request
	conn, err := upgrader.Upgrade(resp.ResponseWriter, req.Request, nil)
	if err != nil {
		return
	}

	// store new websocket conn
	RemoveWebSocketConnection(wsUuid.Value)
	AddWebSocketConnection(wsUuid.Value, conn)

	// send point message to keep this websocket conn alive
	go h.sendHeartbeat(wsUuid.Value, conn)
}

// TriggerWSLogout fetches the websocket conn from wsUuid and sends termination signal
func (h *Handler) TriggerWSLogout(req *http.Request, w http.ResponseWriter) {
	wsUuid, err := req.Cookie(cookieNameWsID)
	if err != nil {
		zlog.Errorf("Fail to get cookieNameWsID from cookie")
		http.Error(w, "Failed to get wsUuid from cookie", http.StatusUnauthorized)
		return
	}

	conn, ok := GetWebSocketConnection(wsUuid.Value)
	if !ok {
		zlog.Errorf("No WebSocket connection found for wsUuid: %s", wsUuid.Value)
		http.Error(w, "WebSocket connection not found", http.StatusUnauthorized)
		return
	}

	if err = conn.Conn.WriteMessage(websocket.TextMessage, []byte(`{"loginStatus": "false"}`)); err != nil {
		zlog.Errorf("Failed to send logout message: %v", err)
		http.Error(w, "Failed to close WebSocket connection", http.StatusUnauthorized)
		return
	}

	if err = conn.Conn.Close(); err != nil {
		zlog.Errorf("Failed to close WebSocket connection: %v", err)
		return
	}

	w.WriteHeader(http.StatusUnauthorized)
	_, err = w.Write([]byte(`{"status": "successfully trigger logout"}`))
	if err != nil {
		zlog.Errorf("Failed to write response: %v", err)
	}
}

// sendHeartbeat inits a timer and send heartbeat each minute
func (h *Handler) sendHeartbeat(wsUuid string, conn *websocket.Conn) {
	// send heartbeat immediately
	if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"loginStatus": "true"}`)); err != nil {
		zlog.Errorf("Failed to send heartbeat to %s: %v", wsUuid, err)
		_ = conn.Close()
		RemoveWebSocketConnection(wsUuid)
		return
	}
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()

	// loop heartbeat
	for range ticker.C {
		if err := conn.WriteMessage(websocket.TextMessage,
			[]byte(`{"loginStatus": "true"}`)); err != nil {
			zlog.Errorf("Failed to send heartbeat to %s: %v", wsUuid, err)
			_ = conn.Close()
			RemoveWebSocketConnection(wsUuid)
			return
		}
	}
}
