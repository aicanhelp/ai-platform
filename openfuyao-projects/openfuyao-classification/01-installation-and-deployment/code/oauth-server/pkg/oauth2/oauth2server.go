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

// Package oauth2
package oauth2

import (
	"encoding/json"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v4"
	"gopkg.in/oauth2.v3"
	"gopkg.in/oauth2.v3/errors"
	"gopkg.in/oauth2.v3/generates"
	"gopkg.in/oauth2.v3/manage"
	"gopkg.in/oauth2.v3/models"
	"gopkg.in/oauth2.v3/server"
	"gopkg.in/oauth2.v3/store"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/fuyaostore"
	"openfuyao/oauth-server/pkg/generators"
	"openfuyao/oauth-server/pkg/httpserver"
	"openfuyao/oauth-server/pkg/sessions"
	"openfuyao/oauth-server/pkg/utils"
	"openfuyao/oauth-server/pkg/zlog"
)

// FuyaoAuthorizeRequest extends the original AuthorizeRequest in go-oauth2
type FuyaoAuthorizeRequest struct {
	server.AuthorizeRequest
	identityProvider string
}

// FuyaoAuthorizeServer extends the original authorize server in go-oauth2
type FuyaoAuthorizeServer struct {
	// inherits and overrides some interfaces
	*server.Server
	// session-store implementation, which stores whether the user has logged in via account password
	idpLoginStore *sessions.CookieStore
	// tokenStore stores the code/access-token in the k8s secret
	tokenStore oauth2.TokenStore
	// csrfCookieName
	csrfCookieName string
}

// NewFuyaoAuthorizeServer inits a FuyaoAuthorizeServer
func NewFuyaoAuthorizeServer(
	cfg *server.Config,
	manager oauth2.Manager,
	idpLoginStore *sessions.CookieStore,
	tokenStore oauth2.TokenStore,
) *FuyaoAuthorizeServer {
	return &FuyaoAuthorizeServer{
		Server:        server.NewServer(cfg, manager),
		idpLoginStore: idpLoginStore,
		tokenStore:    tokenStore,
	}
}

// NewOAuthServer inits the go-oauth2 oauth server
func NewOAuthServer(
	idpLoginStore *sessions.CookieStore,
	tokenStore *fuyaostore.K8sSecretStore, cfg *config.OAuthServerConfig, csrfCookieName string,
) *FuyaoAuthorizeServer {
	// init inner-oauth2-server, manager and configs
	innerOAuthServerConfig := server.NewConfig()
	innerOAuthServerConfig.AllowedResponseTypes = []oauth2.ResponseType{oauth2.Code}
	innerOAuthServerConfig.AllowedGrantTypes = []oauth2.GrantType{oauth2.AuthorizationCode, oauth2.Refreshing}

	manager := manage.NewDefaultManager()
	manager.SetAuthorizeCodeTokenCfg(
		&manage.Config{AccessTokenExp: cfg.AccessTokenExp, RefreshTokenExp: cfg.RefreshTokenExp,
			IsGenerateRefresh: cfg.IsGenerateRefresh})
	manager.SetRefreshTokenCfg(
		&manage.RefreshingConfig{AccessTokenExp: cfg.AccessTokenExp, RefreshTokenExp: cfg.RefreshTokenExp,
			IsGenerateRefresh: cfg.IsGenerateRefresh, IsResetRefreshTime: true, IsRemoveAccess: false,
			IsRemoveRefreshing: true,
		})
	manager.SetAuthorizeCodeExp(cfg.AuthCodeExp)

	// auth code and jwt access token generator
	manager.MapAuthorizeGenerate(generators.NewFuyaoAuthorizeGenerate())
	manager.MapAccessGenerate(
		generates.NewJWTAccessGenerate(cfg.JWTPrivateKey, jwt.SigningMethodHS512))

	// storage
	clientStore := store.NewClientStore()
	for client, secret := range cfg.ClientMapper {
		clientStore.Set(client, &models.Client{
			ID:     client,
			Secret: secret,
		})
	}
	manager.MapClientStorage(clientStore)
	manager.MapTokenStorage(tokenStore)

	srv := NewFuyaoAuthorizeServer(innerOAuthServerConfig, manager, idpLoginStore, tokenStore)
	srv.csrfCookieName = csrfCookieName
	srv.SetInternalErrorHandler(func(err error) (re *errors.Response) {
		log.Println("Internal Error:", err.Error())
		return
	})
	srv.SetResponseErrorHandler(func(re *errors.Response) {
		log.Println("Response Error:", re.Error.Error())
	})

	// token request function configurations
	srv.SetClientInfoHandler(server.ClientFormHandler)
	srv.SetAllowGetAccessRequest(false)

	// token extra returns userid
	srv.SetExtensionFieldsHandler(func(ti oauth2.TokenInfo) map[string]interface{} {
		fieldsValue := make(map[string]interface{})
		fieldsValue[constants.TokenUserID] = ti.GetUserID()
		fieldsValue[constants.RefreshTokenExpiry] = int64(ti.GetRefreshExpiresIn() / time.Second)
		return fieldsValue
	})

	return srv
}

// OAuthAuthorizeHandler http handler for /oauth/authorize
func (s *FuyaoAuthorizeServer) OAuthAuthorizeHandler(w http.ResponseWriter, r *http.Request) {
	// fetch params
	req, err := s.ValidateAuthorizeRequest(r)
	if err != nil {
		if err == fuyaoerrors.ErrRedirectURIIncorrect {
			http.Redirect(w, r, "/", http.StatusFound)
		} else {
			s.redirectAuthorizationCodeError(w, req, err)
		}
		return
	}

	// check whether containing the oauth-session, if so the user has logged in through passwd
	// then check if it is the first time to login
	userResponse, loginStatus, err := s.AuthorizeThroughSession(w, r)
	if err != nil {
		s.redirectAuthorizationCodeError(w, req, err)
		return
	}

	if loginStatus == constants.LoginFailed {
		// redirect to log in (using the selected identityProvider)
		if err = s.RedirectToPage(req, constants.LoginRedirectTemplate, w, r); err != nil {
			s.redirectAuthorizationCodeError(w, req, err)
			return
		}
		return
	}

	if loginStatus == constants.FirstLogin {
		// otherwise redirect to password confirm
		if err = s.RedirectToPage(req, constants.PasswordConfirmRedirectTemplate, w, r); err != nil {
			s.redirectAuthorizationCodeError(w, req, err)
			return
		}
		return
	}

	// generate the oauth code
	req.UserID = userResponse.User.GetName()
	ti, err := s.GetAuthorizeToken(&req.AuthorizeRequest)
	if err != nil {
		s.redirectAuthorizationCodeError(w, req, err)
		return
	}

	// use the default client domain if the redirect URI is empty
	if req.RedirectURI == "" {
		client, err := s.Manager.GetClient(req.ClientID)
		if err != nil {
			s.redirectAuthorizationCodeError(w, req, err)
			return
		}
		req.RedirectURI = client.GetDomain()
	}

	// finally we redirect to the client callback interface
	s.redirectAuthorizationCode(w, req, s.GetAuthorizeData(req.ResponseType, ti))
	return
}

// OAuthTokenHandler http handler for /oauth/token
func (s *FuyaoAuthorizeServer) OAuthTokenHandler(w http.ResponseWriter, r *http.Request) {
	gt, tgr, err := s.ValidationTokenRequest(r)
	if err != nil {
		s.generateTokenError(w, err)
		return
	}

	ti, err := s.GetAccessToken(gt, tgr)
	if err != nil {
		// delete expired authorization code
		if delErr := s.deleteExpiredAuthCode(tgr); delErr != nil {
			zlog.LogErrorf("delete expired auth code goes wrong: err: %v", delErr)
		}
		s.generateTokenError(w, err)
		return
	}

	// 通过ExtensionFieldsHandler来将refresh-token的过期时间传回
	s.returnAccessToken(w, s.GetTokenData(ti), nil)

	return
}

// RedirectToPage redirects to login page when sessionId is missing
func (s *FuyaoAuthorizeServer) RedirectToPage(
	req *FuyaoAuthorizeRequest,
	template string,
	w http.ResponseWriter,
	r *http.Request,
) error {
	if req.identityProvider != constants.FuyaoIdpProvider {
		return fuyaoerrors.ErrIdentityProviderIncorrect
	}
	loginRedirectURL, err := buildRedirectURL(r, req.identityProvider, template)
	if err != nil {
		return err
	}

	http.Redirect(w, r, loginRedirectURL.String(), http.StatusFound)
	return nil
}

func buildRedirectURL(r *http.Request, idp string, tpl string) (*url.URL, error) {
	originalURL := r.URL
	path := strings.Replace(tpl, "%s", idp, 1)

	redirectURL := &url.URL{
		Scheme: originalURL.Scheme,
		Host:   originalURL.Host,
		Path:   path,
	}

	thenParamVal := originalURL.String()
	// 在重定向 URL 的查询参数中添加 then 参数
	query := redirectURL.Query()
	query.Set("then", thenParamVal)
	redirectURL.RawQuery = query.Encode()

	return redirectURL, nil
}

// ValidateAuthorizeRequest makes sure the request params do not miss necessary params
func (s *FuyaoAuthorizeServer) ValidateAuthorizeRequest(r *http.Request) (*FuyaoAuthorizeRequest, error) {
	// original call
	req, err := s.Server.ValidationAuthorizeRequest(r)
	if err != nil {
		return nil, err
	}

	// fetch the path parameter identityProvider
	idp := utils.EscapeSpecialChars(r.FormValue("identity_provider"))
	if idp == "" {
		return nil, fuyaoerrors.ErrIdentityProviderIncorrect
	}

	// fetch the redirect_uri for regex filtering
	if !isValidRedirectURI(r.FormValue("redirect_uri")) {
		return &FuyaoAuthorizeRequest{
			AuthorizeRequest: *req,
			identityProvider: idp,
		}, fuyaoerrors.ErrRedirectURIIncorrect
	}

	return &FuyaoAuthorizeRequest{
		AuthorizeRequest: *req,
		identityProvider: idp,
	}, nil
}

func isValidRedirectURI(s string) bool {
	regexPattern := `^(?:(?:https?://(?:[\w.-]+|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d{1,5})?)|[\w.-]` +
		`+(?:\.\w{2,})?)?(/rest/auth/callback|/[^/]+/oauth/callback)$`
	match, err := regexp.MatchString(regexPattern, s)
	if err != nil {
		zlog.LogErrorf("Error compiling regex:", err)
		return false
	}
	if !match {
		zlog.LogErrorf("redirect_uri parameter does not match")
	}
	return match
}

// AuthorizeThroughSession authorize the user with session stored data
func (s *FuyaoAuthorizeServer) AuthorizeThroughSession(
	w http.ResponseWriter,
	r *http.Request,
) (*authenticator.Response, constants.LoginStatus, error) {
	// fetch the cached user info
	cookieData := s.idpLoginStore.Get(r)

	username, ok1 := cookieData.GetString(constants.UserName)
	uid, ok2 := cookieData.GetString(constants.UserUID)
	groups, ok3 := cookieData.GetArrayString(constants.UserGroups)
	extras, ok4 := cookieData.GetExtras(constants.UserExtra)

	// if it is the first login
	if ok4 && extras[constants.UserFirstLogin][0] == "true" {
		return nil, constants.FirstLogin, nil
	}

	// the session is broken, flush it
	if !ok1 || !ok2 || !ok3 || !ok4 {
		if err := s.idpLoginStore.Put(w, make(sessions.Values)); err != nil {
			zlog.LogErrorf("cannot delete the loginstore used in authorization, err: %v", err)
			return nil, constants.LoginFailed, err
		}
		return nil, constants.LoginFailed, nil
	}

	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name:   username,
			UID:    uid,
			Groups: groups,
			Extra:  extras,
		},
	}, constants.LoggedIn, nil

}

// GetErrorData forms the error return data for oauth2.0
func (s *FuyaoAuthorizeServer) GetErrorData(err error) (map[string]interface{}, int, http.Header) {
	return s.Server.GetErrorData(err)
}

// deleteExpiredAuthCode flush the auth code secret if expiry
func (s *FuyaoAuthorizeServer) deleteExpiredAuthCode(tgr *oauth2.TokenGenerateRequest) error {
	code := tgr.Code
	ti, err := s.tokenStore.GetByCode(code)

	if err != nil {
		zlog.LogErrorf("cannot get auth code, err: %v", err)
		return err
	}
	if ti != nil && ti.GetCodeCreateAt().Add(ti.GetCodeExpiresIn()).Before(time.Now()) {
		// delete the auth code
		if err = s.tokenStore.RemoveByCode(code); err != nil {
			return err
		}
		zlog.LogInfof("successfully delete auth code in secret")
	}

	return nil
}

// SingleLogoutHandler receives requests from console-service logout request flush cookies
func (s *FuyaoAuthorizeServer) SingleLogoutHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpserver.RespondWithStatusMsg(w, http.StatusMethodNotAllowed, 0, fuyaoerrors.ErrStrRequestMethodNotAllowed)
		return
	}
	// fetch the redirect_uri and idpLogin sessionID
	redirectURI := r.URL.Query().Get(constants.LogoutRedirectURI)
	if redirectURI == "" {
		redirectURI = constants.FuyaoLoginEndpoint
	}

	// flush the loginState
	if err := s.idpLoginStore.Put(w, make(sessions.Values)); err != nil {
		zlog.LogErrorf("cannot delete the loginstore used in authorization, err: %v", err)
		httpserver.RespondWithStatusMsg(w, http.StatusInternalServerError, 0, err.Error())
		return
	}
	// flush the csrf cookie
	clearCookie(s.csrfCookieName, w)
	zlog.LogInfof("Logout succeed for user")

	// no content to return
	w.WriteHeader(http.StatusNoContent)
	return
}

// belows are private functions
func (s *FuyaoAuthorizeServer) wrapReturnErrorHandler(w http.ResponseWriter, err error) {
	if err != nil {
		zlog.LogErrorf("fail when writing error back to the http response header, err: %v", err)
		http.Error(w, fuyaoerrors.ErrStrWritingHttpHeader, http.StatusInternalServerError)
		return
	}
	return
}

func (s *FuyaoAuthorizeServer) redirectAuthorizationCode(
	w http.ResponseWriter,
	req *FuyaoAuthorizeRequest,
	data map[string]interface{},
) {
	uri, err := s.GetRedirectURI(&req.AuthorizeRequest, data)
	if err != nil {
		zlog.LogErrorf("%s, err: %s", fuyaoerrors.ErrStrRedirectURIMissing, err)
		http.Error(w, fuyaoerrors.ErrStrRedirectURIMissing, fuyaoerrors.ErrStatusCode[fuyaoerrors.ErrRedirectURIMissing])
		return
	}

	w.Header().Set("Location", uri)
	w.WriteHeader(http.StatusFound)
	return
}

func (s *FuyaoAuthorizeServer) redirectAuthorizationCodeError(
	w http.ResponseWriter,
	req *FuyaoAuthorizeRequest,
	err error,
) {
	data, _, _ := s.GetErrorData(err)
	if req != nil {
		s.redirectAuthorizationCode(w, req, data)
	}
	return
}

func (s *FuyaoAuthorizeServer) generateTokenError(w http.ResponseWriter, err error) {
	errorData, code, errorHeader := s.GetErrorData(err)
	s.returnAccessToken(w, errorData, errorHeader, code)
	return
}

func (s *FuyaoAuthorizeServer) returnAccessToken(
	w http.ResponseWriter,
	data map[string]interface{},
	header http.Header,
	statusCode ...int,
) {
	w.Header().Set("Content-Type", "application/json;charset=UTF-8")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Cache-Control", "no-store")

	for key := range header {
		w.Header().Set(key, header.Get(key))
	}

	status := http.StatusOK
	if len(statusCode) > 0 && statusCode[0] > 0 {
		status = statusCode[0]
	}

	w.WriteHeader(status)
	err := json.NewEncoder(w).Encode(data)
	if err != nil {
		zlog.LogErrorf("%s, err: %v", fuyaoerrors.ErrStrFailToMarshalData, err)
	}

	return
}

func clearCookie(name string, w http.ResponseWriter) {
	cookie := &http.Cookie{
		Name:    name,
		Value:   "",
		Path:    "/",             // cookie的有效路径
		Expires: time.Unix(0, 0), // 过期时间设置为Unix时间戳0，即过去的时间
	}

	// 将cookie设置到HTTP响应头中
	http.SetCookie(w, cookie)
}
