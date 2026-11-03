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

// Package apiserver inits all the necessary components
package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/gorilla/csrf"
	"github.com/gorilla/mux"

	overallconfigs "openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/config"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/fuyaostore"
	"openfuyao/oauth-server/pkg/httpserver"
	"openfuyao/oauth-server/pkg/idp/fuyaopassword"
	"openfuyao/oauth-server/pkg/oauth2"
	"openfuyao/oauth-server/pkg/protector"
	"openfuyao/oauth-server/pkg/sessions"
	"openfuyao/oauth-server/pkg/zlog"
)

// OAuthServerAPIServer is the true apiserver that handles requests
type OAuthServerAPIServer struct {
	Server      *http.Server
	Router      *mux.Router
	Login       *fuyaopassword.Login
	OAuthServer *oauth2.FuyaoAuthorizeServer
	Cfg         *overallconfigs.OAuthServerAPIServerConfig
}

// NewOAuthServerAPIServer inits a new oauthserver apiserver
func NewOAuthServerAPIServer(
	cfg *overallconfigs.OAuthServerAPIServerConfig,
	stopCh <-chan struct{},
) (*OAuthServerAPIServer, error) {
	// init httpserver
	server, err := httpserver.NewHttpServer(cfg.HttpServerConfig)
	if err != nil {
		return nil, err
	}
	router := mux.NewRouter()
	server.Handler = router

	// init each component
	k8sClient := config.GetKubernetesClient(cfg.K8sConfig)
	dynamicClient := config.GetDynamicClient(cfg.K8sConfig)
	idpLoginStore := sessions.NewSessionStore(
		cfg.IDPLoginStoreConfig.SessionName, cfg.IDPLoginStoreConfig.SessionMaxAge,
		cfg.IDPLoginStoreConfig.SigningKey, cfg.IDPLoginStoreConfig.EncryptionKey)
	logUserProtector := protector.NewLoginUserProtector(dynamicClient, cfg.IPProtectorConfig)
	tokenStore := fuyaostore.NewK8sSecretStore(k8sClient, cfg.OAuthServerConfig.CodeTokenNamespace)
	login := fuyaopassword.NewLogin(idpLoginStore, tokenStore, logUserProtector, cfg)
	oauthServer := oauth2.NewOAuthServer(idpLoginStore, tokenStore, cfg.OAuthServerConfig,
		cfg.IDPLoginStoreConfig.CsrfCookieName)

	return &OAuthServerAPIServer{
		Server:      server,
		Router:      router,
		Login:       login,
		OAuthServer: oauthServer,
		Cfg:         cfg,
	}, nil
}

// PrepareRun registers the router and the access logger
func (s *OAuthServerAPIServer) PrepareRun(stopCh <-chan struct{}) error {
	// logging
	s.Router.Use(httpserver.AccessLoggingMiddleware)

	// csrf
	csrfCookieName := s.Cfg.IDPLoginStoreConfig.CsrfCookieName
	if csrfCookieName == "" {
		csrfCookieName = "csrf"
	}
	CSRF := csrf.Protect(s.Cfg.IDPLoginStoreConfig.EncryptionKey, csrf.SameSite(csrf.SameSiteStrictMode),
		csrf.Path("/"), csrf.HttpOnly(true), csrf.MaxAge(0), csrf.CookieName(csrfCookieName),
		csrf.ErrorHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var requestBody fuyaopassword.LoginRequest
			if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
				httpserver.RespondWithStatusMsg(w, http.StatusBadRequest, 0, fuyaoerrors.ErrStrFailToUnmarshalData)
				return
			}
			then := requestBody.Then

			encodedErrString := "您的登录cookie已经过期，请重新登录"
			encodedThen := url.QueryEscape(then)
			redirect := fmt.Sprintf("%s?then=%s&error=%s", r.URL.String(), encodedThen, encodedErrString)

			// redirect to GET handleLogin
			http.Redirect(w, r, redirect, http.StatusFound)
		})))
	loginRouter := s.Router.PathPrefix(constants.FuyaoLoginEndpoint).Subrouter()
	confirmRouter := s.Router.PathPrefix(constants.FuyaoPasswordConfirmEndpoint).Subrouter()
	loginRouter.Use(CSRF)
	confirmRouter.Use(CSRF)

	loginRouter.HandleFunc("", s.Login.LoginHandler)
	s.Router.HandleFunc(constants.FuyaoLogoutEndpoint, s.OAuthServer.SingleLogoutHandler)
	confirmRouter.HandleFunc("", s.Login.PasswordConfirmHandler)
	s.Router.HandleFunc(constants.FuyaoPasswordModifyEndpoint, s.Login.PasswordResetHandler)
	s.Router.HandleFunc(constants.FuyaoOAuthAuthorizeEndpoint, s.OAuthServer.OAuthAuthorizeHandler)
	s.Router.HandleFunc(constants.FuyaoOAuthTokenEndpoint, s.OAuthServer.OAuthTokenHandler)
	return nil
}

// Run simply runs the server and gracefully shutdown the server if errors occur
func (s *OAuthServerAPIServer) Run(ctx context.Context) error {
	shutdownCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-ctx.Done()
		err := s.Server.Shutdown(shutdownCtx)
		zlog.LogErrorf("server shuts down, err: %v", err)
	}()

	zlog.LogInfof("Start listening on %s", s.Server.Addr)
	var err error
	if s.Server.TLSConfig != nil {
		err = s.Server.ListenAndServeTLS("", "")
	} else {
		err = s.Server.ListenAndServe()
	}

	return err
}
