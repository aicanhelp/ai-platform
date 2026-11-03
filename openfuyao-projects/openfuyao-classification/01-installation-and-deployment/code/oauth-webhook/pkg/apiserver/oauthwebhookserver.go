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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"

	"github.com/gorilla/mux"

	"openfuyao/oauth-webhook/cmd/oauth-webhook/app/config"
	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/httpserver"
	"openfuyao/oauth-webhook/pkg/webhooks/fuyao"
	"openfuyao/oauth-webhook/pkg/zlog"
)

// OAuthWebhookAPIServer is the true apiserver that handles requests
type OAuthWebhookAPIServer struct {
	Server  *http.Server
	Webhook *fuyao.InnerFuyaoWebhook
	Router  *mux.Router
}

// NewOAuthWebhookAPIServer inits a new webhook apiserver
func NewOAuthWebhookAPIServer(cfg *config.OAuthWebhookConfig, stopCh <-chan struct{}) (*OAuthWebhookAPIServer, error) {
	server := &http.Server{}

	if cfg.HttpServerConfig.HttpPort != 0 {
		server.Addr = fmt.Sprintf(":%d", cfg.HttpServerConfig.HttpPort)
	}

	if cfg.HttpServerConfig.HttpsPort != 0 {
		// set tls crt and key
		certificate, err := tls.LoadX509KeyPair(cfg.HttpServerConfig.TlsCertFile, cfg.HttpServerConfig.TlsPrivateKeyFile)
		if err != nil {
			zlog.LogErrorf("%s, err: %v", fuyaoerrors.ErrStrFailToLoadCert, err)
			return nil, fuyaoerrors.ErrFailToLoadCert
		}

		// set root CA
		caCrt, err := os.ReadFile(cfg.HttpServerConfig.MasterCAFile)
		if err != nil {
			zlog.LogErrorf("%s, err: %v", fuyaoerrors.ErrStrFailToLoadCert, err)
			return nil, fuyaoerrors.ErrFailToLoadCert
		}
		pool := x509.NewCertPool()
		pool.AppendCertsFromPEM(caCrt)

		server.TLSConfig = &tls.Config{
			Certificates: []tls.Certificate{certificate},
			MinVersion:   tls.VersionTLS13,
			ClientCAs:    pool,
			ClientAuth:   tls.RequireAndVerifyClientCert,
		}
		server.Addr = fmt.Sprintf(":%d", cfg.HttpServerConfig.HttpsPort)
	}

	webhook := fuyao.NewInnerFuyaoWebhook(cfg.JWTPrivateKey)

	router := mux.NewRouter()
	server.Handler = router

	return &OAuthWebhookAPIServer{
		Server:  server,
		Webhook: webhook,
		Router:  router,
	}, nil
}

// PrepareRun registers the router and the access logger
func (s *OAuthWebhookAPIServer) PrepareRun(stopCh <-chan struct{}) error {
	s.Router.Use(httpserver.AccessLoggingMiddleware)
	s.Router.HandleFunc("/oauth/tokenauth/fuyao", s.Webhook.WebhookHandler)
	return nil
}

// Run simply runs the server and gracefully shutdown the server if fuyaoerrors occur
func (s *OAuthWebhookAPIServer) Run(ctx context.Context) error {
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
