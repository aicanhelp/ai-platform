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

// Package httpserver defines the httpserver options, middlewares and responses
package httpserver

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"time"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/zlog"
)

// ServerOptions defines the config for httpserver
type ServerOptions struct {
	HttpPort          int    `json:"HttpPort"`
	HttpsPort         int    `json:"HttpsPort"`
	TlsCertFile       string `json:"TlsCertFile"`
	TlsPrivateKeyFile string `json:"TlsPrivateKeyFile"`
	RootCAFile        string `json:"RootCAFile"`
}

// NewDefaultHttpServerOptions inits the default httpserver option
func NewDefaultHttpServerOptions() *ServerOptions {
	return &ServerOptions{
		HttpPort:          9096,
		HttpsPort:         0,
		TlsCertFile:       "",
		TlsPrivateKeyFile: "",
		RootCAFile:        "",
	}
}

// Validate assures the httpserver option is valid
func (s *ServerOptions) Validate() []error {
	var errs []error
	if s.HttpsPort == constants.MinHttpPort && s.HttpPort == constants.MinHttpPort {
		errs = append(errs, fuyaoerrors.ErrInvalidHttpAndHttpsPort)
	}

	if s.HttpsPort > constants.MinHttpPort && s.HttpsPort < constants.MaxHttpPort {
		if s.TlsCertFile == "" {
			errs = append(errs, fuyaoerrors.ErrEmptyCertFile)
		} else {
			if _, err := os.Stat(s.TlsCertFile); err != nil {
				errs = append(errs, err)
			}
		}

		if s.TlsPrivateKeyFile == "" {
			errs = append(errs, fuyaoerrors.ErrEmptyPrivateKeyFile)
		} else {
			if _, err := os.Stat(s.TlsPrivateKeyFile); err != nil {
				errs = append(errs, err)
			}
		}

		if s.RootCAFile == "" {
			errs = append(errs, fuyaoerrors.ErrEmptyMasterCAFile)
		} else {
			if _, err := os.Stat(s.RootCAFile); err != nil {
				errs = append(errs, err)
			}
		}
	}

	return errs
}

// NewHttpServer inits the httpserver with the httpserveroption
func NewHttpServer(options *ServerOptions) (*http.Server, error) {
	if options == nil {
		zlog.LogError(fuyaoerrors.ErrStrHttpServerOptionsNil)
		return nil, fuyaoerrors.ErrHttpServerOptionsNil
	}

	// Configure server with timeouts to prevent goroutine leaks
	// ReadHeaderTimeout: closes connections that don't send request headers within 10 seconds
	// ReadTimeout: closes slow connections that don't complete request within 60 seconds
	// WriteTimeout: closes connections where response write takes longer than 60 seconds
	// IdleTimeout: closes idle connections after 120 seconds (OAuth flow needs time)
	server := &http.Server{
		Addr:              fmt.Sprintf(":%d", options.HttpPort),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	if options.HttpsPort != 0 {
		// load server.key and server.crt
		certificate, err := tls.LoadX509KeyPair(options.TlsCertFile, options.TlsPrivateKeyFile)
		if err != nil {
			zlog.LogErrorf("%s, err: %v", fuyaoerrors.ErrStrFailToLoadCert, err)
			return nil, fuyaoerrors.ErrFailToLoadCert
		}

		// load RootCA
		caCert, err := os.ReadFile(options.RootCAFile)
		if err != nil {
			zlog.LogErrorf("%s, err: %v", fuyaoerrors.ErrStrFailToLoadCert, err)
			return nil, fuyaoerrors.ErrFailToLoadCert
		}

		// create the cert pool
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)

		// configure the tls
		server.TLSConfig = &tls.Config{
			Certificates: []tls.Certificate{certificate},
			ClientAuth:   tls.RequireAndVerifyClientCert,
			MinVersion:   tls.VersionTLS13,
			ClientCAs:    caCertPool,
		}
		server.Addr = fmt.Sprintf(":%d", options.HttpsPort)
	}

	return server, nil
}
