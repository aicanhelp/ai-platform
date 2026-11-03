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

// Package httputil
package httputil

import (
	"crypto/tls"
	"crypto/x509"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/emicklei/go-restful/v3"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/zlog"
)

// ResponseJson Http Response
type ResponseJson struct {
	Code int32  `json:"code,omitempty"`
	Msg  string `json:"msg,omitempty"`
	Data any    `json:"data,omitempty"`
}

// GetResponseJson get restful response struct
func GetResponseJson(code int32, msg string, data any) *ResponseJson {
	return &ResponseJson{
		Code: code,
		Msg:  msg,
		Data: data,
	}
}

// GetDefaultSuccessResponseJson get default success response json
func GetDefaultSuccessResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: nil,
	}
}

// GetDefaultClientFailureResponseJson get default failure response json
func GetDefaultClientFailureResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.ClientError,
		Msg:  "bad request",
		Data: nil,
	}
}

// GetDefaultServerFailureResponseJson get default failure response json
func GetDefaultServerFailureResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.ServerError,
		Msg:  "remote server busy",
		Data: nil,
	}
}

// GetParamsEmptyErrorResponseJson get default resource empty response json
func GetParamsEmptyErrorResponseJson() *ResponseJson {
	return &ResponseJson{
		Code: constant.ClientError,
		Msg:  "parameters not found",
		Data: nil,
	}
}

var (
	clientInstance *http.Client
	clientOnce     sync.Once
)

// GetHttpConfig get http config
func GetHttpConfig(enableTLS bool) (*tls.Config, error) {
	if enableTLS {
		cert, err := tls.LoadX509KeyPair(constant.TLSCertPath, constant.TLSKeyPath)
		if err != nil {
			return nil, err
		}

		// Load CA cert
		caCert, err := os.ReadFile(constant.CAPath)
		if err != nil {
			return nil, err
		}
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)

		// Setup HTTPS client
		return &tls.Config{
			Certificates: []tls.Certificate{cert},
			MinVersion:   tls.VersionTLS12,
			MaxVersion:   tls.VersionTLS13,
			RootCAs:      caCertPool,
		}, nil
	} else {
		return &tls.Config{
			InsecureSkipVerify: true,
		}, nil
	}
}

// GetHttpTransport returns an HTTP transport
func GetHttpTransport(enableTLS bool) (*http.Transport, error) {
	tlsConfig, err := GetHttpConfig(enableTLS)
	if err != nil {
		return nil, err
	}
	return &http.Transport{
		TLSClientConfig: tlsConfig,
	}, nil
}

// GetDefaultHttpClient returns a singleton http.Client with custom TLS configuration.
func GetDefaultHttpClient() (*http.Client, error) {
	var err error
	clientOnce.Do(func() {
		enableTLS, err := IsHttpsEnabled()
		if err != nil {
			return
		}

		httpsTransport, err := GetHttpTransport(enableTLS)
		if err != nil {
			return
		}
		clientInstance = &http.Client{
			Timeout: time.Second * constant.DefaultHttpRequestSeconds,
		}
		clientInstance.Transport = httpsTransport
	})

	// Check if clientInstance was not properly initialized
	if clientInstance == nil {
		return nil, err
	}
	return clientInstance, nil
}

// IsHttpsEnabled checks whether TLS can be enabled
func IsHttpsEnabled() (bool, error) {
	paths := []string{
		constant.CAPath,
		constant.TLSCertPath,
		constant.TLSKeyPath,
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			if os.IsNotExist(err) {
				zlog.Warnf("tls file %s not exist, use http", p)
				return false, nil
			}
			zlog.Errorf("tls file %s exists but cannot be accessed: %v", p, err)
			return false, err
		}
	}

	return true, nil
}

// HandleNotAuthorized responds with 403 and unauthorized
func HandleNotAuthorized(response *restful.Response, message string, err error) {
	apiResponse := ResponseJson{
		Code: http.StatusForbidden,
		Msg:  message,
		Data: err,
	}
	zlog.Warn(message)
	response.WriteHeader(http.StatusForbidden)
	writeErr := response.WriteEntity(apiResponse)
	if err != nil {
		errorMessage := "Error Write Entity: "
		zlog.Error(errorMessage, writeErr)
	}
}
