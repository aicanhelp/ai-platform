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

package httputil

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"time"

	"helm.sh/helm/v3/pkg/repo"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/zlog"
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

// GetHarborHttpClientByRepo returns a singleton http.Client with custom TLS configuration.
func GetHarborHttpClientByRepo(repoEntry *repo.Entry) (*http.Client, error) {
	var err error
	httpClient := &http.Client{
		Timeout: time.Second * constant.DefaultHttpRequestSeconds,
	}

	tlsConfig, err := GetHarborHTTPConfig(repoEntry)
	if err != nil {
		return nil, err
	}

	httpClient.Transport = &http.Transport{
		TLSClientConfig: tlsConfig,
	}

	return httpClient, nil
}

// IsHttpsEnabled checks whether TLS can be enabled
func IsHttpsEnabled(certPath, keyPath, caPath string) (bool, error) {
	paths := []string{
		caPath,
		certPath,
		keyPath,
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

// GetHttpConfig get http config
func GetHttpConfig(certPath, keyPath, caPath string, enableTLS bool) (*tls.Config, error) {
	if enableTLS {
		cert, err := tls.LoadX509KeyPair(certPath, keyPath)
		if err != nil {
			return nil, err
		}

		// Load CA cert
		caCert, err := os.ReadFile(caPath)
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
			ClientCAs:    caCertPool,
			ClientAuth:   tls.VerifyClientCertIfGiven,
		}, nil
	} else {
		return &tls.Config{
			InsecureSkipVerify: true,
		}, nil
	}
}

// IsHttpsEnabledByPEM checks whether TLS can be enabled
func IsHttpsEnabledByPEM(certPEM, keyPEM, caPEM string) (bool, error) {
	if certPEM == "" {
		zlog.Warn("tls cert not provided, use http")
		return false, nil
	}
	if keyPEM == "" {
		zlog.Warn("tls key not provided, use http")
		return false, nil
	}
	if caPEM == "" {
		zlog.Warn("tls ca not provided, use http")
		return false, nil
	}

	return true, nil
}

// GetHttpConfigByPEM creates tls.Config from PEM strings (instead of file paths)
func GetHttpConfigByPEM(certPEM, keyPEM, caPEM string, enableTLS bool) (*tls.Config, error) {
	if enableTLS {
		// 解析证书和私钥
		cert, err := tls.X509KeyPair([]byte(certPEM), []byte(keyPEM))
		if err != nil {
			return nil, fmt.Errorf("failed to load x509 key pair: %w", err)
		}

		// 解析 CA
		caCertPool := x509.NewCertPool()
		if ok := caCertPool.AppendCertsFromPEM([]byte(caPEM)); !ok {
			return nil, fmt.Errorf("failed to append CA cert")
		}

		// Setup HTTPS client
		return &tls.Config{
			Certificates: []tls.Certificate{cert},
			MinVersion:   tls.VersionTLS12,
			MaxVersion:   tls.VersionTLS13,
			RootCAs:      caCertPool,
			ClientAuth:   tls.VerifyClientCertIfGiven,
		}, nil
	}

	// 非 TLS 模式
	return &tls.Config{
		InsecureSkipVerify: true,
	}, nil
}

// GetHarborHTTPConfig eturns a TLS config with optional certs and CA
func GetHarborHTTPConfig(repoEntry *repo.Entry) (*tls.Config, error) {
	repoName := repoEntry.Name
	if repoName == "openfuyao" {
		return nil, nil
	}
	enabled, err := IsHttpsEnabledByPEM(repoEntry.CertFile, repoEntry.KeyFile, repoEntry.CAFile)
	enabled = enabled && repoEntry.InsecureSkipTLSverify
	if err != nil {
		return nil, fmt.Errorf("error get https enabling status")
	}
	tlsClientConfig, err := GetHttpConfigByPEM(repoEntry.CertFile, repoEntry.KeyFile, repoEntry.CAFile, enabled)
	if err != nil {
		return nil, fmt.Errorf("error get http config")
	}
	if repoName == "local" && enabled {
		tlsClientConfig.MinVersion = tls.VersionTLS12
		tlsClientConfig.MaxVersion = tls.VersionTLS12
		tlsClientConfig.CipherSuites = []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
		}
	}

	return tlsClientConfig, nil
}
