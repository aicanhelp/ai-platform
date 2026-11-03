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

// Package configs define necessary http configs
package config

import "openfuyao/oauth-webhook/pkg/httpserver"

// OAuthWebhookConfig stores the necessary configuration options for OAuth webhook
type OAuthWebhookConfig struct {
	// configs for http httpserver
	HttpServerConfig *httpserver.ServerOptions `json:"HttpServerConfig"`
	// configs for jwt private key
	JWTPrivateKey []byte `json:"JWTPrivateKey"`
}

// NewDefaultOAuthWebhookConfig inits the config
func NewDefaultOAuthWebhookConfig() *OAuthWebhookConfig {
	return &OAuthWebhookConfig{
		HttpServerConfig: httpserver.NewDefaultHttpServerOptions(),
	}
}

// Validate validates the config
func (c *OAuthWebhookConfig) Validate() []error {
	var errs []error

	// validate each part of config
	if tmpErrs := c.HttpServerConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	return errs
}

// Complete ensures the completeness of the config
func (c *OAuthWebhookConfig) Complete() *OAuthWebhookConfig {
	defaultConfig := NewDefaultOAuthWebhookConfig()
	if c.JWTPrivateKey == nil {
		c.JWTPrivateKey = defaultConfig.JWTPrivateKey
	}

	if c.HttpServerConfig == nil {
		c.HttpServerConfig = defaultConfig.HttpServerConfig
	}

	return c
}
