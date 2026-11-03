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

// Package config defines the overall configurations for OAuthServerAPIServer
package config

import (
	"time"

	"openfuyao/oauth-server/pkg/config"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/httpserver"
	"openfuyao/oauth-server/pkg/zlog"
)

// OAuthServerAPIServerConfig stores the necessary configuration options for OAuth server
type OAuthServerAPIServerConfig struct {
	// config for http httpserver
	HttpServerConfig *httpserver.ServerOptions `json:"HttpServerConfig"`

	// idpLoginStore config
	IDPLoginStoreConfig *IDPLoginStoreConfig `json:"IDPLoginStoreConfig"`

	// ipprotector config
	IPProtectorConfig *IPProtectorConfig `json:"IPProtectorConfig"`

	// k8s config
	K8sConfig *config.KubernetesConfig `json:"K8SConfig"`

	// login config
	LoginConfig *LoginConfig `json:"LoginConfig"`

	// OAuthAPIServerConfig for the inner oauth server options
	OAuthServerConfig *OAuthServerConfig `json:"OAuthServerConfig"`
}

// NewDefaultOAuthAPIServerServerConfig inits the config
func NewDefaultOAuthAPIServerServerConfig() *OAuthServerAPIServerConfig {
	return &OAuthServerAPIServerConfig{
		HttpServerConfig:    httpserver.NewDefaultHttpServerOptions(),
		K8sConfig:           config.NewKubernetesConfig(),
		LoginConfig:         newDefaultLoginConfig(),
		IPProtectorConfig:   newDefaultIPProtectorConfig(),
		IDPLoginStoreConfig: newIDPLoginStoreConfig(),
		OAuthServerConfig:   newOAuthServerConfig(),
	}
}

// Validate validates the config
func (c *OAuthServerAPIServerConfig) Validate() []error {
	var errs []error

	// validate each part of the config
	if tmpErrs := c.HttpServerConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	if tmpErrs := c.K8sConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	if tmpErrs := c.LoginConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	if tmpErrs := c.IPProtectorConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	if tmpErrs := c.IDPLoginStoreConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	if tmpErrs := c.OAuthServerConfig.Validate(); len(tmpErrs) > 0 {
		errs = append(errs, tmpErrs...)
	}

	return errs
}

// Complete ensures the completeness of the config
func (c *OAuthServerAPIServerConfig) Complete() *OAuthServerAPIServerConfig {
	defaultConfig := NewDefaultOAuthAPIServerServerConfig()

	if c.HttpServerConfig == nil {
		c.HttpServerConfig = defaultConfig.HttpServerConfig
	}
	if c.K8sConfig == nil {
		c.K8sConfig = defaultConfig.K8sConfig
	}
	if c.LoginConfig == nil {
		c.LoginConfig = defaultConfig.LoginConfig
	}
	if c.IPProtectorConfig == nil {
		c.IPProtectorConfig = defaultConfig.IPProtectorConfig
	}
	if c.IDPLoginStoreConfig == nil {
		c.IDPLoginStoreConfig = defaultConfig.IDPLoginStoreConfig
	}
	if c.OAuthServerConfig == nil {
		c.OAuthServerConfig = defaultConfig.OAuthServerConfig
	}

	return c
}

// LoginConfig defines all config used by fuyao login provider
type LoginConfig struct {
	Provider string `json:"Provider"`
}

func newDefaultLoginConfig() *LoginConfig {
	return &LoginConfig{
		Provider: "fuyaoPaswordProvider",
	}
}

// Validate ensures legality of LoginConfig
func (l *LoginConfig) Validate() []error {
	var errs []error

	if l.Provider == "" {
		errs = append(errs, fuyaoerrors.ErrLoginConfigMissing)
	}

	return errs
}

// IPProtectorConfig defines all config used by ipprotector
type IPProtectorConfig struct {
	FailTimes    int           `json:"FailTimes"`
	FailDuration time.Duration `json:"FailDuration"`
	LockDuration time.Duration `json:"LockDuration"`
}

func newDefaultIPProtectorConfig() *IPProtectorConfig {
	const (
		failDurationMins = 5
		lockDurationMins = 30
	)
	return &IPProtectorConfig{
		FailTimes:    5,
		FailDuration: time.Minute * failDurationMins,
		LockDuration: time.Minute * lockDurationMins,
	}
}

// Validate either all fields are not set (0) or all fields are set (not 0) is legal
func (i *IPProtectorConfig) Validate() []error {
	var errs []error
	if i.FailTimes == 0 && i.FailDuration == 0 || i.LockDuration == 0 {
		return nil
	}

	if i.FailTimes == 0 || i.FailDuration == 0 || i.LockDuration == 0 {
		errs = append(errs, fuyaoerrors.ErrIPProtectorConfigMissing)
	}

	return errs
}

// IDPLoginStoreConfig configures the fuyaostore that temporally saves the user info
type IDPLoginStoreConfig struct {
	SessionName    string `json:"SessionName"`
	SessionMaxAge  int    `json:"SessionMaxAge"`
	CsrfCookieName string `json:"CsrfCookieName"`
	SigningKey     []byte `json:"SigningKey"`
	EncryptionKey  []byte `json:"EncryptionKey"`
}

func newIDPLoginStoreConfig() *IDPLoginStoreConfig {
	return &IDPLoginStoreConfig{
		SessionName:    "idpLogin",
		SessionMaxAge:  300,
		CsrfCookieName: "csrf",
	}
}

// Validate ensures that IDPLoginStore is legal
func (s *IDPLoginStoreConfig) Validate() []error {
	var errs []error
	if s.SessionName == "" {
		zlog.LogError("session name is missing to set the cookie")
		errs = append(errs, fuyaoerrors.ErrIdpLoginStoreConfigMissing)
	}

	if s.SessionMaxAge <= 0 {
		zlog.LogWarn("the session-cookie will not expire")
	}

	if s.SigningKey == nil {
		zlog.LogWarn("no signing key is provided to store the idp login state cookie")
	}

	if s.EncryptionKey == nil {
		zlog.LogWarn("no encryption key is provided to store the idp login state cookie")
	}

	return errs
}

// OAuthServerConfig configures the inner oauth2 server
type OAuthServerConfig struct {
	CodeTokenNamespace string            `json:"CodeTokenNamespace"`
	AuthCodeExp        time.Duration     `json:"AuthCodeExp"`
	AccessTokenExp     time.Duration     `json:"AccessTokenExp"`
	RefreshTokenExp    time.Duration     `json:"RefreshTokenExp"`
	IsGenerateRefresh  bool              `json:"IsGenerateRefresh"`
	JWTKeyID           string            `json:"JWTKeyID"`
	JWTPrivateKey      []byte            `json:"JWTPrivateKey"`
	ClientMapper       map[string]string `json:"ClientMapper"`
}

func newOAuthServerConfig() *OAuthServerConfig {
	const (
		authCodeExpMins      = 5
		accessTokenExpHours  = 2
		refreshTokenExpHours = 2
	)

	return &OAuthServerConfig{
		CodeTokenNamespace: "oauth-code-token",
		AuthCodeExp:        time.Minute * authCodeExpMins,
		AccessTokenExp:     time.Hour * accessTokenExpHours,
		RefreshTokenExp:    time.Hour * refreshTokenExpHours,
		IsGenerateRefresh:  false,
		JWTKeyID:           "access_token_sign_key",
		ClientMapper: map[string]string{
			"console": "console-password",
		},
	}
}

// Validate ensures OAuthServerConfig is legal
func (o *OAuthServerConfig) Validate() []error {
	var errs []error
	if o.CodeTokenNamespace == "" {
		zlog.LogWarn("namespace to store code/token is not provided, using default instead")
		o.CodeTokenNamespace = "default"
	}

	if o.AuthCodeExp <= 0 {
		zlog.LogWarn("the authorization code will not expire")
	}

	if o.AccessTokenExp <= 0 {
		zlog.LogWarn("the access token will not expire")
	}

	if o.JWTKeyID == "" {
		zlog.LogWarn("the key id for JWT header is not provided")
	}

	if o.JWTPrivateKey == nil {
		zlog.LogError(fuyaoerrors.ErrStrJWTPrivateKeyMissing)
		errs = append(errs, fuyaoerrors.ErrJWTPrivateKeyMissing)
	}

	if o.ClientMapper == nil || len(o.ClientMapper) == 0 {
		zlog.LogError("no client info is provided to oauth-server so it cannot authenticate anything")
		errs = append(errs, fuyaoerrors.ErrClientInfoMissing)
	}

	return errs
}
