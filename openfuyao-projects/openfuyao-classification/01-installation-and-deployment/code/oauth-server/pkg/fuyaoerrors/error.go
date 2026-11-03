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

// Package fuyaoerrors define all formatted errors
package fuyaoerrors

import (
	"errors"
	"net/http"
)

const (
	// ErrStrFailToDisplayLogin represents an internal runtime error indicating the inability to display the login page.
	ErrStrFailToDisplayLogin = "unable to display login page"

	// ErrStrUsernameOrPasswordMissing represents an internal runtime error indicating missing username or password
	// in the post request.
	ErrStrUsernameOrPasswordMissing = "missing username or password in the post request"

	// ErrStrFailToParseForm represents an internal runtime error indicating the inability to parse the form.
	ErrStrFailToParseForm = "cannot parse the form"

	// ErrStrLoginServiceDown represents an internal runtime error indicating that the login server returns an error.
	// It advises checking logs for details.
	ErrStrLoginServiceDown = "the login server returns error, please check logs for details"

	// ErrStrPasswordAuthenticationFailed represents an internal runtime error indicating incorrect
	// username or password during authentication.
	ErrStrPasswordAuthenticationFailed = "用户名或密码错误"

	// ErrStrPasswordResetFailed represents an internal runtime error indicating incorrect
	// username or password during authentication.
	ErrStrPasswordResetFailed = "当前密码错误"

	// ErrStrPasswordAuthenticationFailedWithCount represents an internal runtime error indicating incorrect
	// username or password during authentication, and will also return the remaining attempt counts
	ErrStrPasswordAuthenticationFailedWithCount = "用户名或密码错误，再输错 %s 次用户将锁定"

	// ErrStrPasswordAuthenticationFailedLocked represents an internal runtime error indicating incorrect
	// username or password during authentication, and will block the ip for sometime
	ErrStrPasswordAuthenticationFailedLocked = "用户名或密码错误，用户已经被锁定，请 %s 分钟后重试"

	// ErrStrPasswordSame represents an internal runtime error indicating that the new password is the same as the
	// original password.
	ErrStrPasswordSame = "新输入的密码和旧密码相同"

	// ErrStrPasswordTooWeak represents an internal runtime error indicating that the input new password cannot
	// pass the complexity check.
	ErrStrPasswordTooWeak = "the input new password cannot pass the complexity check"

	// ErrStrIdentityProviderIncorrect represents an internal runtime error indicating that the input identity_provider
	// is not valid.
	ErrStrIdentityProviderIncorrect = "the input identity_provider is not valid"

	// ErrStrRedirectURIIncorrect represents the redirect_uri is invalid
	ErrStrRedirectURIIncorrect = "the input redirect_uri is not valid"

	// ErrStrTokenTypeUnrecognized represents an internal runtime error indicating the input token type is not valid.
	ErrStrTokenTypeUnrecognized = "the input token type is not valid"

	// ErrStrFailToCreateSecret represents an internal runtime error failing to create a Kubernetes secret resource.
	ErrStrFailToCreateSecret = "fail to create k8s secret resource"

	// ErrStrFailToDeleteSecret represents an internal runtime error failing to delete a Kubernetes secret resource.
	ErrStrFailToDeleteSecret = "fail to delete k8s secret resource"

	// ErrStrFailToGetSecret represents an internal runtime error failing to get a Kubernetes secret resource.
	ErrStrFailToGetSecret = "fail to get k8s secret resource"

	// ErrStrFailToPatchSecret represents an internal runtime error failing to patch a Kubernetes secret resource.
	ErrStrFailToPatchSecret = "fail to patch k8s secret resource"

	// ErrStrNotImplemented represents an internal runtime error indicating that a function is not implemented.
	ErrStrNotImplemented = "function is not implemented"

	// ErrStrFailToMarshalData represents an internal runtime error indicating the inability to marshal data.
	ErrStrFailToMarshalData = "cannot marshal data"

	// ErrStrFailToUnmarshalData represents an internal runtime error indicating the inability to unmarshal data.
	ErrStrFailToUnmarshalData = "cannot unmarshal data"

	// ErrStrNotFirstLogin represents an internal runtime error indicating that the user is not logging in
	// for the first time and should go to the login page.
	ErrStrNotFirstLogin = "this user is not the first time logging in, please go to the login page"

	// ErrStrNotLogin represents an internal runtime error indicating that the user is not logged in and
	// unauthorized to perform any action.
	ErrStrNotLogin = "user not logged in, unauthorized to do anything"

	// ErrStrLoginBlocked represents an internal runtime error indicating that the request is blocked due to
	// multiple failed login attempts.
	ErrStrLoginBlocked = "账户锁定中，请等待 %s 分钟后重试"

	// ErrStrInvalidHttpAndHttpsPort represents an HTTP server error indicating that both HTTP and HTTPS ports
	// cannot be invalid at the same time.
	ErrStrInvalidHttpAndHttpsPort = "http and https port cannot be invalid at the same time"

	// ErrStrWritingHttpHeader represents an HTTP server error failing to write back to http header
	ErrStrWritingHttpHeader = "fail to write content back the http header"

	// ErrStrEmptyCertFile represents an HTTP server error indicating that the TLS cert file is empty to
	// serve HTTPS requests.
	ErrStrEmptyCertFile = "the tls cert file is empty to serve https requests"

	// ErrStrEmptyPrivateKeyFile represents an HTTP server error indicating that the TLS private key file is empty to
	// serve HTTPS requests.
	ErrStrEmptyPrivateKeyFile = "the tls private key file is empty to serve https requests"

	// ErrStrEmptyMasterCAFile represents an HTTP server error indicating that the master CA file is empty to
	// serve HTTPS requests.
	ErrStrEmptyMasterCAFile = "the master CA file is empty to serve https requests"

	// ErrStrFailToLoadCert represents an HTTP server error indicating that the cert and private key do not match.
	ErrStrFailToLoadCert = "the cert and private key does not match"

	// ErrStrOAuthServerConfigFileMissing represents an OAuth server option error indicating that
	// the config file is missing to start the HTTP server.
	ErrStrOAuthServerConfigFileMissing = "the config file is missing to start the httpserver"

	// ErrStrJWTPrivateKeyMissing represents an OAuth server option error indicating that the JWT private key
	// is missing to start the HTTP server.
	ErrStrJWTPrivateKeyMissing = "the JWT private key is missing to start the httpserver"

	// ErrStrHttpServerConfigMissing represents an OAuth server option error indicating that the HTTP server config is
	// incomplete to start the server.
	ErrStrHttpServerConfigMissing = "the httpserver config is incomplete to start the server"

	// ErrStrIPProtectorConfigMissing represents an OAuth server option error indicating that the IP protector config
	// is incomplete to start the server.
	ErrStrIPProtectorConfigMissing = "the ip protector config is incomplete to start the server"

	// ErrStrIdpLoginStoreConfigMissing represents an OAuth server option error indicating that the IDP login store
	// config is incomplete to start the server.
	ErrStrIdpLoginStoreConfigMissing = "the idploginstore config is incomplete to start the server"

	// ErrStrLoginConfigMissing represents an OAuth server option error indicating that the login config is
	// incomplete to start the server.
	ErrStrLoginConfigMissing = "the login config is incomplete to start the server"

	// ErrStrClientInfoMissing represents an OAuth server option error indicating that the client-info is
	// incomplete to start the server.
	ErrStrClientInfoMissing = "the client-info is incomplete to start the server"

	// ErrStrRequestMethodNotAllowed represents the error when requesting method is illegal
	ErrStrRequestMethodNotAllowed = "request method not allowed"

	// ErrStrRedirectURIMissing represents the redirect uri is missing
	ErrStrRedirectURIMissing = "the redirect uri is missing or incomplete"

	// ErrStrInvalidThen represents the then param is invalid
	ErrStrInvalidThen = "错误的重定向地址，请从首页重新登录"

	// ErrStrHttpServerOptionsNil represents an error when HTTP server options are nil
	ErrStrHttpServerOptionsNil = "http server options cannot be nil"
)

var (
	// ErrFailToDisplayLogin represents an internal runtime error indicating the inability to display the login page.
	ErrFailToDisplayLogin = errors.New(ErrStrFailToDisplayLogin)

	// ErrUsernameOrPasswordMissing represents an internal runtime error indicating missing username or password
	// in the post request.
	ErrUsernameOrPasswordMissing = errors.New(ErrStrUsernameOrPasswordMissing)

	// ErrFailToParseForm represents an internal runtime error indicating the inability to parse the form.
	ErrFailToParseForm = errors.New(ErrStrFailToParseForm)

	// ErrLoginServiceDown represents an internal runtime error indicating that the login server returns an error.
	// It advises checking logs for details.
	ErrLoginServiceDown = errors.New(ErrStrLoginServiceDown)

	// ErrPasswordAuthenticationFailed represents an internal runtime error indicating incorrect username or
	// password during authentication.
	ErrPasswordAuthenticationFailed = errors.New(ErrStrPasswordAuthenticationFailed)

	// ErrPasswordResetFailed represents an internal runtime error indicating incorrect username or
	// password during authentication.
	ErrPasswordResetFailed = errors.New(ErrStrPasswordResetFailed)

	// ErrPasswordSame represents an internal runtime error indicating that the new password is the same as
	// the original password.
	ErrPasswordSame = errors.New(ErrStrPasswordSame)

	// ErrPasswordTooWeak represents an internal runtime error indicating that the input new password cannot
	// pass the complexity check.
	ErrPasswordTooWeak = errors.New(ErrStrPasswordTooWeak)

	// ErrIdentityProviderIncorrect represents an internal runtime error indicating that the input
	// identity_provider is not valid.
	ErrIdentityProviderIncorrect = errors.New(ErrStrIdentityProviderIncorrect)

	// ErrRedirectURIIncorrect represents redirect_uri is not valid
	ErrRedirectURIIncorrect = errors.New(ErrStrRedirectURIIncorrect)

	// ErrTokenTypeUnrecognized represents an internal runtime error indicating that the input token type is not valid.
	ErrTokenTypeUnrecognized = errors.New(ErrStrTokenTypeUnrecognized)

	// ErrFailToCreateSecret represents an internal runtime error indicating failure to create a
	// Kubernetes secret resource.
	ErrFailToCreateSecret = errors.New(ErrStrFailToCreateSecret)

	// ErrFailToDeleteSecret represents an internal runtime error indicating failure to delete a
	// Kubernetes secret resource.
	ErrFailToDeleteSecret = errors.New(ErrStrFailToDeleteSecret)

	// ErrFailToGetSecret represents an internal runtime error indicating failure to get a
	// Kubernetes secret resource.
	ErrFailToGetSecret = errors.New(ErrStrFailToGetSecret)

	// ErrFailToPatchSecret represents an internal runtime error indicating failure to patch a
	// Kubernetes secret resource.
	ErrFailToPatchSecret = errors.New(ErrStrFailToPatchSecret)

	// ErrNotImplemented represents an internal runtime error indicating that a function is not implemented.
	ErrNotImplemented = errors.New(ErrStrNotImplemented)

	// ErrFailToMarshalData represents an internal runtime error indicating the inability to marshal data.
	ErrFailToMarshalData = errors.New(ErrStrFailToMarshalData)

	// ErrFailToUnmarshalData represents an internal runtime error indicating the inability to unmarshal data.
	ErrFailToUnmarshalData = errors.New(ErrStrFailToUnmarshalData)

	// ErrNotFirstLogin represents an internal runtime error indicating that the user is not logging in for the
	// first time and should go to the login page.
	ErrNotFirstLogin = errors.New(ErrStrNotFirstLogin)

	// ErrNotLogin represents an internal runtime error indicating that the user is not logged in and unauthorized
	// to perform any action.
	ErrNotLogin = errors.New(ErrStrNotLogin)

	// ErrLoginBlocked represents an internal runtime error indicating that the request is blocked due to multiple
	// failed login attempts.
	ErrLoginBlocked = errors.New(ErrStrLoginBlocked)

	// ErrInvalidHttpAndHttpsPort represents an HTTP server error indicating that both HTTP and HTTPS ports cannot
	// be invalid at the same time.
	ErrInvalidHttpAndHttpsPort = errors.New(ErrStrInvalidHttpAndHttpsPort)

	// ErrWritingHttpHeader represents an HTTP server error failing to write back to http header
	ErrWritingHttpHeader = errors.New(ErrStrWritingHttpHeader)

	// ErrEmptyCertFile represents an HTTP server error indicating that the TLS cert file is empty to
	// serve HTTPS requests.
	ErrEmptyCertFile = errors.New(ErrStrEmptyCertFile)

	// ErrEmptyPrivateKeyFile represents an HTTP server error indicating that the TLS private key file is empty to
	// serve HTTPS requests.
	ErrEmptyPrivateKeyFile = errors.New(ErrStrEmptyPrivateKeyFile)

	// ErrEmptyMasterCAFile represents an HTTP server error indicating that the master CA file is empty to
	// serve HTTPS requests.
	ErrEmptyMasterCAFile = errors.New(ErrStrEmptyMasterCAFile)

	// ErrFailToLoadCert represents an HTTP server error indicating that the cert and private key do not match.
	ErrFailToLoadCert = errors.New(ErrStrFailToLoadCert)

	// ErrIntExitSignal represents an internal exit signal error with the code 255.
	ErrIntExitSignal = 255

	// ErrOAuthServerConfigFileMissing represents an OAuth server option error indicating that the config file is
	// missing to start the HTTP server.
	ErrOAuthServerConfigFileMissing = errors.New(ErrStrOAuthServerConfigFileMissing)

	// ErrJWTPrivateKeyMissing represents an OAuth server option error indicating that the JWT private key is
	// missing to start the HTTP server.
	ErrJWTPrivateKeyMissing = errors.New(ErrStrJWTPrivateKeyMissing)

	// ErrHttpServerConfigMissing represents an OAuth server option error indicating that the HTTP server config is
	// incomplete to start the server.
	ErrHttpServerConfigMissing = errors.New(ErrStrHttpServerConfigMissing)

	// ErrIPProtectorConfigMissing represents an OAuth server option error indicating that the IP protector config is
	// incomplete to start the server.
	ErrIPProtectorConfigMissing = errors.New(ErrStrIPProtectorConfigMissing)

	// ErrIdpLoginStoreConfigMissing represents an OAuth server option error indicating that the IDP login store
	// config is incomplete to start the server.
	ErrIdpLoginStoreConfigMissing = errors.New(ErrStrIdpLoginStoreConfigMissing)

	// ErrLoginConfigMissing represents an OAuth server option error indicating that the login config is incomplete
	// to start the server.
	ErrLoginConfigMissing = errors.New(ErrStrLoginConfigMissing)

	// ErrClientInfoMissing represents an OAuth server option error indicating that the client-info is incomplete
	// to start the server.
	ErrClientInfoMissing = errors.New(ErrStrClientInfoMissing)

	// ErrRequestMethodNotAllowed represents the error when requesting method is illegal
	ErrRequestMethodNotAllowed = errors.New(ErrStrRequestMethodNotAllowed)

	// ErrRedirectURIMissing represents the redirect uri is missing
	ErrRedirectURIMissing = errors.New(ErrStrRedirectURIMissing)

	// ErrHttpServerOptionsNil represents an error when HTTP server options are nil
	ErrHttpServerOptionsNil = errors.New(ErrStrHttpServerOptionsNil)
)

// ErrStatusCode is the mapper from error to http return error
var ErrStatusCode = map[error]int{
	ErrFailToDisplayLogin:           http.StatusInternalServerError,
	ErrUsernameOrPasswordMissing:    http.StatusBadRequest,
	ErrFailToParseForm:              http.StatusInternalServerError,
	ErrLoginServiceDown:             http.StatusInternalServerError,
	ErrPasswordAuthenticationFailed: http.StatusUnauthorized,
	ErrPasswordResetFailed:          http.StatusUnauthorized,
	ErrPasswordSame:                 http.StatusUnauthorized,
	ErrPasswordTooWeak:              http.StatusBadRequest,
	ErrIdentityProviderIncorrect:    http.StatusBadRequest,
	ErrTokenTypeUnrecognized:        http.StatusBadRequest,
	ErrFailToCreateSecret:           http.StatusInternalServerError,
	ErrFailToDeleteSecret:           http.StatusInternalServerError,
	ErrFailToGetSecret:              http.StatusInternalServerError,
	ErrFailToPatchSecret:            http.StatusInternalServerError,
	ErrNotImplemented:               http.StatusNotImplemented,
	ErrFailToMarshalData:            http.StatusInternalServerError,
	ErrFailToUnmarshalData:          http.StatusInternalServerError,
	ErrNotFirstLogin:                http.StatusConflict,
	ErrNotLogin:                     http.StatusUnauthorized,
	ErrLoginBlocked:                 http.StatusUnauthorized,
	ErrRedirectURIMissing:           http.StatusBadRequest,
	ErrHttpServerOptionsNil:         http.StatusInternalServerError,
}
