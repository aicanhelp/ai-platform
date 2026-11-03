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

// Package fuyaoerrors defines all used fuyaoerrors
package fuyaoerrors

import "errors"

const (
	// ErrStrFailToDecodeTokenReview defines the error message for failing to decode from k8s TokenReview resource
	ErrStrFailToDecodeTokenReview = "cannot decode from k8s TokenReview resource"
	// ErrStrFailToEncodeTokenReviewResponse defines the error message for failing to respond TokenReview to web
	ErrStrFailToEncodeTokenReviewResponse = "fail to encode the TokenReview response to web"
	// ErrStrJWTValidationFailed defines the error message for failing to validate access token through JWT
	ErrStrJWTValidationFailed = "fail to validate access token through JWT"
	// ErrStrRequestMethodNotAllowed defines the error message for the request method not being allowed
	ErrStrRequestMethodNotAllowed = "the request method is not allowed, only post method is allowed"
	// ErrStrInvalidHttpAndHttpsPort defines the error message for http and https port cannot be invalid simultaneously
	ErrStrInvalidHttpAndHttpsPort = "http and https port cannot be invalid at the same time"
	// ErrStrInvalidHttpPort defines the error message that http port is invalid
	ErrStrInvalidHttpPort = "http port is out of range"
	// ErrStrInvalidHttpsPort defines the error message that https port is invalid
	ErrStrInvalidHttpsPort = "https port is out of range"
	// ErrStrEmptyCertFile defines the error message for the tls cert file being empty to serve https requests
	ErrStrEmptyCertFile = "the tls cert file is empty to serve https requests"
	// ErrStrEmptyPrivateKeyFile defines the error message for tls private key file being empty to serve https requests
	ErrStrEmptyPrivateKeyFile = "the tls private key file is empty to serve https requests"
	// ErrStrEmptyMasterCAFile defines the error message for the master CA file being empty to serve https requests
	ErrStrEmptyMasterCAFile = "the master CA file is empty to serve https requests"
	// ErrStrFailToLoadCert defines the error message for the cert and private key does not match
	ErrStrFailToLoadCert = "the cert and private key does not match"
	// ErrStrOAuthWebhookConfigFileMissing defines the error message for the configfile missing to start the httpserver
	ErrStrOAuthWebhookConfigFileMissing = "the configfile is missing to start the httpserver"
	// ErrStrFileNotExist defines the error message for the target file missing
	ErrStrFileNotExist = "the target file does not exist"
	// ErrIntExitSignal is an internal error code for exit signal
	ErrIntExitSignal = 255
)

var (
	// ErrFailToDecodeTokenReview represents the error for failing to decode from k8s TokenReview resource
	ErrFailToDecodeTokenReview = errors.New(ErrStrFailToDecodeTokenReview)
	// ErrFailToEncodeTokenReviewResponse represents the error for failing to encode the TokenReview response to web
	ErrFailToEncodeTokenReviewResponse = errors.New(ErrStrFailToEncodeTokenReviewResponse)
	// ErrJWTValidationFailed represents the error for failing to validate access token through JWT
	ErrJWTValidationFailed = errors.New(ErrStrJWTValidationFailed)
	// ErrRequestMethodNotAllowed represents the error for the request method not being allowed
	ErrRequestMethodNotAllowed = errors.New(ErrStrRequestMethodNotAllowed)
	// ErrInvalidHttpAndHttpsPort represents the error for http and https port cannot be invalid at the same time
	ErrInvalidHttpAndHttpsPort = errors.New(ErrStrInvalidHttpAndHttpsPort)
	// ErrInvalidHttpPort defines the error message that http port is invalid
	ErrInvalidHttpPort = errors.New(ErrStrInvalidHttpPort)
	// ErrInvalidHttpsPort defines the error message that https port is invalid
	ErrInvalidHttpsPort = errors.New(ErrStrInvalidHttpsPort)
	// ErrEmptyCertFile represents the error for the tls cert file being empty to serve https requests
	ErrEmptyCertFile = errors.New(ErrStrEmptyCertFile)
	// ErrEmptyPrivateKeyFile represents the error for the tls private key file being empty to serve https requests
	ErrEmptyPrivateKeyFile = errors.New(ErrStrEmptyPrivateKeyFile)
	// ErrEmptyMasterCAFile represents the error for the master CA file being empty to serve https requests
	ErrEmptyMasterCAFile = errors.New(ErrStrEmptyMasterCAFile)
	// ErrFailToLoadCert represents the error for the cert and private key does not match
	ErrFailToLoadCert = errors.New(ErrStrFailToLoadCert)
	// ErrOAuthWebhookConfigFileMissing represents the error for the configfile being missing to start the httpserver
	ErrOAuthWebhookConfigFileMissing = errors.New(ErrStrOAuthWebhookConfigFileMissing)
	// ErrFileNotExist defines the error message for the target file missing
	ErrFileNotExist = errors.New(ErrStrFileNotExist)
)
