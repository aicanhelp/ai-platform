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

// Package httpserver defines the httpserver options, middlewares, and responses
package httpserver

import (
	"os"

	"openfuyao/oauth-webhook/pkg/constants"
	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
)

// ServerOptions defines the configs for httpserver
type ServerOptions struct {
	HttpPort          int    `json:"HttpPort"`
	HttpsPort         int    `json:"HttpsPort"`
	TlsCertFile       string `json:"TlsCertFile"`
	TlsPrivateKeyFile string `json:"TlsPrivateKeyFile"`
	MasterCAFile      string `json:"MasterCAFile"`
}

// NewDefaultHttpServerOptions inits the default httpserver option
func NewDefaultHttpServerOptions() *ServerOptions {
	return &ServerOptions{
		HttpPort:          9095,
		HttpsPort:         0,
		TlsCertFile:       "",
		TlsPrivateKeyFile: "",
		MasterCAFile:      "",
	}
}

// Validate assures the httpserver option is valid
func (s *ServerOptions) Validate() []error {
	var errs []error
	if s.HttpsPort == constants.MinHttpPort && s.HttpPort == constants.MinHttpPort {
		errs = append(errs, fuyaoerrors.ErrInvalidHttpAndHttpsPort)
	}
	if s.HttpPort < constants.MinHttpPort || s.HttpPort >= constants.MaxHttpPort {
		errs = append(errs, fuyaoerrors.ErrInvalidHttpPort)
	}
	if s.HttpsPort < constants.MinHttpPort || s.HttpsPort >= constants.MaxHttpPort {
		errs = append(errs, fuyaoerrors.ErrInvalidHttpsPort)
	}

	if s.HttpsPort > constants.MinHttpPort && s.HttpsPort < constants.MaxHttpPort {
		if s.TlsCertFile == "" {
			errs = append(errs, fuyaoerrors.ErrEmptyCertFile)
		} else {
			if _, err := os.Stat(s.TlsCertFile); err != nil {
				errs = append(errs, fuyaoerrors.ErrFileNotExist)
			}
		}

		if s.TlsPrivateKeyFile == "" {
			errs = append(errs, fuyaoerrors.ErrEmptyPrivateKeyFile)
		} else {
			if _, err := os.Stat(s.TlsPrivateKeyFile); err != nil {
				errs = append(errs, fuyaoerrors.ErrFileNotExist)
			}
		}

		if s.MasterCAFile == "" {
			errs = append(errs, fuyaoerrors.ErrEmptyMasterCAFile)
		} else {
			if _, err := os.Stat(s.MasterCAFile); err != nil {
				errs = append(errs, fuyaoerrors.ErrFileNotExist)
			}
		}
	}

	return errs
}
