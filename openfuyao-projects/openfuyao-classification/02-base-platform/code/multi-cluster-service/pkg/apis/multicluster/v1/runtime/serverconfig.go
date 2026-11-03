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

// Package runtime defines ServerConfig and validates it.
package runtime

import (
	"fmt"
	"os"
	"strconv"

	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

const (
	defaultServicePort = 9022
	maxSecurePort      = 65535
	tlsCAPath          = ""
	tlsCertPath        = ""
	tlsKeyPath         = ""
)

// ServerConfig 定义一个 http.server 结构
type ServerConfig struct {
	// server bind address
	BindAddress string

	// secure port number
	SecurePort int

	// insecure port number
	InsecurePort int

	// tls private key file
	PrivateKeyFile string

	// tls cert file
	CertFile string

	// tls CA file
	CAFile string
}

// NewServerConfig create new server config
func NewServerConfig() *ServerConfig {
	port, err := strconv.Atoi(os.Getenv("SERVICE_PORT"))
	if err != nil {
		port = defaultServicePort
	}
	s := ServerConfig{
		BindAddress:    "0.0.0.0",
		InsecurePort:   0,
		SecurePort:     0,
		CertFile:       "",
		PrivateKeyFile: "",
	}
	if _, err := os.Stat(tlsCertPath); os.IsNotExist(err) {
		s.InsecurePort = port
		return &s
	} else if err != nil {
		zlog.LogErrorf("Error accessing file: %v", err)
		return nil
	}
	s.SecurePort = port
	s.CertFile = tlsCertPath
	s.PrivateKeyFile = tlsKeyPath
	s.CAFile = tlsCAPath
	return &s
}

// Validate server 校验
func (s *ServerConfig) Validate() []error {
	var errs []error

	if s.SecurePort == 0 && s.InsecurePort == 0 {
		err := fmt.Errorf("insecure and secure port can not be disabled at the same time")
		errs = append(errs, err)
	}

	if s.SecurePort > 0 && s.SecurePort < maxSecurePort {
		if s.CertFile == "" {
			err := fmt.Errorf("tls private key file is empty while secure serving")
			errs = append(errs, err)
		} else {
			if _, err := os.Stat(s.CertFile); err != nil {
				errs = append(errs, err)
			}
		}

		if s.PrivateKeyFile == "" {
			err := fmt.Errorf("tls private key file is empty while secure serving")
			errs = append(errs, err)
		} else {
			if _, err := os.Stat(s.PrivateKeyFile); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return errs
}
