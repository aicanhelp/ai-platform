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

// Package runtime provides the foundational structures and utilities for configuring
// and managing HTTP servers within an application. It is designed to facilitate the setup
// and validation of server configurations, ensuring robust and secure communication channels
// for web services.
package runtime

import (
	"os"
	"strconv"

	"openfuyao.com/monitoring-service/pkg/zlog"
)

const (
	defaultServicePort = 9083
	tlsCAPath          = "/ssl/ca.pem"
	tlsCertPath        = "/ssl/server.crt"
	tlsKeyPath         = "/ssl/server.key"
)

// Server 定义一个 http.server 结构
type Server struct {
	// server bind address
	BindAddress string

	// insecure port number
	InsecurePort int

	// secure port number
	SecurePort int

	CertFile string `json:"CertFile"`

	PrivateKeyFile string `json:"PrivateKeyFile"`

	RootCAFile string `json:"RootCAFile"`
}

// NewServer initializes and returns a new instance of Server with default configuration settings.
// It sets up a server ready to listen on all network interfaces (0.0.0.0) on port 9083.
func NewServer() *Server {
	port, err := strconv.Atoi(os.Getenv("SERVICE_PORT"))
	if err != nil {
		zlog.Warn("service port not provided, use default port: 9083")
		port = defaultServicePort
	}
	// create default server run options
	s := Server{
		BindAddress:  "0.0.0.0",
		InsecurePort: 0,
		SecurePort:   0,
	}
	if _, err := os.Stat(tlsCertPath); os.IsNotExist(err) {
		s.InsecurePort = port
		return &s
	} else if err != nil {
		zlog.FormatError("LogError accessing file: %v", err)
		return nil
	}
	zlog.FormatInfo("init user-management server with tls config, port: %d", port)
	s.SecurePort = port
	s.CertFile = tlsCertPath
	s.PrivateKeyFile = tlsKeyPath
	s.RootCAFile = tlsCAPath
	return &s
}
