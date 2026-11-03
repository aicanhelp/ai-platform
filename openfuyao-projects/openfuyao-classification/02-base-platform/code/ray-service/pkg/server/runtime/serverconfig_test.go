/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package runtime ServerConfig and validate
package runtime

import (
	"os"
	"reflect"
	"testing"
)

func TestNewServerConfig(t *testing.T) {
	tests := []struct {
		name string
		want *ServerConfig
	}{
		{
			name: "default server config",
			want: &ServerConfig{
				BindAddress:  "0.0.0.0",
				InsecurePort: 12345,
				SecurePort:   0,
				CertFile:     "",
				PrivateKey:   "",
				FileLocation: "/home/static",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewServerConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewServerConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestServerConfigValidate(t *testing.T) {
	certFile, keyFile, cleanup := createTempCertKeyFiles(t)
	defer cleanup()

	type fields struct {
		BindAddress  string
		SecurePort   int
		InsecurePort int
		PrivateKey   string
		CertFile     string
		FileLocation string
	}
	tests := []struct {
		name       string
		fields     fields
		wantErrNum int
	}{
		{
			name: "valid server config",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 12345,
				SecurePort:   0,
				CertFile:     "",
				PrivateKey:   "",
				FileLocation: "/home/static",
			},
			wantErrNum: 0,
		},
		{
			name: "both ports disabled",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   0,
				CertFile:     "",
				PrivateKey:   "",
				FileLocation: "/home/static",
			},
			wantErrNum: 1,
		},
		{
			name: "secure serving with empty cert and key",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   6443,
				CertFile:     "",
				PrivateKey:   "",
				FileLocation: "/home/static",
			},
			wantErrNum: 2,
		},
		{
			name: "secure serving with not exist cert and key files",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   6443,
				CertFile:     "/tmp/not-exist-cert.crt",
				PrivateKey:   "/tmp/not-exist-key.key",
				FileLocation: "/home/static",
			},
			wantErrNum: 2,
		},
		{
			name: "secure serving with exist cert and key files",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   6443,
				CertFile:     certFile,
				PrivateKey:   keyFile,
				FileLocation: "/home/static",
			},
			wantErrNum: 0,
		},
		{
			name: "max port number skips cert checks",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   65535,
				CertFile:     "",
				PrivateKey:   "",
				FileLocation: "/home/static",
			},
			wantErrNum: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerConfig{
				BindAddress:  tt.fields.BindAddress,
				SecurePort:   tt.fields.SecurePort,
				InsecurePort: tt.fields.InsecurePort,
				PrivateKey:   tt.fields.PrivateKey,
				CertFile:     tt.fields.CertFile,
				FileLocation: tt.fields.FileLocation,
			}
			if got := s.Validate(); len(got) != tt.wantErrNum {
				t.Errorf("ServerConfig.Validate() = %v, want %d errors", got, tt.wantErrNum)
			}
		})
	}
}

// createTempCertKeyFiles creates temporary placeholder cert and key files for validation tests.
func createTempCertKeyFiles(t *testing.T) (string, string, func()) {
	t.Helper()
	dir := t.TempDir()
	certFile := dir + "/test-cert.crt"
	keyFile := dir + "/test-key.key"
	if err := os.WriteFile(certFile, []byte("test-cert"), 0600); err != nil {
		t.Fatalf("failed to create temp cert file: %v", err)
	}
	if err := os.WriteFile(keyFile, []byte("test-key"), 0600); err != nil {
		t.Fatalf("failed to create temp key file: %v", err)
	}
	return certFile, keyFile, func() {}
}
