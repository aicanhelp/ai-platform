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

package runtime

import (
	"reflect"
	"testing"
)

func TestNewServer(t *testing.T) {
	tests := []struct {
		name string
		want *ServerConfig
	}{
		{
			name: "TestNewServer",
			want: NewServerConfig(),
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

func TestServerconfigValidate(t *testing.T) {
	tests := []struct {
		name         string
		BindAddress  string
		InsecurePort int
		SecurePort   int
		CertFile     string
		PrivateKey   string
		wantErrCount int
	}{
		{
			name:         "valid insecure only",
			BindAddress:  "0.0.0.0",
			InsecurePort: 9032,
			SecurePort:   0,
			CertFile:     "",
			PrivateKey:   "",
			wantErrCount: 0,
		},
		{
			name:         "both ports disabled",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   0,
			CertFile:     "",
			PrivateKey:   "",
			wantErrCount: 1,
		},
		{
			name:         "secure port with empty cert",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   9443,
			CertFile:     "",
			PrivateKey:   "",
			wantErrCount: 2,
		},
		{
			name:         "secure port with non-existent cert and key",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   9443,
			CertFile:     "/nonexistent/cert.pem",
			PrivateKey:   "/nonexistent/key.pem",
			wantErrCount: 2,
		},
		{
			name:         "secure port with cert but empty key",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   9443,
			CertFile:     "/nonexistent/cert.pem",
			PrivateKey:   "",
			wantErrCount: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerConfig{
				BindAddress:  tt.BindAddress,
				InsecurePort: tt.InsecurePort,
				SecurePort:   tt.SecurePort,
				CertFile:     tt.CertFile,
				PrivateKey:   tt.PrivateKey,
			}
			errs := s.Validate()
			if len(errs) != tt.wantErrCount {
				t.Errorf("Validate() returned %d errors, want %d", len(errs), tt.wantErrCount)
			}
		})
	}
}
