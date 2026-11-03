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
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"

	"console-service/pkg/constant"
)

func TestNewServerConfigInsecure(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer patch.Reset()
	got := NewServerConfig()
	want := &ServerConfig{
		BindAddress:  "",
		InsecurePort: 9037,
		SecurePort:   0,
		CertFile:     "",
		PrivateKey:   "",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("NewServerConfig() = %v, want %v", got, want)
	}
}

func TestNewServerConfigSecurePermissionError(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrPermission
	})
	defer patch.Reset()
	got := NewServerConfig()
	if got != nil {
		t.Errorf("NewServerConfig() = %v, want %v", got, nil)
	}
}

func TestNewServerConfigSecureSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, nil
	})
	defer patch.Reset()
	got := NewServerConfig()
	want := &ServerConfig{
		BindAddress:  "",
		InsecurePort: 0,
		SecurePort:   9037,
		CertFile:     constant.TLSCertPath,
		PrivateKey:   constant.TLSKeyPath,
		CAFile:       constant.CAPath,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("NewServerConfig() = %v, want %v", got, want)
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
		wantErrors   bool
	}{
		{
			name:         "valid_config_with_insecure_port",
			BindAddress:  "",
			InsecurePort: 9032,
			SecurePort:   0,
			CertFile:     "",
			PrivateKey:   "",
			wantErrors:   false,
		},
		{
			name:         "invalid_both_ports_disabled",
			BindAddress:  "",
			InsecurePort: 0,
			SecurePort:   0,
			CertFile:     "",
			PrivateKey:   "",
			wantErrors:   true,
		},
		{
			name:         "invalid_secure_port_with_empty_filenames",
			BindAddress:  "",
			InsecurePort: 0,
			SecurePort:   8443,
			CertFile:     "",
			PrivateKey:   "",
			wantErrors:   true,
		},
		{
			name:         "invalid_secure_port_with_nonexistent_private_key",
			BindAddress:  "",
			InsecurePort: 0,
			SecurePort:   8443,
			CertFile:     "/nonexistent/cert.pem",
			PrivateKey:   "/nonexistent/key.pem",
			wantErrors:   true,
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
			got := s.Validate()

			if tt.wantErrors {
				if len(got) == 0 {
					t.Errorf("Validate() returned no errors, want errors")
				}
			} else {
				if len(got) != 0 {
					t.Errorf("Validate() = %v, want no errors", got)
				}
			}
		})
	}
}
