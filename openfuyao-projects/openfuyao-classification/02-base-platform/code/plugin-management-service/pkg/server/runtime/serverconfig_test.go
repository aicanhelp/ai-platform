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
	"fmt"
	"os"
	"reflect"
	"testing"
)

func TestNewServerConfig(t *testing.T) {
	oldServicePort := os.Getenv("SERVICE_PORT")
	oldEnableTLS := os.Getenv("ENABLE_TLS")
	defer func() {
		_ = os.Setenv("SERVICE_PORT", oldServicePort)
		_ = os.Setenv("ENABLE_TLS", oldEnableTLS)
	}()

	t.Run("invalid SERVICE_PORT falls back to default insecure port", func(t *testing.T) {
		_ = os.Setenv("SERVICE_PORT", "not-a-number")
		_ = os.Setenv("ENABLE_TLS", "false")

		cfg := NewServerConfig()
		if cfg == nil {
			t.Fatalf("expected config, got nil")
		}
		if cfg.InsecurePort != 9040 {
			t.Fatalf("expected default insecure port 9040, got %d", cfg.InsecurePort)
		}
		if cfg.SecurePort != 0 {
			t.Fatalf("expected secure port 0, got %d", cfg.SecurePort)
		}
	})

	t.Run("ENABLE_TLS not true uses insecure port from SERVICE_PORT", func(t *testing.T) {
		_ = os.Setenv("SERVICE_PORT", "18080")
		_ = os.Setenv("ENABLE_TLS", "false")

		cfg := NewServerConfig()
		if cfg == nil {
			t.Fatalf("expected config, got nil")
		}
		if cfg.InsecurePort != 18080 {
			t.Fatalf("expected insecure port 18080, got %d", cfg.InsecurePort)
		}
		if cfg.SecurePort != 0 {
			t.Fatalf("expected secure port 0, got %d", cfg.SecurePort)
		}
	})

	// Note: TLS enabled + cert exists branches are environment-dependent because cert path is a const.
	// We at least cover the TLS enabled but cert missing branch (falls back to insecure).
	t.Run("ENABLE_TLS true but cert missing falls back to insecure", func(t *testing.T) {
		_ = os.Setenv("SERVICE_PORT", "19090")
		_ = os.Setenv("ENABLE_TLS", "true")

		cfg := NewServerConfig()
		if cfg == nil {
			t.Fatalf("expected config, got nil")
		}
		if cfg.InsecurePort != 19090 {
			t.Fatalf("expected insecure port 19090, got %d", cfg.InsecurePort)
		}
		if cfg.SecurePort != 0 {
			t.Fatalf("expected secure port 0, got %d", cfg.SecurePort)
		}
	})
}

func TestServerConfigValidate(t *testing.T) {
	_, errCert := os.Stat("/ssl/server.crt")
	if errCert == nil {
		t.Fatal("cert /ssl/server.crt should not exists")
	}
	_, errKey := os.Stat("/ssl/server.key")
	if errKey == nil {
		t.Fatal("cert key /ssl/key.crt should not exists")
	}

	tests := []struct {
		name         string
		BindAddress  string
		InsecurePort int
		SecurePort   int
		CertFile     string
		PrivateKey   string
		want         []error
	}{
		{
			name:         "TestValidateZeroPorts",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   0,
			CertFile:     "",
			PrivateKey:   "",
			want: []error{
				fmt.Errorf("insecure and secure port can not be disabled at the same time"),
			},
		},
		{
			name:         "TestValidateInsecure",
			BindAddress:  "0.0.0.0",
			InsecurePort: 9032,
			SecurePort:   0,
			CertFile:     "",
			PrivateKey:   "",
			want:         []error{},
		},
		{
			name:         "TestValidateSecureEmptyCertNoKey",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   4443,
			CertFile:     "",
			PrivateKey:   "/ssl/server.key",
			want: []error{
				fmt.Errorf("tls certificate file is empty while secure serving"),
				errKey,
			},
		},
		{
			name:         "TestValidateSecureNoCertEmptyKey",
			BindAddress:  "0.0.0.0",
			InsecurePort: 0,
			SecurePort:   4443,
			CertFile:     "/ssl/server.crt",
			PrivateKey:   "",
			want: []error{
				errCert,
				fmt.Errorf("tls private key file is empty while secure serving"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerConfig{
				BindAddress:    tt.BindAddress,
				InsecurePort:   tt.InsecurePort,
				SecurePort:     tt.SecurePort,
				CertFile:       tt.CertFile,
				PrivateKeyFile: tt.PrivateKey,
			}
			if got := s.Validate(); len(got) != 0 {
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("Validate() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}
