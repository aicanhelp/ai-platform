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
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"application-management-service/pkg/constant"
)

func TestNewServerConfig(t *testing.T) {
	origServicePort := os.Getenv("SERVICE_PORT")
	origEnableTLS := os.Getenv("ENABLE_TLS")
	defer func() {
		_ = os.Setenv("SERVICE_PORT", origServicePort)
		_ = os.Setenv("ENABLE_TLS", origEnableTLS)
	}()

	t.Run("SERVICE_PORT invalid -> default port", func(t *testing.T) {
		_ = os.Setenv("SERVICE_PORT", "not-an-int")
		_ = os.Setenv("ENABLE_TLS", "false")

		got := NewServerConfig()
		if got == nil {
			t.Fatalf("expected non-nil config")
		}
		if got.InsecurePort != defaultServicePort {
			t.Fatalf("expected InsecurePort=%d, got %d", defaultServicePort, got.InsecurePort)
		}
		if got.SecurePort != 0 {
			t.Fatalf("expected SecurePort=0, got %d", got.SecurePort)
		}
	})

	t.Run("TLS disabled -> insecure port set", func(t *testing.T) {
		_ = os.Setenv("SERVICE_PORT", "12345")
		_ = os.Setenv("ENABLE_TLS", "false")

		got := NewServerConfig()
		if got == nil {
			t.Fatalf("expected non-nil config")
		}
		if got.InsecurePort != 12345 {
			t.Fatalf("expected InsecurePort=12345, got %d", got.InsecurePort)
		}
		if got.SecurePort != 0 {
			t.Fatalf("expected SecurePort=0, got %d", got.SecurePort)
		}
	})

	t.Run("TLS enabled but cert missing -> fallback to insecure", func(t *testing.T) {
		_ = os.Setenv("SERVICE_PORT", "23456")
		_ = os.Setenv("ENABLE_TLS", "true")

		// Ensure TLSCertPath does not exist
		_ = os.Remove(constant.TLSCertPath)

		got := NewServerConfig()
		if got == nil {
			t.Fatalf("expected non-nil config")
		}
		if got.InsecurePort != 23456 {
			t.Fatalf("expected InsecurePort=23456, got %d", got.InsecurePort)
		}
		if got.SecurePort != 0 {
			t.Fatalf("expected SecurePort=0, got %d", got.SecurePort)
		}
	})

}

func TestServerConfigValidate(t *testing.T) {
	tmp := t.TempDir()
	existingCert := filepath.Join(tmp, "cert.pem")
	existingKey := filepath.Join(tmp, "key.pem")
	if err := os.WriteFile(existingCert, []byte("cert"), 0o644); err != nil {
		t.Fatalf("write existingCert: %v", err)
	}
	if err := os.WriteFile(existingKey, []byte("key"), 0o644); err != nil {
		t.Fatalf("write existingKey: %v", err)
	}

	tests := []struct {
		name       string
		cfg        *ServerConfig
		wantErrNum int
		wantSubstr []string
		wantNotExist bool
	}{
		{
			name: "both ports disabled -> error",
			cfg: &ServerConfig{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   0,
			},
			wantErrNum: 1,
			wantSubstr: []string{"can not be disabled"},
		},
		{
			name: "secure port enabled, cert empty, key empty -> 2 errors",
			cfg: &ServerConfig{
				SecurePort: 1,
				CertFile:   "",
				PrivateKey: "",
			},
			wantErrNum: 2,
			wantSubstr: []string{"file is empty"},
		},
		{
			name: "secure port enabled, cert missing, key exists -> 1 stat error",
			cfg: &ServerConfig{
				SecurePort: 1,
				CertFile:   filepath.Join(tmp, "missing-cert.pem"),
				PrivateKey: existingKey,
			},
			wantErrNum: 1,
			wantNotExist: true,
		},
		{
			name: "secure port enabled, cert exists, key missing -> 1 stat error",
			cfg: &ServerConfig{
				SecurePort: 1,
				CertFile:   existingCert,
				PrivateKey: filepath.Join(tmp, "missing-key.pem"),
			},
			wantErrNum: 1,
			wantNotExist: true,
		},
		{
			name: "secure port enabled, cert exists, key exists -> ok",
			cfg: &ServerConfig{
				SecurePort: 1,
				CertFile:   existingCert,
				PrivateKey: existingKey,
			},
			wantErrNum: 0,
		},
		{
			name: "secure port too large -> secure validation skipped",
			cfg: &ServerConfig{
				SecurePort:   maxSecurePort,
				InsecurePort: 123,
				CertFile:     "",
				PrivateKey:   "",
			},
			wantErrNum: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.cfg.Validate()
			if len(errs) != tt.wantErrNum {
				t.Fatalf("expected %d errors, got %d: %v", tt.wantErrNum, len(errs), errs)
			}
			if len(tt.wantSubstr) > 0 {
				joined := strings.Join(func() []string {
					out := make([]string, 0, len(errs))
					for _, e := range errs {
						if e == nil {
							out = append(out, "<nil>")
							continue
						}
						out = append(out, e.Error())
					}
					return out
				}(), "\n")

				for _, sub := range tt.wantSubstr {
					if !strings.Contains(joined, sub) {
						t.Fatalf("expected error to contain %q, got: %s", sub, joined)
					}
				}
			}

			if tt.wantNotExist {
				if len(errs) == 0 {
					t.Fatalf("expected not-exist error, got empty errs")
				}
				for _, e := range errs {
					if e == nil {
						t.Fatalf("expected not-exist error, got nil error")
					}
					if !os.IsNotExist(e) {
						t.Fatalf("expected os.IsNotExist(err)=true, got: %v", e)
					}
				}
			}
		})
	}

	for _, e := range (&ServerConfig{InsecurePort: 1}).Validate() {
		if errors.Is(e, nil) {
			t.Fatalf("unexpected nil-like error")
		}
	}
}
