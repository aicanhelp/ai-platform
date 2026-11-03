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
	"fmt"
	"io/fs"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

func TestNewServerConfig(t *testing.T) {
	tests := []struct {
		name string
		want *ServerConfig
	}{
		{
			name: "mock",
			want: &ServerConfig{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   9072,
				CertFile:     "/ssl/server.crt",
				CAFile:       "/ssl/ca.pem",
				PrivateKey:   "/ssl/server.key"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mock FileInfo
			mockFileInfo := &mockFileInfo{
				name: "test.log",
				size: 1024,
				mode: 0644,
			}

			patch := gomonkey.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
				return mockFileInfo, nil
			})
			defer patch.Reset()

			if got := NewServerConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewServerConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewServerConfig_StatError(t *testing.T) {
	t.Setenv("SERVICE_PORT", "9999")
	patch := gomonkey.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
		return nil, errors.New("stat error")
	})
	defer patch.Reset()

	if got := NewServerConfig(); got != nil {
		t.Errorf("NewServerConfig() = %v, want nil", got)
	}
}

func (m *mockFileInfo) ModTime() time.Time { return time.Now() }
func (m *mockFileInfo) IsDir() bool        { return false }
func (m *mockFileInfo) Sys() interface{}   { return nil }
func (m *mockFileInfo) Name() string       { return m.name }
func (m *mockFileInfo) Size() int64        { return m.size }
func (m *mockFileInfo) Mode() os.FileMode  { return m.mode }

// 实现一个简单的 FileInfo mock
type mockFileInfo struct {
	name string
	size int64
	mode os.FileMode
}

func TestServerConfigValidate(t *testing.T) {
	type fields struct {
		BindAddress  string
		SecurePort   int
		InsecurePort int
		PrivateKey   string
		CertFile     string
		CAFile       string
	}
	tests := []struct {
		name    string
		fields  fields
		want    []error
		mockErr error
	}{
		{
			name: "ok",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   9072,
				CertFile:     "",
				PrivateKey:   "",
			},
			want: []error{fmt.Errorf("tls private key file is empty while secure serving"),
				fmt.Errorf("tls private key file is empty while secure serving")},
		},
		{
			name: "ports disabled",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 0,
				SecurePort:   0,
			},
			want: []error{fmt.Errorf("insecure and secure port can not be disabled at the same time")},
		},
		{
			name: "error",
			fields: fields{
				BindAddress:  "0.0.0.0",
				InsecurePort: 100,
				SecurePort:   100,
				CertFile:     "gh",
				PrivateKey:   "ggd",
			},
			want: []error{errors.New("g"), errors.New("g")},
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
				CAFile:       tt.fields.CAFile,
			}
			patch := gomonkey.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
				return nil, errors.New("g")
			})
			defer patch.Reset()
			if got := s.Validate(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestServerConfigValidate_SecureFilesExist(t *testing.T) {
	s := &ServerConfig{
		BindAddress:  "0.0.0.0",
		SecurePort:   9443,
		InsecurePort: 0,
		CertFile:     "cert.pem",
		PrivateKey:   "key.pem",
	}
	patch := gomonkey.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
		return &mockFileInfo{name: name, size: 1, mode: 0644}, nil
	})
	defer patch.Reset()

	errs := s.Validate()
	assert.Len(t, errs, 0)
}
