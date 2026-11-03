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

	"github.com/agiledragon/gomonkey/v2"
)

func TestNewServerConfig(t *testing.T) {
	tests := []struct {
		name string
		want *ServerConfig
	}{
		{
			name: " nice",
			want: &ServerConfig{
				BindAddress:    "0.0.0.0",
				InsecurePort:   9022,
				SecurePort:     0,
				CertFile:       "",
				PrivateKeyFile: ""},
		},
		{
			name: "mock",
			want: &ServerConfig{
				BindAddress:    "0.0.0.0",
				InsecurePort:   0,
				SecurePort:     9022,
				CertFile:       "",
				PrivateKeyFile: ""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "mock" {
				patch := gomonkey.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
					return nil, nil
				})
				defer patch.Reset()
			}
			if got := NewServerConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewServerConfig() = %v, want %v", got, tt.want)
			}
		})
	}
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
				SecurePort:   9022,
				CertFile:     "",
				PrivateKey:   "",
			},
			want: []error{fmt.Errorf("tls private key file is empty while secure serving"),
				fmt.Errorf("tls private key file is empty while secure serving")},
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
				BindAddress:    tt.fields.BindAddress,
				SecurePort:     tt.fields.SecurePort,
				InsecurePort:   tt.fields.InsecurePort,
				PrivateKeyFile: tt.fields.PrivateKey,
				CertFile:       tt.fields.CertFile,
				CAFile:         tt.fields.CAFile,
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
