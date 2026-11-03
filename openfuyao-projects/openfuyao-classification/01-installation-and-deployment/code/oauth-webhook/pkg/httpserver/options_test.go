/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"testing"

	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
)

func TestServerOptionsValidate(t *testing.T) {
	type fields struct {
		HttpPort          int
		HttpsPort         int
		TlsCertFile       string
		TlsPrivateKeyFile string
		MasterCAFile      string
	}
	tests := []struct {
		name   string
		fields fields
		want   []error
	}{
		{
			"miss http and https", fields{}, []error{fuyaoerrors.ErrInvalidHttpAndHttpsPort},
		},
		{
			"miss https tls files",
			fields{
				HttpsPort: 9096,
			},
			[]error{fuyaoerrors.ErrEmptyCertFile, fuyaoerrors.ErrEmptyPrivateKeyFile, fuyaoerrors.ErrEmptyMasterCAFile},
		},
		{
			"invalid https tls files",
			fields{
				HttpsPort:         9096,
				TlsCertFile:       "test",
				TlsPrivateKeyFile: "test",
				MasterCAFile:      "test",
			},
			[]error{fuyaoerrors.ErrFileNotExist, fuyaoerrors.ErrFileNotExist, fuyaoerrors.ErrFileNotExist},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ServerOptions{
				HttpPort:          tt.fields.HttpPort,
				HttpsPort:         tt.fields.HttpsPort,
				TlsCertFile:       tt.fields.TlsCertFile,
				TlsPrivateKeyFile: tt.fields.TlsPrivateKeyFile,
				MasterCAFile:      tt.fields.MasterCAFile,
			}
			if got := s.Validate(); !checkIdenticalErrors(got, tt.want) {
				t.Errorf("Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewDefaultHttpServerOptions(t *testing.T) {
	options := NewDefaultHttpServerOptions()
	if options.HttpPort != 9095 || options.HttpsPort != 0 {
		t.Fatalf("unexpected default server options: %+v", options)
	}
}

func checkIdenticalErrors(x, y []error) bool {
	if len(x) != len(y) {
		return false
	}

	for i, ex := range x {
		if ex.Error() != y[i].Error() {
			return false
		}
	}

	return true
}
