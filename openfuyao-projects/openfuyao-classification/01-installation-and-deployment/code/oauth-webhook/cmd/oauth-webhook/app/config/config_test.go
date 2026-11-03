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

package config

import (
	"reflect"
	"testing"

	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/httpserver"
)

func TestNewDefaultOAuthWebhookConfig(t *testing.T) {
	tests := []struct {
		name string
		want *OAuthWebhookConfig
	}{
		{
			"sucessfully configured",
			&OAuthWebhookConfig{
				HttpServerConfig: httpserver.NewDefaultHttpServerOptions(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewDefaultOAuthWebhookConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewDefaultOAuthWebhookConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOAuthWebhookConfigComplete(t *testing.T) {
	type fields struct {
		HttpServerConfig *httpserver.ServerOptions
		JWTPrivateKey    []byte
	}
	tests := []struct {
		name   string
		fields fields
		want   *OAuthWebhookConfig
	}{
		{
			"successfully complete",
			fields{
				HttpServerConfig: nil,
				JWTPrivateKey:    nil,
			},
			&OAuthWebhookConfig{
				HttpServerConfig: httpserver.NewDefaultHttpServerOptions(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OAuthWebhookConfig{
				HttpServerConfig: tt.fields.HttpServerConfig,
				JWTPrivateKey:    tt.fields.JWTPrivateKey,
			}
			if got := c.Complete(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Complete() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOAuthWebhookConfigValidate(t *testing.T) {
	type fields struct {
		HttpServerConfig *httpserver.ServerOptions
		JWTPrivateKey    []byte
	}
	tests := []struct {
		name   string
		fields fields
		want   []error
	}{
		{
			"successfully installed",
			fields{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort: 9095,
				},
				JWTPrivateKey: []byte("ttt"),
			},
			make([]error, 0),
		},
		{
			"invalid port",
			fields{
				HttpServerConfig: &httpserver.ServerOptions{
					HttpPort: -1,
				},
				JWTPrivateKey: []byte("ttt"),
			},
			[]error{fuyaoerrors.ErrInvalidHttpPort},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OAuthWebhookConfig{
				HttpServerConfig: tt.fields.HttpServerConfig,
				JWTPrivateKey:    tt.fields.JWTPrivateKey,
			}
			if tt.name == "successfully installed" {
				if got := c.Validate(); len(got) > 0 {
					t.Errorf("Validate() = %v, want %v", got, tt.want)
				}
			} else {
				if got := c.Validate(); !reflect.DeepEqual(got, tt.want) {
					t.Errorf("Validate() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}
