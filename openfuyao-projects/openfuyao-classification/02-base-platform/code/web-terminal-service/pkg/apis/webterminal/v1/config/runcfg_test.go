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
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/client/k8s"
	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/runtime"
)

func TestNewRunConfig(t *testing.T) {
	tests := []struct {
		name       string
		want       *RunConfig
		mockServer *runtime.ServerConfig
		mockCfg    *k8s.KubernetesCfg
	}{
		{
			name: "ok",
			want: &RunConfig{
				Server:        &runtime.ServerConfig{},
				KubernetesCfg: &k8s.KubernetesCfg{},
			},
			mockServer: &runtime.ServerConfig{},
			mockCfg:    &k8s.KubernetesCfg{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch1 := gomonkey.ApplyFunc(runtime.NewServerConfig, func() *runtime.ServerConfig {
				return &runtime.ServerConfig{}
			})
			patch2 := gomonkey.ApplyFunc(k8s.NewKubernetesCfg, func() *k8s.KubernetesCfg {
				return &k8s.KubernetesCfg{}
			})
			defer patch1.Reset()
			defer patch2.Reset()
			if got := NewRunConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewRunConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRunConfigValidate(t *testing.T) {
	type fields struct {
		Server        *runtime.ServerConfig
		KubernetesCfg *k8s.KubernetesCfg
	}
	tests := []struct {
		name   string
		fields fields
		want   []error
	}{
		{
			name: "ok",
			fields: fields{
				Server:        &runtime.ServerConfig{},
				KubernetesCfg: &k8s.KubernetesCfg{},
			},
			want: []error{fmt.Errorf("insecure and secure port can not be disabled at the same time"),
				errors.New("k8s config get nil")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RunConfig{
				Server:        tt.fields.Server,
				KubernetesCfg: tt.fields.KubernetesCfg,
			}
			if got := cfg.Validate(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}
