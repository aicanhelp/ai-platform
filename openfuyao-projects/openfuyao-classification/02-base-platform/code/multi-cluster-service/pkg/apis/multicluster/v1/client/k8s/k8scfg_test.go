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

package k8s

import (
	"errors"
	"io/fs"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/homedir"
)

func setupHomeEnv(t *testing.T, homePath string) {
	originalHome := os.Getenv("HOME")
	t.Setenv("HOME", homePath)

	t.Cleanup(func() {
		if originalHome != "" {
			err := os.Setenv("HOME", originalHome)
			if err != nil {
				return
			}
		}
	})
}

func TestGetKubeConfigFile(t *testing.T) {

	tests := []struct {
		name    string
		want    string
		mockErr error
	}{
		{
			name: "get",
			want: "~/.kube/config",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setupHomeEnv(t, "~")
			patches := gomonkey.NewPatches()
			patches.ApplyFunc(homedir.HomeDir, func() string {
				return "~"
			})
			patches.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
				return nil, nil
			})
			defer patches.Reset()
			if got := getKubeConfigFile(); got != tt.want {
				t.Errorf("getKubeConfigFile() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetKubeConfig(t *testing.T) {
	tests := []struct {
		name     string
		want     *rest.Config
		mockData *rest.Config
		mockErr  error
	}{
		{
			name: "success",
			mockData: &rest.Config{
				Host:        "ok",
				BearerToken: "test",
			},
			want: &rest.Config{
				Host:        "ok",
				BearerToken: "test",
			},
			mockErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyFunc(rest.InClusterConfig, func() (*rest.Config, error) {
				if tt.mockErr != nil {
					return nil, tt.mockErr
				}
				return tt.mockData, nil
			})
			defer patch.Reset()
			if got := GetKubeConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetKubeConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestKubernetesCfgValidate(t *testing.T) {
	type fields struct {
		KubeConfigFile string
		KubeConfig     *rest.Config
		QPS            float32
		Burst          int
	}
	tests := []struct {
		name   string
		fields fields
		want   []error
	}{
		{
			name: "ok",
			fields: fields{
				KubeConfigFile: "test",
				KubeConfig:     nil,
				QPS:            1,
				Burst:          12,
			},
			want: []error{errors.New("new"), errors.New("k8s config get nil")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyFunc(os.Stat, func(name string) (fs.FileInfo, error) {
				return nil, errors.New("new")
			})
			defer patch.Reset()
			k := &KubernetesCfg{
				KubeConfigFile: tt.fields.KubeConfigFile,
				KubeConfig:     tt.fields.KubeConfig,
				QPS:            tt.fields.QPS,
				Burst:          tt.fields.Burst,
			}
			if got := k.Validate(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewKubernetesCfg(t *testing.T) {
	tests := []struct {
		name string
		want *KubernetesCfg
		mock *KubernetesCfg
	}{
		{
			name: "mock",
			want: &KubernetesCfg{
				KubeConfigFile: "",
				KubeConfig:     nil,
				QPS:            1e6,
				Burst:          1e6,
			},
			mock: &KubernetesCfg{
				KubeConfigFile: "",
				KubeConfig:     nil,
				QPS:            1e6,
				Burst:          1e6,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			patches.ApplyFunc(NewKubernetesCfg, func() *KubernetesCfg {
				return tt.mock
			})
			defer patches.Reset()
			if got := NewKubernetesCfg(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewKubernetesCfg() = %v, want %v", got, tt.want)
			}
		})
	}
}
