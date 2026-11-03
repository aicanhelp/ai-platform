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
package utils

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"
)

func TestCreateHostClient(t *testing.T) {
	tests := []struct {
		name        string
		want        *kubernetes.Clientset
		want1       *api.Config
		wantErr     bool
		mockData    *rest.Config
		mockCluster *kubernetes.Clientset
		mockErr     error
		mockK8Err   error
	}{
		{
			name:        "fail",
			want:        nil,
			want1:       nil,
			wantErr:     true,
			mockData:    &rest.Config{},
			mockCluster: &kubernetes.Clientset{DiscoveryClient: &discovery.DiscoveryClient{}},
			mockErr:     fmt.Errorf("error"),
		},
		{
			name: "success",
			want: &kubernetes.Clientset{},
			want1: &api.Config{Clusters: map[string]*api.Cluster{
				"kubernetes": {
					"", "", "", false,
					"", []byte{}, "", false,
					make(map[string]runtime.Object),
				},
			}},
			wantErr:     false,
			mockData:    &rest.Config{},
			mockCluster: &kubernetes.Clientset{},
			mockErr:     nil,
		},
		{
			name:        "Error creating cluster client",
			want:        nil,
			want1:       nil,
			wantErr:     true,
			mockData:    &rest.Config{},
			mockCluster: &kubernetes.Clientset{DiscoveryClient: &discovery.DiscoveryClient{}},
			mockK8Err:   fmt.Errorf("error"),
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
			batch := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(c *rest.Config) (*kubernetes.Clientset, error) {
				if tt.mockK8Err != nil {
					return nil, tt.mockK8Err
				}
				return tt.mockCluster, nil
			})
			defer patch.Reset()
			defer batch.Reset()
			got, got1, err := CreateHostClient()
			if (err != nil) != tt.wantErr {
				t.Errorf("CreateHostClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CreateHostClient() got = %v, want %v", got, tt.want)
			}
			if tt.name == "success" {
				fmt.Printf("got %+v, tt %+v", *got1.Clusters["kubernetes"], *tt.want1.Clusters["kubernetes"])
			}
		})
	}
}

func TestRestConfig2ApiConfig(t *testing.T) {
	type args struct {
		restCfg *rest.Config
		apiCfg  *api.Config
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "suceess",
			args: args{
				restCfg: &rest.Config{},
				apiCfg: &api.Config{Preferences: *api.NewPreferences(),
					Clusters:   make(map[string]*api.Cluster),
					AuthInfos:  make(map[string]*api.AuthInfo),
					Contexts:   make(map[string]*api.Context),
					Extensions: make(map[string]runtime.Object)},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := restConfig2ApiConfig(tt.args.restCfg, tt.args.apiCfg); (err != nil) != tt.wantErr {
				t.Errorf("restConfig2ApiConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateClusterClient(t *testing.T) {
	type args struct {
		config *api.Config
	}
	tests := []struct {
		name           string
		args           args
		want           *kubernetes.Clientset
		wantErr        bool
		mockrestConfig *restclient.Config
		mockErr        error
		mockCluster    *kubernetes.Clientset
		mockClusterErr error
	}{
		{
			name: "suceess",
			args: args{
				config: &api.Config{Kind: "config",
					APIVersion:  "v1",
					Preferences: *api.NewPreferences(),
					Clusters:    map[string]*api.Cluster{},
					AuthInfos:   map[string]*api.AuthInfo{},
					Contexts:    map[string]*api.Context{},
					Extensions:  map[string]runtime.Object{},
				},
			},
			want:           &kubernetes.Clientset{},
			wantErr:        false,
			mockrestConfig: &restclient.Config{},
			mockErr:        nil,
			mockCluster:    &kubernetes.Clientset{},
		},
		{
			name: "Error creating cluster config",
			args: args{config: &api.Config{Kind: "config",
				APIVersion:  "v1",
				Preferences: *api.NewPreferences(),
				Contexts:    map[string]*api.Context{},
				Extensions:  map[string]runtime.Object{},
				Clusters:    map[string]*api.Cluster{},
				AuthInfos:   map[string]*api.AuthInfo{},
			}},
			want:           nil,
			mockrestConfig: &restclient.Config{},
			wantErr:        true,
			mockErr:        fmt.Errorf("error"),
			mockCluster:    &kubernetes.Clientset{},
		},
		{
			name: "Error creating cluster client",
			args: args{config: &api.Config{Kind: "config",
				APIVersion:  "v1",
				Preferences: *api.NewPreferences(),
				Clusters:    map[string]*api.Cluster{},
				AuthInfos:   map[string]*api.AuthInfo{},
				Contexts:    map[string]*api.Context{},
				Extensions:  map[string]runtime.Object{},
			}},
			want:           nil,
			wantErr:        true,
			mockrestConfig: &restclient.Config{},
			mockClusterErr: fmt.Errorf("error"),
			mockCluster:    &kubernetes.Clientset{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyMethod(reflect.TypeOf(&clientcmd.DirectClientConfig{}),
				"ClientConfig", func(_ *clientcmd.DirectClientConfig) (*rest.Config, error) {
					if tt.mockErr != nil {
						return nil, tt.mockErr
					}
					return tt.mockrestConfig, nil
				})
			batch := gomonkey.ApplyFunc(kubernetes.NewForConfig,
				func(config *rest.Config) (*kubernetes.Clientset, error) {
					if tt.mockClusterErr != nil {
						return nil, tt.mockClusterErr
					}
					return tt.mockCluster, nil
				})
			defer patch.Reset()
			defer batch.Reset()
			got, err := CreateClusterClient(tt.args.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("CreateClusterClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CreateClusterClient() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLoadKubeconfigRawData(t *testing.T) {
	type args struct {
		kubeconfigData json.RawMessage
		apiKubeconfig  *api.Config
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "success",
			args: args{
				kubeconfigData: json.RawMessage(`
			{
				"apiVersion": "v1",
				"kind": "Config",
				"clusters": [
					{
						"name": "cluster1",
						"cluster": {
						}
					}
				],
				"contexts": [
					{
						"name": "context1",
						"context": {
						}
					}
				],
				"current-context": "context1",
				"preferences": {},
				"users": [
					{
						"name": "user1",
						"user": {
						}
					}
				]
			}`),
				apiKubeconfig: &api.Config{},
			},
			wantErr: false,
		},
		{
			name: "fail empty json",
			args: args{
				kubeconfigData: make(json.RawMessage, 0),
				apiKubeconfig:  &api.Config{},
			},
			wantErr: true,
		},
		{
			name: "fail invalid json",
			args: args{
				kubeconfigData: json.RawMessage("invalid-json"),
				apiKubeconfig:  &api.Config{},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := LoadKubeconfigRawData(tt.args.kubeconfigData, tt.args.apiKubeconfig); (err != nil) != tt.wantErr {
				t.Errorf("LoadKubeconfigRawData() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
