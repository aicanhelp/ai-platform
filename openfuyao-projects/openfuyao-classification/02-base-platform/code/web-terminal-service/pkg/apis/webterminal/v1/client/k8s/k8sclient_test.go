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
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func TestNewKubernetesClient(t *testing.T) {
	type args struct {
		cfg *KubernetesCfg
	}
	tests := []struct {
		name     string
		args     args
		want     Client
		wantErr  bool
		mockData *kubernetes.Clientset
		mockErr  error
	}{
		{
			name: "ok",
			args: args{
				cfg: &KubernetesCfg{
					KubeConfig: &rest.Config{},
				},
			},
			want: &kubernetesClient{
				k8s:    &kubernetes.Clientset{},
				config: &rest.Config{},
			},
			wantErr:  false,
			mockData: &kubernetes.Clientset{},
		},
		{
			name: "error",
			args: args{
				cfg: &KubernetesCfg{
					KubeConfig: &rest.Config{},
				},
			},
			want:     nil,
			wantErr:  true,
			mockData: &kubernetes.Clientset{},
			mockErr:  errors.New("yes"),
		},
		{
			name: "nil",
			args: args{
				cfg: &KubernetesCfg{},
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(c *rest.Config) (*kubernetes.Clientset, error) {
				if tt.mockErr != nil {
					return nil, tt.mockErr
				}
				return tt.mockData, nil
			})
			defer patch.Reset()
			got, err := NewKubernetesClient(tt.args.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewKubernetesClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewKubernetesClient() got = %v, want %v", got.Cfg(), tt.want.Cfg())
			}
		})
	}
}
