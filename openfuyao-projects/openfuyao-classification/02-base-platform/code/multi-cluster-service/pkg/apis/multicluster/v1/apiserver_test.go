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
package v1

import (
	"errors"
	"net/http"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/config"
	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/runtime"
)

func getTestCases() []struct {
	name string
	args struct {
		cfg *config.RunConfig
	}
	want    *http.Server
	wantErr bool
	mockErr error
} {
	return []struct {
		name string
		args struct {
			cfg *config.RunConfig
		}
		want    *http.Server
		wantErr bool
		mockErr error
	}{
		{
			name: "success",
			args: struct {
				cfg *config.RunConfig
			}{
				cfg: &config.RunConfig{
					Server: &runtime.ServerConfig{SecurePort: 0},
				},
			},
			want: &http.Server{
				Addr: ":0",
			},
			wantErr: false,
		},
		{
			name: "error loading",
			args: struct {
				cfg *config.RunConfig
			}{
				cfg: &config.RunConfig{
					Server: &runtime.ServerConfig{SecurePort: 2},
				},
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "mock error",
			args: struct {
				cfg *config.RunConfig
			}{
				cfg: &config.RunConfig{
					Server: &runtime.ServerConfig{SecurePort: 1},
				},
			},
			want:    nil,
			wantErr: true,
			mockErr: errors.New("err"),
		},
	}
}

func TestInitServer(t *testing.T) {
	testCases := getTestCases()
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "mock error" {
				patch := gomonkey.ApplyFunc(os.ReadFile, func(path string) ([]byte, error) {
					if tt.mockErr != nil {
						return nil, tt.mockErr
					}
					return []byte{}, nil
				})
				defer patch.Reset()
			}
			got, err := initServer(tt.args.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("initServer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("initServer() got = %v, want %v", got, *tt.want)
			}
		})
	}
}
