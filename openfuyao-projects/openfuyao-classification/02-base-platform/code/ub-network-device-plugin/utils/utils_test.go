/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
	"context"
	"reflect"
	"sort"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	cri "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	critesting "k8s.io/cri-api/pkg/apis/testing"
)

var (
	fakeCli = critesting.NewFakeRuntimeService()
)

func TestGetPodUIDByDeviceID(t *testing.T) {
	type args struct {
		deviceID string
		filePath string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "readfile err",
			args: args{
				deviceID: "vfe0",
				filePath: "",
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "get pod uid",
			args: args{
				deviceID: "urma_7",
				filePath: "utils_test.json",
			},
			want:    "fjdsklf9gdf-gfgr-4e543fgddg-gfdgd342gdf32",
			wantErr: false,
		},
		{
			name: "pod uid not found",
			args: args{
				deviceID: "shared0",
				filePath: "utils_test.json",
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "get pod uid V2 format",
			args: args{
				deviceID: "urma_29",
				filePath: "utils_test_v2.json",
			},
			want:    "fjdsklf9gdf-gd4r-4egdf43ddg-gfdga21342gd32",
			wantErr: false,
		},
		{
			name: "get pod uid V2 format 1",
			args: args{
				deviceID: "urma_111",
				filePath: "utils_test_v2.json",
			},
			want:    "abcdefg",
			wantErr: false,
		},
		{
			name: "get pod uid V2 format 2",
			args: args{
				deviceID: "urma_222",
				filePath: "utils_test_v2.json",
			},
			want:    "abcdefg",
			wantErr: false,
		},
		{
			name: "pod uid not found V2 format",
			args: args{
				deviceID: "not-exist",
				filePath: "utils_test_v2.json",
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetPodUIDByDeviceID(tt.args.deviceID, tt.args.filePath)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetPodUIDByDeviceID() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("GetPodUIDByDeviceID() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSandboxes(t *testing.T) {
	type args struct {
		ctx    context.Context
		client cri.RuntimeService
		podUid string
	}

	tests := []struct {
		name    string
		args    args
		want    []*runtimeapi.PodSandbox
		wantErr bool
	}{
		{
			name: "get sandboxes",
			args: args{
				ctx:    context.Background(),
				client: fakeCli,
				podUid: "",
			},
			want:    []*runtimeapi.PodSandbox{},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSandboxes(tt.args.ctx, tt.args.client, tt.args.podUid)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSandboxes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetSandboxes() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSandboxPid(t *testing.T) {
	type args struct {
		ctx       context.Context
		client    cri.RuntimeService
		sandboxId string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "sandboxStatus empty",
			args: args{
				ctx:       context.Background(),
				client:    fakeCli,
				sandboxId: "",
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "parseJsonStr err",
			args: args{
				ctx:       context.Background(),
				client:    fakeCli,
				sandboxId: "test1",
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "get pid err",
			args: args{
				ctx:       context.Background(),
				client:    fakeCli,
				sandboxId: "test1",
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "get pid success",
			args: args{
				ctx:       context.Background(),
				client:    fakeCli,
				sandboxId: "test1",
			},
			want:    "12345",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			switch tt.name {
			case "parseJsonStr err":
				fakeCli.SetFakeSandboxes([]*critesting.FakePodSandbox{
					{
						runtimeapi.PodSandboxStatus{Id: "test1"},
						"",
					},
				})
			case "get pid err":
				mock := &MockUtils{}
				defer mock.Reset()

				mock.mockParseJsonStr(`{"pid":fsfd,"aaa":"ddfd","fdfsd":false}`)
			case "get pid success":
				mock := &MockUtils{}
				defer mock.Reset()

				mock.mockParseJsonStr(`{"pid":12345,"aaa":"ddfd","fdfsd":false}`)
			}

			got, err := GetSandboxPid(tt.args.ctx, tt.args.client, tt.args.sandboxId)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSandboxPid() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("GetSandboxPid() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_parseJsonStr(t *testing.T) {
	type args struct {
		jsonString string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "parseJsonStr err",
			args: args{
				jsonString: "",
			},
			want:    "",
			wantErr: true,
		},
		{
			name: "parseJsonStr success",
			args: args{
				jsonString: `{"pid":12345,"aaa":"ddfd","fdfsd":false}`,
			},
			want:    "12345",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseJsonStr(tt.args.jsonString)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseJsonStr() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			pid := ""
			if got != nil {
				pid = strconv.Itoa(int(got["pid"].(float64)))
			}
			if !reflect.DeepEqual(pid, tt.want) {
				t.Errorf("parseJsonStr() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEnsureDevicePermission(t *testing.T) {
	type args struct {
		path string
		perm int
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "file not exist",
			args: args{
				path: "",
				perm: 0666,
			},
			wantErr: true,
		},
		{
			name: "file already has permission",
			args: args{
				path: "",
				perm: 0666,
			},
			wantErr: false,
		},
		{
			name: "file's permission can be set",
			args: args{
				path: "",
				perm: 0666,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		mock := &MockUtils{}
		defer mock.Reset()

		mock.mockStat(tt.name)
		mock.mockChmod()
		t.Run(tt.name, func(t *testing.T) {
			if err := EnsureDevicePermission(tt.args.path, tt.args.perm); (err != nil) != tt.wantErr {
				t.Errorf("EnsureDevicePermission() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetAllDeviceIdsFromKubeletCheckpoint(t *testing.T) {
	type args struct {
		filePath string
	}
	tests := []struct {
		name    string
		args    args
		want    []string
		wantErr bool
	}{
		{
			name:    "get all dev ids V1",
			args:    args{filePath: "utils_test.json"},
			want:    []string{"urma_7", "urma_29"},
			wantErr: false,
		},
		{
			name:    "get all dev ids V2",
			args:    args{filePath: "utils_test_v2.json"},
			want:    []string{"urma_7", "urma_29", "urma_111", "urma_222"},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetAllDeviceIdsFromKubeletCheckpoint(tt.args.filePath)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetAllDeviceIdsFromKubeletCheckpoint() error = %v, wantErr %v", err, tt.wantErr)
			}
			sort.Strings(tt.want)
			sort.Strings(got)
			assert.Equalf(t, tt.want, got, "GetAllDeviceIdsFromKubeletCheckpoint()")
		})
	}
}

func Test_parseKubeletCheckpoint(t *testing.T) {
	tests := []struct {
		name      string
		filePath  string
		wantCount int
		wantErr   bool
	}{
		{
			name:      "parse V1 format",
			filePath:  "utils_test.json",
			wantCount: 2,
			wantErr:   false,
		},
		{
			name:      "parse V2 format",
			filePath:  "utils_test_v2.json",
			wantCount: 3,
			wantErr:   false,
		},
		{
			name:      "file not exist",
			filePath:  "not_exist.json",
			wantCount: 0,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseKubeletCheckpoint(tt.filePath)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseKubeletCheckpoint() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			switch c := got.(type) {
			case *FileContentV1:
				assert.Equal(t, tt.wantCount, len(c.Data.PodDeviceEntries))
			case *FileContentV2:
				assert.Equal(t, tt.wantCount, len(c.Data.PodDeviceEntries))
			default:
				t.Errorf("unknown type")
			}
		})
	}
}
