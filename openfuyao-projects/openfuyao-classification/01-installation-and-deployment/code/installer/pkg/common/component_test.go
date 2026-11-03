/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package common

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
)

func TestGetComponentManifestsWithHubTmp(t *testing.T) {
	type args struct {
		assetAbsPath string
		pkgPath      string
	}
	tests := []struct {
		name    string
		args    args
		want    *manifests.ComponentManifests
		wantErr assert.ErrorAssertionFunc
	}{
		{
			name: "empty paths",
			args: args{
				assetAbsPath: "",
				pkgPath:      "",
			},
			want:    nil,
			wantErr: assert.Error,
		},
		{
			name: "non-existent asset path",
			args: args{
				assetAbsPath: "/invalid/path",
				pkgPath:      "/invalid/path",
			},
			want:    nil,
			wantErr: assert.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetComponentManifestsWithHubTmp(tt.args.assetAbsPath, tt.args.pkgPath)
			if !tt.wantErr(t, err, fmt.Sprintf("GetComponentManifestsWithHubTmp(%v, %v)", tt.args.assetAbsPath, tt.args.pkgPath)) {
				return
			}
			assert.Equalf(t, tt.want, got, "GetComponentManifestsWithHubTmp(%v, %v)", tt.args.assetAbsPath, tt.args.pkgPath)
		})
	}
}

func TestIsContainsComponentByCfg(t *testing.T) {
	testCluster := &cluster.Cluster{
		Spec: cluster.Spec{
			Components: []cluster.Component{
				{Name: "etcd"},
				{Name: "api-server"},
			},
		},
	}

	type args struct {
		cluster2 *cluster.Cluster
		name     string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "component exists",
			args: args{
				cluster2: testCluster,
				name:     "etcd",
			},
			want: true,
		},
		{
			name: "component missing",
			args: args{
				cluster2: testCluster,
				name:     "missing-component",
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, IsContainsComponentByCfg(tt.args.cluster2, tt.args.name), "IsContainsComponentByCfg(%v, %v)", tt.args.cluster2, tt.args.name)
		})
	}
}
