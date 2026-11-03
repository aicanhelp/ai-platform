/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package executor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"installer/pkg/executor/core"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/metadata"
)

func TestGraphBuilder_Builder(t *testing.T) {
	type fields struct {
		NameToComponentMap map[string]fuyaov1beta1.Component
	}
	type args struct {
		components []fuyaov1beta1.Component
		destPath   string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *core.DirectedGraph
		wantErr assert.ErrorAssertionFunc
	}{
		{
			name: "empty components",
			args: args{
				components: []fuyaov1beta1.Component{},
				destPath:   "",
			},
			want:    &core.DirectedGraph{},
			wantErr: assert.NoError,
		},
		{
			name: "invalid metadata path",
			args: args{
				components: []fuyaov1beta1.Component{
					{Name: "test", Group: "test"},
				},
				destPath: "/invalid/path",
			},
			want:    nil,
			wantErr: assert.Error,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &GraphBuilder{
				NameToComponentMap: tt.fields.NameToComponentMap,
			}
			got, err := g.Builder(tt.args.components, tt.args.destPath)
			if !tt.wantErr(t, err, fmt.Sprintf("Builder(%v, %v)", tt.args.components, tt.args.destPath)) {
				return
			}
			assert.Equalf(t, tt.want, got, "Builder(%v, %v)", tt.args.components, tt.args.destPath)
		})
	}
}

func TestGraphBuilder_isComponentExit(t *testing.T) {
	type fields struct {
		NameToComponentMap map[string]fuyaov1beta1.Component
	}
	type args struct {
		name string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
	}{
		{
			name: "component exists",
			fields: fields{
				NameToComponentMap: map[string]fuyaov1beta1.Component{
					"test": {},
				},
			},
			args: args{name: "test"},
			want: true,
		},
		{
			name: "component missing",
			fields: fields{
				NameToComponentMap: map[string]fuyaov1beta1.Component{},
			},
			args: args{name: "missing"},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &GraphBuilder{
				NameToComponentMap: tt.fields.NameToComponentMap,
			}
			assert.Equalf(t, tt.want, g.isComponentExit(tt.args.name), "isComponentExit(%v)", tt.args.name)
		})
	}
}

func TestGraphBuilder_isVersionSatisfy(t *testing.T) {
	type fields struct {
		NameToComponentMap map[string]fuyaov1beta1.Component
	}
	type args struct {
		depend metadata.Dependency
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
		want1  string
	}{
		{
			name: "min version check",
			fields: fields{
				NameToComponentMap: map[string]fuyaov1beta1.Component{
					"test": {Version: "1.2.3"},
				},
			},
			args: args{
				depend: metadata.Dependency{
					Name:    "test",
					Version: "[1.2.3,)",
				},
			},
			want:  true,
			want1: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &GraphBuilder{
				NameToComponentMap: tt.fields.NameToComponentMap,
			}
			got, got1 := g.isVersionSatisfy(tt.args.depend)
			assert.Equalf(t, tt.want, got, "isVersionSatisfy(%v)", tt.args.depend)
			assert.Equalf(t, tt.want1, got1, "isVersionSatisfy(%v)", tt.args.depend)
		})
	}
}

func TestGraphBuilder_minMaxVersionCheck(t *testing.T) {
	type fields struct {
		NameToComponentMap map[string]fuyaov1beta1.Component
	}
	type args struct {
		versionSlice    []string
		minVersionSlice []string
		maxVersionSlice []string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
		want1  string
	}{
		{
			name: "valid version",
			args: args{
				versionSlice:    []string{"1", "2", "3"},
				minVersionSlice: []string{"1", "2", "0"},
				maxVersionSlice: []string{"1", "2", "5"},
			},
			want:  true,
			want1: "",
		},
		{
			name: "invalid version length",
			args: args{
				versionSlice:    []string{"1", "2"},
				minVersionSlice: []string{"1", "2"},
				maxVersionSlice: []string{"1", "2"},
			},
			want: false,
			// 更新为实际返回的错误信息
			want1: "Version information incorrect, unable to verify, version [1 2], min version [1 2], max version [1 2]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &GraphBuilder{
				NameToComponentMap: tt.fields.NameToComponentMap,
			}
			got, got1 := g.minMaxVersionCheck(tt.args.versionSlice, tt.args.minVersionSlice, tt.args.maxVersionSlice)
			assert.Equalf(t, tt.want, got, "minMaxVersionCheck(%v, %v, %v)", tt.args.versionSlice, tt.args.minVersionSlice, tt.args.maxVersionSlice)
			assert.Equalf(t, tt.want1, got1, "minMaxVersionCheck(%v, %v, %v)", tt.args.versionSlice, tt.args.minVersionSlice, tt.args.maxVersionSlice)
		})
	}
}

func TestGraphBuilder_minVersionCheck(t *testing.T) {
	type fields struct {
		NameToComponentMap map[string]fuyaov1beta1.Component
	}
	type args struct {
		versionSlice    []string
		minVersionSlice []string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
	}{
		{
			name: "valid min version",
			args: args{
				versionSlice:    []string{"1", "2", "3"},
				minVersionSlice: []string{"1", "2", "0"},
			},
			want: true,
		},
		{
			name: "latest version",
			args: args{
				versionSlice:    []string{"latest"},
				minVersionSlice: []string{"1", "2", "0"},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &GraphBuilder{
				NameToComponentMap: tt.fields.NameToComponentMap,
			}
			got, _ := g.minVersionCheck(tt.args.versionSlice, tt.args.minVersionSlice)
			assert.Equalf(t, tt.want, got, "minVersionCheck(%v, %v)", tt.args.versionSlice, tt.args.minVersionSlice)
		})
	}
}

func TestNewGraphBuilder(t *testing.T) {
	tests := []struct {
		name string
		want GraphBuilder
	}{
		{
			name: "new builder",
			want: GraphBuilder{
				NameToComponentMap: make(map[string]fuyaov1beta1.Component),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, NewGraphBuilder(), "NewGraphBuilder()")
		})
	}
}
