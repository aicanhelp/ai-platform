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
package cluster

import (
	"reflect"
	"testing"
)

func TestFilterClusterList(t *testing.T) {
	type args struct {
		clusterInfo *ListOfCluster
		memList     []string
	}
	tests := []struct {
		name string
		args args
		want *ListOfCluster
	}{
		{
			name: "shit",
			args: args{
				clusterInfo: &ListOfCluster{
					Info: map[string]*InformationOfCluster{},
				},
				memList: []string{"ok"},
			},
			want: &ListOfCluster{
				Info: map[string]*InformationOfCluster{},
			},
		},
		{
			name: "memlist = 1",
			args: args{
				clusterInfo: &ListOfCluster{
					Info: map[string]*InformationOfCluster{"*": &InformationOfCluster{ClusterName: "*"}},
				},
				memList: []string{"*"},
			},
			want: &ListOfCluster{
				Info: map[string]*InformationOfCluster{"*": &InformationOfCluster{ClusterName: "*"}},
			},
		},
		{
			name: "memlist equal 0",
			args: args{
				clusterInfo: &ListOfCluster{
					Info: map[string]*InformationOfCluster{},
				},
				memList: []string{},
			},
			want: &ListOfCluster{
				Info: map[string]*InformationOfCluster{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FilterClusterList(tt.args.clusterInfo, tt.args.memList); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("FilterClusterList() = %v, want %v", got, tt.want)
			}
		})
	}
}
