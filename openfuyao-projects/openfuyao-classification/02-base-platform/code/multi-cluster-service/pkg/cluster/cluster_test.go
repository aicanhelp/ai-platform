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

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestTransfer2AuthorizeOpts(t *testing.T) {
	type args struct {
		identityInfo IdentityCollection
	}
	tests := []struct {
		name string
		args args
		want AuthorizeOptions
	}{
		{
			name: "OK",
			args: args{
				identityInfo: IdentityCollection{
					UserInfo: map[string]*IdentityDescriptor{
						"user": &IdentityDescriptor{
							IdentityName: "my",
						},
					},
				},
			},
			want: AuthorizeOptions{
				Users: []*IdentityDescriptor{
					{
						IdentityName: "my",
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Transfer2AuthorizeOpts(tt.args.identityInfo); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Transfer2AuthorizeOpts() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTransfer2EditOpts(t *testing.T) {
	type args struct {
		clusterTag string
		tags       TagOfCluster
	}
	tests := []struct {
		name string
		args args
		want EditOptions
	}{
		{
			name: "niubi",
			args: args{
				clusterTag: "fine",
				tags: TagOfCluster{
					ClusterLabels: map[string]string{"A": "b"},
				},
			},
			want: EditOptions{
				ClusterLabels: map[string]string{"A": "b"},
				ClusterName:   "fine",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Transfer2EditOpts(tt.args.clusterTag, tt.args.tags); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Transfer2EditOpts() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTransfer2InspectOpts(t *testing.T) {
	type args struct {
		clusterTag string
	}
	tests := []struct {
		name string
		args args
		want InspectOptions
	}{
		{
			name: " fengle",
			args: args{
				clusterTag: "shangban",
			},
			want: InspectOptions{
				ClusterName: "shangban",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Transfer2InspectOpts(tt.args.clusterTag); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Transfer2InspectOpts() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTransfer2UnjoinOpts(t *testing.T) {
	type args struct {
		clusterTag string
	}
	tests := []struct {
		name string
		args args
		want UnjoinOptions
	}{
		{
			name: "diu",
			args: args{
				clusterTag: "lei",
			},
			want: UnjoinOptions{
				ClusterName: "lei",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Transfer2UnjoinOpts(tt.args.clusterTag); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Transfer2UnjoinOpts() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTransfer2JoinOpts(t *testing.T) {
	type args struct {
		clusterinfo InfoOfCluster
	}
	tests := []struct {
		name string
		args args
		want JoinOptions
	}{
		{
			name: "lou",
			args: args{
				clusterinfo: InfoOfCluster{
					ClusterLabels: map[string]string{"po": "gai"},
				},
			},
			want: JoinOptions{
				ClusterLabels: map[string]string{"po": "gai"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Transfer2JoinOpts(tt.args.clusterinfo); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Transfer2JoinOpts() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewClusteClient(t *testing.T) {
	type args struct {
		mgr      client.Client
		kClient  *kubernetes.Clientset
		kVClient *versioned.Clientset
	}
	tests := []struct {
		name string
		args args
		want OperationsInterface
	}{
		{
			name: "qixin",
			args: args{},
			want: &OperationClient{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewClusteClient(tt.args.mgr, tt.args.kClient, tt.args.kVClient); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewClusteClient() = %v, want %v", got, tt.want)
			}
		})
	}
}
