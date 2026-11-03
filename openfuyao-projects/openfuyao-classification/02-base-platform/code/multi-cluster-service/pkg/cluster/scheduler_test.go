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
	"testing"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestOperationClientMemberAuthPolicy(t *testing.T) {
	type fields struct {
		MgrClient              client.Client
		KarmadaClient          *kubernetes.Clientset
		KarmadaVersionedClient *versioned.Clientset
	}
	type args struct {
		data       []byte
		clusterTag string
		method     string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
		mockErr error
	}{
		{
			name: "success",
			fields: fields{
				MgrClient: CreatefakeClient([]byte(`{"apiversion":v2}`)),
			},
			args: args{
				data: []byte(`{
  "userinfo": {
    "user1": {
      "identityname": "user1",
      "apigroup": "admin",
      "platformadmin": true,
      "memberclusters": ["cluster1", "cluster2"]
    },
    "admin": {
      "identityname": "user2",
      "apigroup": "user",
      "platformadmin": false,
      "memberclusters": ["cluster1"]
    }
  }
}`),
				clusterTag: "testg",
				method:     "join",
			},
			wantErr: true,
			mockErr: nil,
		},
		{
			name:   "no join",
			fields: fields{},
			args: args{
				data:       make([]byte, 0),
				clusterTag: "y",
				method:     "s",
			},
			wantErr: true,
		},
		{
			name:   "host",
			fields: fields{},
			args: args{
				data:       make([]byte, 0),
				clusterTag: "host",
				method:     "join",
			},
			wantErr: false,
		},
		{
			name:   "invalid data",
			fields: fields{},
			args: args{
				data:   []byte(`sadas:sa`),
				method: "unjoin",
			},
			wantErr: true,
			mockErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &OperationClient{
				MgrClient:              tt.fields.MgrClient,
				KarmadaClient:          tt.fields.KarmadaClient,
				KarmadaVersionedClient: tt.fields.KarmadaVersionedClient,
			}
			if err := c.MemberAuthPolicy(tt.args.data, tt.args.clusterTag, tt.args.method); (err != nil) != tt.wantErr {
				t.Errorf("MemberAuthPolicy() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCheckClusterrolebinding(t *testing.T) {
	type args struct {
		c     kubernetes.Interface
		uname string
	}
	tests := []struct {
		name    string
		args    args
		want    bool
		wantErr bool
	}{
		{
			name: "matched platform-admin role",
			args: args{c: func() kubernetes.Interface {
				client := fake.NewSimpleClientset(&rbacv1.ClusterRoleBinding{
					ObjectMeta: metav1.ObjectMeta{Name: "testuser-platform-admin"}})
				return client
			}(), uname: "testuser",
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "matched cluster-admin role",
			args: args{c: func() kubernetes.Interface {
				fakeClient := fake.NewSimpleClientset(&rbacv1.ClusterRoleBinding{
					ObjectMeta: metav1.ObjectMeta{Name: "testuser-cluster-admin"}})
				return fakeClient
			}(), uname: "testuser",
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "no matching role",
			args: args{c: func() kubernetes.Interface {
				client := fake.NewSimpleClientset(&rbacv1.ClusterRoleBinding{
					ObjectMeta: metav1.ObjectMeta{Name: "someone-else"}})
				return client
			}(), uname: "testuser",
			},
			want:    false,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := checkClusterrolebinding(tt.args.c, tt.args.uname)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkClusterrolebinding() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("checkClusterrolebinding() got = %v, want %v", got, tt.want)
			}
		})
	}
}
