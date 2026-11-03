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
	"errors"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	er "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	ktest "k8s.io/client-go/testing"
)

func TestGeneratePlatformAuthPolicy(t *testing.T) {
	type args struct {
		c         kubernetes.Interface
		username  string
		operation string
		isAdmin   bool
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		mockErr error
	}{
		{
			name: "success to generate ",
			args: args{
				c:         fake.NewSimpleClientset(),
				username:  "myname",
				operation: "join",
				isAdmin:   true,
			},
			wantErr: false,
		},
		{
			name: "success unjoin not dound",
			args: args{
				c:         fake.NewSimpleClientset(),
				username:  "user",
				operation: "unjoin",
				isAdmin:   true,
			},
			wantErr: true,
			mockErr: errors.New("e"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockErr != nil {
				tt.args.c.(*fake.Clientset).PrependReactor("delete", "clusterrolebindings",
					func(action ktest.Action) (bool, runtime.Object, error) {
						return true, nil, tt.mockErr
					})
			}
			if err := GeneratePlatformAuthPolicy(tt.args.c, tt.args.username, tt.args.operation, tt.args.isAdmin); (err != nil) != tt.wantErr {
				t.Errorf("GeneratePlatformAuthPolicy() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateNamespace(t *testing.T) {
	type args struct {
		c     kubernetes.Interface
		nsObj *corev1.Namespace
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		mockErr error
	}{
		{
			name: "create namespace",
			args: args{
				c: fake.NewSimpleClientset(),
				nsObj: &corev1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user",
						Namespace: "",
					},
				},
			},
		},
		{
			name: "fail",
			args: args{
				c:     fake.NewSimpleClientset(),
				nsObj: &corev1.Namespace{},
			},
			wantErr: true,
			mockErr: errors.New("error"),
		},
		{
			name: "exist",
			args: args{
				c: fake.NewSimpleClientset(),
				nsObj: &corev1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test",
					},
				},
			},
			wantErr: false,
			mockErr: er.NewAlreadyExists(corev1.Resource("namespaces"), "test"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockErr != nil {
				tt.args.c.(*fake.Clientset).PrependReactor("create", "namespaces",
					func(action ktest.Action) (bool, runtime.Object, error) {
						createAction, ok := action.(ktest.CreateAction)
						if !ok {
							t.Errorf("expected CreateAction but got %T", action)
							return false, nil, nil
						}
						ns, ok := createAction.GetObject().(*corev1.Namespace)
						if !ok {
							t.Errorf("not ok for type *corev1.Namespace")
							return false, nil, nil
						}
						if ns.Name != tt.args.nsObj.Name {
							t.Errorf("expected namespace %s but got %s", tt.args.nsObj.Name, ns.Name)
						}
						return true, nil, tt.mockErr
					})
			}
			if err := createNamespace(tt.args.c, tt.args.nsObj); (err != nil) != tt.wantErr {
				t.Errorf("createNamespace() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeleteClusterResources(t *testing.T) {
	type args struct {
		c   kubernetes.Interface
		tag string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "delete cluster resources",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "",
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := DeleteClusterResources(tt.args.c, tt.args.tag); (err != nil) != tt.wantErr {
				t.Errorf("DeleteClusterResources() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGenerateClusterSecret(t *testing.T) {
	type args struct {
		c   kubernetes.Interface
		tag string
	}
	tests := []struct {
		name    string
		args    args
		want    *corev1.Secret
		want1   *corev1.Secret
		wantErr bool
		Data    []rbacv1.ClusterRoleBinding
		mockErr error
	}{
		{
			name: "fail",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "",
			},
			want:    nil,
			want1:   nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := GenerateClusterSecret(tt.args.c, tt.args.tag)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateClusterSecret() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GenerateClusterSecret() got = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("GenerateClusterSecret() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
