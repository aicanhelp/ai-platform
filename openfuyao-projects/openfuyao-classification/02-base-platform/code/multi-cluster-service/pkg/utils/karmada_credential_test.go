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
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	ktest "k8s.io/client-go/testing"
)

func TestDeleteKarmadaResources(t *testing.T) {
	type args struct {
		c   kubernetes.Interface
		tag string
	}
	tests := []struct {
		name          string
		args          args
		wantErr       bool
		mockDeleteErr error
		mockUpdateErr error
	}{
		{
			name: "success",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "pod",
			},
			wantErr: false,
		},
		{
			name: "fail delete",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "pod",
			},
			wantErr:       true,
			mockDeleteErr: fmt.Errorf("error"),
		},
		{
			name: "fail update",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "pod",
			},
			wantErr:       true,
			mockUpdateErr: fmt.Errorf("error"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name != "success" {
				patch := gomonkey.ApplyFunc(deleteKarmadaSecret, func(kubernetes.Interface, string) error {
					if tt.mockDeleteErr != nil {
						return fmt.Errorf("e")
					}
					return nil
				})
				batch := gomonkey.ApplyFunc(updateKarmadaClusterrole, func(kubernetes.Interface, string) error {
					if tt.mockUpdateErr != nil {
						return fmt.Errorf("e")
					}
					return nil
				})
				defer patch.Reset()
				defer batch.Reset()
			}

			if err := DeleteKarmadaResources(tt.args.c, tt.args.tag); (err != nil) != tt.wantErr {
				t.Errorf("DeleteKarmadaResources() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeleteKarmadaSecret(t *testing.T) {
	type args struct {
		c   kubernetes.Interface
		tag string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		mockErr error
	}{
		{
			name: "Success - Secret deleted",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "test-secret",
			},
			wantErr: false,
		},
		{
			name: "Fail - Secret deletion failed",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "test-secret",
			},
			wantErr: true,
			mockErr: errors.New("error"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockErr != nil {
				tt.args.c.(*fake.Clientset).PrependReactor("delete", "secrets",
					func(action ktest.Action) (bool, runtime.Object, error) {
						return true, nil, tt.mockErr
					})
			}
			err := deleteKarmadaSecret(tt.args.c, tt.args.tag)
			if (err != nil) != tt.wantErr {
				t.Errorf("deleteKarmadaSecret() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateKarmadaSecret(t *testing.T) {
	type args struct {
		c     kubernetes.Interface
		stObj *corev1.Secret
	}
	tests := []struct {
		name    string
		args    args
		want    *corev1.Secret
		wantErr bool
		mockErr error
	}{
		{
			name: "Success - Secret created",
			args: args{
				c: fake.NewSimpleClientset(),
				stObj: &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-secret",
						Namespace: "default",
					},
					Data: map[string][]byte{
						"key": []byte("value"),
					},
				},
			},
			want: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-secret",
					Namespace: "default",
				},
				Data: map[string][]byte{
					"key": []byte("value"),
				},
			},
			wantErr: false,
		},
		{
			name: "Fail - Create Secret error",
			args: args{
				c: fake.NewSimpleClientset(),
				stObj: &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-secret",
						Namespace: "default",
					},
				},
			},
			wantErr: true,
			mockErr: fmt.Errorf(""),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mockErr != nil {
				tt.args.c.(*fake.Clientset).PrependReactor("create", "secrets",
					func(action ktest.Action) (bool, runtime.Object, error) {
						return true, nil, tt.mockErr
					})
			}
			got, err := createKarmadaSecret(tt.args.c, tt.args.stObj)
			if (err != nil) != tt.wantErr {
				t.Errorf("createKarmadaSecret() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) && tt.wantErr == false {
				t.Errorf("createKarmadaSecret() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCheckUserAccessList(t *testing.T) {
	type args struct {
		c    kubernetes.Interface
		name string
	}
	tests := []struct {
		name    string
		args    args
		want    []string
		wantErr bool
		Data    []rbacv1.ClusterRole
		mockErr error
	}{
		{
			name: "success",
			args: args{
				c:    fake.NewSimpleClientset(),
				name: "test",
			},
			want:    []string{"cluster-1", "cluster-2"},
			wantErr: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "openfuyao:multicluster:platform:test",
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   "o",
						},
					},
					Rules: []rbacv1.PolicyRule{
						{
							Resources:     []string{"cluster", "cluster/proxy"},
							ResourceNames: []string{"cluster-1", "cluster-2"},
						},
					},
				},
			},
		},
		{
			name: "Error Retrieving karmada clusterrole",
			args: args{
				c:    fake.NewSimpleClientset(),
				name: "test failed",
			},
			want:    nil,
			wantErr: true,
			Data:    make([]rbacv1.ClusterRole, 0),
			mockErr: errors.New("eooro"),
		},
		{
			name: "username != name",
			args: args{
				c:    fake.NewSimpleClientset(),
				name: "test",
			},
			want:    nil,
			wantErr: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "openfuyao:multicluster:platform:0",
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   "yes",
						},
					},
					Rules: []rbacv1.PolicyRule{
						{
							Resources:     []string{"cluster"},
							ResourceNames: []string{"c3"},
						},
					},
				},
			},
		},
		{
			name: "crObj.Labels[DefaultopenFuyaoLabel] == DefaultPlatformRoleLabel",
			args: args{
				c:    fake.NewSimpleClientset(),
				name: "test",
			},
			want:    []string{"*"},
			wantErr: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "openfuyao:multicluster:platform:test",
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   DefaultPlatformRoleLabel,
						},
					},
					Rules: []rbacv1.PolicyRule{
						{
							Resources:     []string{"cluster/proxy"},
							ResourceNames: []string{"cluster"},
						},
					},
				},
			},
		},
		{
			name: "nil",
			args: args{
				c:    fake.NewSimpleClientset(),
				name: "user",
			},
			want:    nil,
			wantErr: false,
			Data:    []rbacv1.ClusterRole{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, cr := range tc.Data {
				_, err := tc.args.c.RbacV1().ClusterRoles().Create(context.TODO(), &cr, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("error: %v", err)
				}
			}
			if tc.mockErr != nil {
				tc.args.c.(*fake.Clientset).PrependReactor("list", "clusterroles",
					func(action ktest.Action) (bool, runtime.Object, error) {
						return true, nil, tc.mockErr
					})
			}
			got, err := CheckUserAccessList(tc.args.c, tc.args.name)
			if (err != nil) != tc.wantErr {
				t.Errorf("CheckUserAccessList() error = %v, wantErr %v", err, tc.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("CheckUserAccessList() got = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGetClusterRoleList(t *testing.T) {
	type args struct {
		c kubernetes.Interface
	}
	tests := []struct {
		name    string
		args    args
		want    map[string]*rbacv1.ClusterRoleList
		wantErr bool
		Data    []rbacv1.ClusterRole
		mockErr error
	}{
		{
			name: "ok",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			want: map[string]*rbacv1.ClusterRoleList{
				"cluster": {
					Items: nil,
				},
				"platform": {
					Items: []rbacv1.ClusterRole{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name: "test",
								Labels: map[string]string{
									CreatedByopenFuyaoLabel: "true",
									DefaultopenFuyaoLabel:   DefaultPlatformRoleLabel,
								},
							},
						},
					},
				},
			},
			wantErr: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   DefaultPlatformRoleLabel,
						},
						Name: "test",
					},
				},
			},
		},
		{
			name: "Error Retrieving karmada clusterrole list",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			want:    nil,
			wantErr: true,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "openfuyao:multicluster:platform:test",
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   DefaultMemCusterRoleLabel,
						},
					},
				},
			},
			mockErr: errors.New("error"),
		},
		{
			name: "success",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			wantErr: false,
			want: map[string]*rbacv1.ClusterRoleList{
				"platform": {
					Items: nil,
				},
				"cluster": {
					Items: []rbacv1.ClusterRole{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name: "test",
								Labels: map[string]string{
									DefaultopenFuyaoLabel:   DefaultMemCusterRoleLabel,
									CreatedByopenFuyaoLabel: "OK",
								},
							},
						},
					},
				},
			},
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "OK",
							DefaultopenFuyaoLabel:   DefaultMemCusterRoleLabel,
						},
						Name: "test",
					},
				},
			},
		},
		{
			name: "len <=1 ",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			want: map[string]*rbacv1.ClusterRoleList{
				"platform": {
					Items: nil,
				},
				"cluster": {
					Items: nil,
				},
			},
			wantErr: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "test",
						Labels: nil,
					},
				},
			},
			mockErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, cr := range tt.Data {
				_, err := tt.args.c.RbacV1().ClusterRoles().Create(context.TODO(), &cr, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("error: %v", err)
				}
			}
			if tt.mockErr != nil {
				tt.args.c.(*fake.Clientset).PrependReactor("list", "clusterroles",
					func(action ktest.Action) (bool, runtime.Object, error) {
						return true, nil, tt.mockErr
					})
			}
			got, err := GetClusterRoleList(tt.args.c)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetClusterRoleList() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetClusterRoleList() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestUpdateClusterRole(t *testing.T) {
	type args struct {
		c        kubernetes.Interface
		crList   *rbacv1.ClusterRoleList
		userInfo map[string][]string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		Data    []rbacv1.ClusterRole
	}{
		{
			name: "success",
			args: args{
				c: fake.NewSimpleClientset(),
				crList: &rbacv1.ClusterRoleList{
					Items: []rbacv1.ClusterRole{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name: "user",
								Labels: map[string]string{
									CreatedByopenFuyaoLabel: "users",
								},
							},
							Rules: []rbacv1.PolicyRule{
								{
									Resources:     []string{"clusters/proxy"},
									ResourceNames: []string{"cluster1"},
								},
							},
						},
					},
				},
				userInfo: map[string][]string{
					"user": {"user"},
				},
			},
			wantErr: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "user",
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   DefaultPlatformRoleLabel,
						},
					},
					Rules: []rbacv1.PolicyRule{
						{
							Resources:     []string{"cluster/proxy"},
							ResourceNames: []string{"user"},
						},
					},
				},
			},
		},
	}
	for _, ts := range tests {
		t.Run(ts.name, func(t *testing.T) {
			for _, cr := range ts.Data {
				_, err := ts.args.c.RbacV1().ClusterRoles().Create(context.TODO(), &cr, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("error: %v", err)
				}
			}

			if err := UpdateClusterRole(ts.args.c, ts.args.crList, ts.args.userInfo); (err != nil) != ts.wantErr {
				t.Errorf("UpdateClusterRole() error = %v, wantErr %v", err, ts.wantErr)
			}
		})
	}
}

func TestUpdatePlatformRole(t *testing.T) {
	type args struct {
		c        kubernetes.Interface
		crList   *rbacv1.ClusterRoleList
		userList []string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: " success to update ",
			args: args{
				c: fake.NewSimpleClientset(),
				crList: &rbacv1.ClusterRoleList{
					Items: []rbacv1.ClusterRole{
						{
							Rules: []rbacv1.PolicyRule{
								{
									Resources:     []string{"clusters/proxy"},
									ResourceNames: []string{"cluster1"},
								},
							},
							ObjectMeta: metav1.ObjectMeta{
								Name: "ap",
								Labels: map[string]string{
									CreatedByopenFuyaoLabel: "aps",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := UpdatePlatformRole(tt.args.c, tt.args.crList, tt.args.userList); (err != nil) != tt.wantErr {
				t.Errorf("UpdatePlatformRole() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateClusterProxyResource(t *testing.T) {
	type args struct {
		c        kubernetes.Interface
		username string
		memList  []string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "success to create",
			args: args{
				c:        fake.NewSimpleClientset(),
				username: "HUAWEI",
				memList:  []string{"c1"},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := createClusterProxyResource(tt.args.c, tt.args.username, tt.args.memList); (err != nil) != tt.wantErr {
				t.Errorf("createClusterProxyResource() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreatClusterRoleResource(t *testing.T) {
	type args struct {
		c        kubernetes.Interface
		username string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "success  CreatClusterRoleResource",
			args: args{
				c:        fake.NewSimpleClientset(),
				username: "openfuyao",
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := creatClusterRoleResource(tt.args.c, tt.args.username); (err != nil) != tt.wantErr {
				t.Errorf("creatClusterRoleResource() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
