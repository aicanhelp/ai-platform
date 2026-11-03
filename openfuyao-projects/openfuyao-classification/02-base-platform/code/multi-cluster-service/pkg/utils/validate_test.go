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
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	v1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

func TestIsAccessPass(t *testing.T) {
	type args struct {
		c         kubernetes.Interface
		name      string
		operation string
	}
	tests := []struct {
		name    string
		args    args
		want    bool
		wantErr bool
		Data    []rbacv1.ClusterRole
	}{
		{
			name: "access pass",
			args: args{
				c:         fake.NewSimpleClientset(),
				name:      "me",
				operation: "join:cluster",
			},
			want: true,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "me",
						Labels: map[string]string{
							CreatedByopenFuyaoLabel: "true",
							DefaultopenFuyaoLabel:   DefaultPlatformRoleLabel,
						},
					},
					Rules: []rbacv1.PolicyRule{
						{
							ResourceNames: []string{"cluster-1", "cluster-2"},
							Resources:     []string{"cluster", "cluster/proxy"},
						},
					},
				},
			},
		},
		{
			name: "return false",
			args: args{
				c:         fake.NewSimpleClientset(),
				name:      "ooo",
				operation: "unjoin:cluster",
			},
			want: false,
			Data: []rbacv1.ClusterRole{
				{
					ObjectMeta: metav1.ObjectMeta{},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, cr := range tt.Data {
				_, err := tt.args.c.RbacV1().ClusterRoles().Create(context.TODO(), &cr, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("bad")
				}
			}
			got, err := IsAccessPass(tt.args.c, tt.args.name, tt.args.operation)
			if (err != nil) != tt.wantErr {
				t.Errorf("IsAccessPass() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("IsAccessPass() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHasKarmadaServiceOrMCService(t *testing.T) {
	type args struct {
		c kubernetes.Interface
	}
	tests := []struct {
		name  string
		args  args
		want  bool
		want1 bool
		DataS []corev1.Namespace
		DataD []v1.Deployment
	}{
		{
			name: "none",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			want:  false,
			want1: false,
		},
		{
			name: "karmada-system",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			want:  true,
			want1: false,
			DataS: []corev1.Namespace{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "karmada-system",
					},
				},
			},
			DataD: []v1.Deployment{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "karmada-apiserver",
					},
				},
			},
		},
		{
			name: " has ms",
			args: args{
				c: fake.NewSimpleClientset(),
			},
			want: false,
			DataS: []corev1.Namespace{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "karmada-system",
					},
				},
			},
			want1: false,
			DataD: []v1.Deployment{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "multicluster-service",
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name != "none" {
				for _, cr := range tt.DataS {
					_, err := tt.args.c.CoreV1().Namespaces().Create(context.TODO(), &cr, metav1.CreateOptions{})
					if err != nil {
						t.Fatalf("namespace error")
					}
				}
				for _, cd := range tt.DataD {
					_, err := tt.args.c.AppsV1().Deployments(tt.name).Create(context.TODO(), &cd, metav1.CreateOptions{})
					if err != nil {
						t.Fatalf("deployment error")
					}
				}
			}
			got, got1 := HasKarmadaServiceOrMCService(tt.args.c)
			if got != tt.want {
				t.Errorf("HasKarmadaServiceOrMCService() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("HasKarmadaServiceOrMCService() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
func TestIsNamespaceDeleted(t *testing.T) {
	type args struct {
		c   kubernetes.Interface
		tag string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		Data    []corev1.Namespace
	}{
		{
			name: "name space deleted",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "user",
			},
			wantErr: true,
			Data: []corev1.Namespace{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user",
						Namespace: "",
					},
					Spec: corev1.NamespaceSpec{
						Finalizers: []corev1.FinalizerName{
							"test",
						},
					},
					Status: corev1.NamespaceStatus{
						Phase: corev1.NamespacePhase("Terminating"),
					},
				},
			},
		},
		{
			name: "namespace not found",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "user",
			},
			wantErr: false,
			Data:    nil,
		},
		{
			name: "namespace with empty finalizers",
			args: args{
				c:   fake.NewSimpleClientset(),
				tag: "user",
			},
			wantErr: true,
			Data: []corev1.Namespace{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user",
						Namespace: "",
					},
					Spec: corev1.NamespaceSpec{
						Finalizers: []corev1.FinalizerName{},
					},
					Status: corev1.NamespaceStatus{
						Phase: corev1.NamespacePhase("Terminating"),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, cr := range tt.Data {
				_, err := tt.args.c.CoreV1().Namespaces().Create(context.TODO(), &cr, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("error %v", err)
				}
			}
			if err := isNamespaceDeleted(tt.args.c, tt.args.tag); (err != nil) != tt.wantErr {
				t.Errorf("isNamespaceDeleted() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestIsNameStringValid(t *testing.T) {
	type args struct {
		name string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "success",
			args: args{
				name: "userfinal",
			},
			want: true,
		},
		{
			name: "has space",
			args: args{
				name: "      /",
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsNameStringValid(tt.args.name); got != tt.want {
				t.Errorf("IsNameStringValid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsLabelsValid(t *testing.T) {
	type args struct {
		labels map[string]string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "all ok",
			args: args{
				labels: map[string]string{
					"som": "ok",
				},
			},
			want: true,
		},
		{
			name: "one no ok",
			args: args{
				labels: map[string]string{
					"/": "yes",
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsLabelsValid(tt.args.labels); got != tt.want {
				t.Errorf("IsLabelsValid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsKarmadaAPIServiceReady(t *testing.T) {
	tests := []struct {
		name       string
		url        string
		wantErr    bool
		httpErr    error
		statusCode int
	}{
		{
			name:       "Karmada service ready",
			url:        "https://karmada-ready",
			wantErr:    false,
			statusCode: http.StatusOK,
		},
		{
			name:       "Karmada service not ready",
			url:        "https://karmada-not-ready",
			wantErr:    true,
			statusCode: http.StatusServiceUnavailable,
		},
		{
			name:    "Karmada service timeout",
			url:     "https://karmada-timeout",
			wantErr: true,
			httpErr: errors.New("connection timeout"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyMethod(reflect.TypeOf(&http.Client{}),
				"Get", func(_ *http.Client, url string) (*http.Response, error) {
					if tt.httpErr != nil {
						return nil, tt.httpErr
					}
					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       ioutil.NopCloser(strings.NewReader("")),
					}, nil
				})
			defer patch.Reset()
			err := isKarmadaAPIServiceReady(tt.url)
			if (err != nil) != tt.wantErr {
				t.Errorf("isKarmadaAPIServiceReady() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestIsClusterUIDUnique(t *testing.T) {
	type args struct {
		clusterclient kubernetes.Interface
		karmadaclient *versioned.Clientset
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "ok",
			args: args{
				clusterclient: fake.NewSimpleClientset(&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{
					Name: "kube",
					UID:  "unique-uid",
				}}),
				karmadaclient: &versioned.Clientset{},
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			got, err := IsClusterUIDUnique(tt.args.clusterclient, tt.args.karmadaclient)
			if (err != nil) != tt.wantErr {
				t.Errorf("IsClusterUIDUnique() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("IsClusterUIDUnique() got = %v, want %v", got, tt.want)
			}
		})
	}
}
