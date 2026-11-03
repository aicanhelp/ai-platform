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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
)

func TestGenerateKarmadaSecretExtra(t *testing.T) {
	newSecret := func(name string) *corev1.Secret {
		s := &corev1.Secret{}
		s.Name = name
		s.Data = map[string][]byte{TLSTokenData: []byte("token-" + name)}
		return s
	}

	t.Run("success creates both secrets", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		impName := "member1-" + DefaultImpersonatorPrefix
		clusterSecret, impersonatorSecret, err := GenerateKarmadaSecret(cs, newSecret("member1"), newSecret(impName), "member1")
		if err != nil {
			t.Fatalf("GenerateKarmadaSecret() error = %v", err)
		}
		if clusterSecret.Name != "member1" || impersonatorSecret.Name != impName {
			t.Errorf("names = %q, %q; want member1, %s", clusterSecret.Name, impersonatorSecret.Name, impName)
		}

		for _, name := range []string{"member1", impName} {
			_, err := cs.CoreV1().Secrets(DefaultKarmadaNamespace).Get(testCtx, name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("secret %s not created: %v", name, err)
			}
		}
	})

	t.Run("create cluster secret failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("create", "secrets",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})

		if _, _, err := GenerateKarmadaSecret(cs, newSecret("member1"), newSecret("member1-imp"), "member1"); err == nil {
			t.Error("GenerateKarmadaSecret() expected error when secret creation fails")
		}
	})
}

func TestRemoveString(t *testing.T) {
	tests := []struct {
		name  string
		slice []string
		str   string
		want  []string
	}{
		{
			name:  "remove existing element",
			slice: []string{"a", "b", "c"},
			str:   "b",
			want:  []string{"a", "c"},
		},
		{
			name:  "remove missing element keeps slice",
			slice: []string{"a", "b"},
			str:   "z",
			want:  []string{"a", "b"},
		},
		{
			name:  "remove from empty slice",
			slice: []string{},
			str:   "a",
			want:  nil,
		},
		{
			name:  "remove all occurrences",
			slice: []string{"a", "a", "b"},
			str:   "a",
			want:  []string{"b"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := removeString(tt.slice, tt.str); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("removeString() = %v, want %v", got, tt.want)
			}
		})
	}
}
