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

package util

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	"console-service/pkg/constant"
)

func TestDecrypt(t *testing.T) {

	tests := []struct {
		name       string
		key        []byte
		cipherText []byte
		want       []byte
		wantErr    bool
	}{
		{
			name: "正常解密",
			key:  []byte("12345678901234567890123456789012"),
			cipherText: func() []byte {
				key := []byte("12345678901234567890123456789012")
				plain := []byte("hello world")
				cipherText, _ := Encrypt(plain, key)
				return cipherText
			}(),
			want:    []byte("hello world"),
			wantErr: false,
		},
		{
			name:       "密钥长度错误",
			key:        []byte("short"),
			cipherText: []byte("xxxx"),
			want:       nil,
			wantErr:    true,
		},
		{
			name: "密文被破坏",
			key:  []byte("12345678901234567890123456789012"),
			cipherText: func() []byte {
				key := []byte("12345678901234567890123456789012")
				plain := []byte("hello world")
				cipherText, _ := Encrypt(plain, key)
				cipherText[len(cipherText)-1] ^= 0xFF // 篡改密文
				return cipherText
			}(),
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plain, err := Decrypt(tt.cipherText, tt.key)
			if (err != nil) == tt.wantErr {
				if !reflect.DeepEqual(plain, tt.want) {
					t.Errorf("Decrypt() got = %v, want %v", plain, tt.want)
				}
			} else {
				t.Errorf("Decrypt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}

func TestEncrypt(t *testing.T) {

	tests := []struct {
		name      string
		plainText []byte
		key       []byte
		wantErr   bool
	}{
		{
			name:      "正常加密",
			plainText: []byte("hello world"),
			key:       []byte("12345678901234567890123456789012"),
			wantErr:   false,
		},
		{
			name:      "密钥长度错误",
			plainText: []byte("hello world"),
			key:       []byte("short"),
			wantErr:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Encrypt(tt.plainText, tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("Encrypt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func getSymmetricKeyTestClient() kubernetes.Interface {
	return fake.NewSimpleClientset(
		&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "empty-secret",
				Namespace: constant.ConsoleServiceDefaultNamespace,
			},
		},
		&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "symmetric-key",
				Namespace: constant.ConsoleServiceDefaultNamespace,
			},
			Data: map[string][]byte{
				"console-service-symmetric-key": []byte("test-key"),
			},
		},
	)
}

func TestGetSecretSymmetricEncryptKey(t *testing.T) {
	client := getSymmetricKeyTestClient()

	tests := []struct {
		name       string
		secretName string
		want       []byte
		wantErr    bool
	}{
		{
			"TestNonExisting",
			"non-existing",
			nil,
			true,
		},
		{
			"TestNotContainingKey",
			"empty-secret",
			nil,
			true,
		},
		{
			"TestValidKey",
			"symmetric-key",
			[]byte("test-key"),
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSecretSymmetricEncryptKey(client, tt.secretName)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSecretSymmetricEncryptKey() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetSecretSymmetricEncryptKey() got = %v, want %v", got, tt.want)
			}
		})
	}
}
