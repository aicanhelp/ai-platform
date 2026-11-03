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
	"bytes"
	"crypto/rand"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

func TestDecodeBase64String(t *testing.T) {
	type args struct {
		input string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			"TestInvalidB64Str",
			args{
				"not-valid",
			},
			"",
			true,
		},
		{
			"TestValidB64Str",
			args{
				"Zm9vYmFy",
			},
			"foobar",
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DecodeBase64String(tt.args.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("DecodeBase64String() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("DecodeBase64String() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func getFakeSecretClient() kubernetes.Interface {
	return fake.NewSimpleClientset(
		&v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-key",
				Namespace: "openfuyao-system",
			},
			Data: map[string][]byte{
				"plugin-management-service-symmetric-key": []byte("test-key"),
			},
		},
		&v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "not-a-key",
				Namespace: "openfuyao-system",
			},
			Data: map[string][]byte{
				"username": []byte("test-username"),
				"password": []byte("test-password"),
			},
		},
	)
}

func TestGetSecret(t *testing.T) {
	fakeClient := getFakeSecretClient()
	type args struct {
		secret    string
		namespace string
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			"TestGetSecretNotFound",
			args{
				"xx",
				"yy",
			},
			true,
		},
		{
			"TestGetSecretSuccess",
			args{
				"test-key",
				"openfuyao-system",
			},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := GetSecret(fakeClient, tt.args.secret, tt.args.namespace)
			if (err != nil) != tt.wantErr {
				t.Errorf("Getting secret %s in namespace %s error: wantError %t, get Error %t",
					tt.args.secret, tt.args.namespace, tt.wantErr, err != nil)
			}
		})
	}
}

func TestGetSecretSymmetricEncryptKey(t *testing.T) {
	fakeClient := getFakeSecretClient()
	type args struct {
		secretName string
	}
	tests := []struct {
		name    string
		args    args
		want    []byte
		wantErr bool
	}{
		{
			"TestGetKeyWrongName",
			args{
				"wrong-key-name",
			},
			nil,
			true,
		},
		{
			"TestGetKeyFieldNotFound",
			args{
				"not-a-key",
			},
			nil,
			true,
		},
		{
			"TestGetKeySuccess",
			args{
				"test-key",
			},
			[]byte("test-key"),
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSecretSymmetricEncryptKey(fakeClient, tt.args.secretName)
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

func TestEncryptDecrypt(t *testing.T) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("rand.Read key error: %v", err)
	}

	plain := []byte("hello-plugin-management-service")
	cipherText, err := Encrypt(plain, key)
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}
	if len(cipherText) == 0 {
		t.Fatalf("Encrypt() returned empty ciphertext")
	}

	got, err := Decrypt(cipherText, key)
	if err != nil {
		t.Fatalf("Decrypt() error = %v", err)
	}
	if !bytes.Equal(got, plain) {
		t.Fatalf("Decrypt() got %q, want %q", string(got), string(plain))
	}
}

func TestEncryptInvalidKey(t *testing.T) {
	key := []byte("short-key")
	_, err := Encrypt([]byte("x"), key)
	if err == nil {
		t.Fatalf("Encrypt() expected error for invalid key length")
	}
}

func TestDecryptInvalidKey(t *testing.T) {
	key := []byte("short-key")
	_, err := Decrypt([]byte("whatever"), key)
	if err == nil {
		t.Fatalf("Decrypt() expected error for invalid key length")
	}
}

func TestDecryptTamperedCiphertext(t *testing.T) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("rand.Read key error: %v", err)
	}

	plain := []byte("hello")
	cipherText, err := Encrypt(plain, key)
	if err != nil {
		t.Fatalf("Encrypt() error = %v", err)
	}

	cipherText[len(cipherText)-1] ^= 0xFF
	_, err = Decrypt(cipherText, key)
	if err == nil {
		t.Fatalf("Decrypt() expected error for tampered ciphertext")
	}
}

func TestDecryptCiphertextTooShort(t *testing.T) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("rand.Read key error: %v", err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("Decrypt() expected panic for too-short ciphertext")
		}
	}()
	_, _ = Decrypt([]byte("short"), key)
}
