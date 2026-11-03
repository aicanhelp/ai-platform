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

package authenticators

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"

	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
)

// TestPBKDF2EncryptorEncryptPassword test EncryptPassword interface
func TestPBKDF2EncryptorEncryptPassword(t *testing.T) {
	type args struct {
		rawPassword []byte
	}

	encryptor := &PBKDF2Encryptor{
		saltLength:    16,
		iterations:    100000,
		keyLength:     64,
		encryptMethod: sha256.New,
	}

	tests := []struct {
		name    string
		args    args
		want    []byte
		wantErr bool
	}{
		{
			"encrypt succeed but since the salt is generated totally randomly, want cannot equal to got",
			args{rawPassword: []byte("Soup4@LL")},
			[]byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3zc4JXvEr4j4ldx" +
				"lf541zErHyqRHE+I2ik7ww5M="),
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := encryptor.EncryptPassword(tt.args.rawPassword)
			if err != nil || len(got) == 0 {
				t.Errorf("EncryptPassword() got = %v, want %v", string(got), string(tt.want))
			}
		})
	}
}

// TestPBKDF2EncryptorVerifyPassword test Verify Password interface
func TestPBKDF2EncryptorVerifyPassword(t *testing.T) {
	type args struct {
		rawPassword       []byte
		encryptedPassword []byte
	}

	encryptor := &PBKDF2Encryptor{
		saltLength:    16,
		iterations:    100000,
		keyLength:     64,
		encryptMethod: sha256.New,
	}

	tests := []struct {
		name    string
		args    args
		want    bool
		wantErr bool
	}{
		{
			"verify password succeed",
			args{
				rawPassword: []byte("Soup4@LL"),
				encryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3zc4J" +
					"XvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
			true,
			false,
		},
		{
			"verify password fail: raw password does not match",
			args{
				rawPassword: []byte("soup4@LL"),
				encryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3zc4JX" +
					"vEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
			false,
			false,
		},
		{
			"verify password fail: cannot decode base64 encrypted password",
			args{
				rawPassword: []byte("soup4@LL"),
				encryptedPassword: []byte("r1R43niQz47OcynRlWibwtM+lmDNdgYVr84I6ZWDb0E8WOSZu/PZ46mnP7H/FyIlV7S6pIu8ir" +
					"FEQU4P988bPB2QTHJlaTISol+Hnl7SVkE#"),
			},
			false,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := encryptor.VerifyPassword(tt.args.rawPassword, tt.args.encryptedPassword)
			if (err != nil) != tt.wantErr {
				t.Errorf("VerifyPassword() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("VerifyPassword() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewPBKDF2Encryptor(t *testing.T) {
	tests := []struct {
		name string
		want *PBKDF2Encryptor
	}{
		{
			"successfully tested",
			&PBKDF2Encryptor{
				saltLength:    16,
				iterations:    100000,
				keyLength:     64,
				encryptMethod: sha256.New,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewPBKDF2Encryptor(); !equalPBKDF2Encryptor(got, tt.want) {
				t.Errorf("NewPBKDF2Encryptor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func equalPBKDF2Encryptor(a, b *PBKDF2Encryptor) bool {
	if a == nil || b == nil {
		return a == b
	}
	return a.saltLength == b.saltLength &&
		a.iterations == b.iterations &&
		a.keyLength == b.keyLength &&
		// 比较 encryptMethod 是否指向同一个函数
		fmt.Sprintf("%p", a.encryptMethod) == fmt.Sprintf("%p", b.encryptMethod)
}

func TestNewFuyaoPasswordAuthenticator(t *testing.T) {
	type args struct {
		k8sClient dynamic.Interface
		namespace string
	}
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	tests := []struct {
		name string
		args args
		want *FuyaoPasswordAuthenticator
	}{
		{
			"successfully init",
			args{
				k8sClient: fakeClient,
				namespace: "oauth-user",
			},
			&FuyaoPasswordAuthenticator{
				k8sClient: fakeClient,
				encryptor: NewPBKDF2Encryptor(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewFuyaoPasswordAuthenticator(tt.args.k8sClient); !reflect.DeepEqual(got.k8sClient, tt.want.k8sClient) {
				t.Errorf("NewFuyaoPasswordAuthenticator() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFuyaoPasswordAuthenticator_checkPasswordComplexity(t *testing.T) {
	type fields struct {
		k8sClient dynamic.Interface
		ns        string
		encryptor Encryptor
	}
	type args struct {
		username string
		passwd   []byte
	}

	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	encryptor := &PBKDF2Encryptor{saltLength: 16, iterations: 100000, keyLength: 64, encryptMethod: sha256.New}

	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
	}{
		{
			"succeed",
			fields{k8sClient: fakeClient, ns: "oauth-user", encryptor: encryptor},
			args{username: "admin", passwd: []byte("test@123")},
			true,
		},
		{
			"len fewer than 8",
			fields{k8sClient: fakeClient, ns: "oauth-user", encryptor: encryptor},
			args{username: "admin", passwd: []byte("test@12")},
			false,
		},
		{
			"no special chars",
			fields{k8sClient: fakeClient, ns: "oauth-user", encryptor: encryptor},
			args{username: "admin", passwd: []byte("test1234")},
			false,
		},
		{
			"same as username",
			fields{k8sClient: fakeClient, ns: "oauth-user", encryptor: encryptor},
			args{username: "admin", passwd: []byte("admin")},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := &FuyaoPasswordAuthenticator{
				k8sClient: tt.fields.k8sClient,
				encryptor: tt.fields.encryptor,
			}
			if got := a.checkPasswordComplexity(tt.args.username, tt.args.passwd); got != tt.want {
				t.Errorf("checkPasswordComplexity() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_reverseString(t *testing.T) {
	type args struct {
		s string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			"reverse string",
			args{s: "test@123"},
			"321@tset",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := reverseString(tt.args.s); got != tt.want {
				t.Errorf("reverseString() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFuyaoPasswordAuthenticator_fetchUserInfoAndStoredPassword(t *testing.T) {
	type fields struct {
		k8sClient dynamic.Interface
		ns        string
		encryptor Encryptor
	}
	type args struct {
		username string
	}

	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	encryptor := &PBKDF2Encryptor{
		saltLength:    16,
		iterations:    100000,
		keyLength:     64,
		encryptMethod: sha256.New,
	}
	userinfo := &user.DefaultInfo{
		Name:   "admin",
		Groups: []string{"system:authenticated"},
		Extra: map[string][]string{
			"first-login": {"true"},
		},
	}
	patches := createUserInfoPatches()
	defer patches.Reset()

	tests := []struct {
		name    string
		fields  fields
		args    args
		want    user.Info
		want1   []byte
		wantErr bool
	}{
		{
			"successfully fetch",
			fields{k8sClient: fakeClient, ns: "oauth-user", encryptor: encryptor}, args{username: "admin"}, userinfo,
			[]byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw" +
				"3zc4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="), false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := &FuyaoPasswordAuthenticator{
				k8sClient: tt.fields.k8sClient,
				encryptor: tt.fields.encryptor,
			}
			got, got1, err := a.fetchUserInfoAndStoredPassword(tt.args.username)
			if (err != nil) != tt.wantErr {
				t.Errorf("fetchUserInfoAndStoredPassword() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("fetchUserInfoAndStoredPassword() got = %v, want %v", got, tt.want)
			}
			if !bytes.Equal(got1, tt.want1) {
				t.Errorf("fetchUserInfoAndStoredPassword() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func createUserInfoPatches() *gomonkey.Patches {
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, _ string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: "admin"},
			Spec: fuyaouser.UserSpec{
				Username:     "admin",
				PlatformRole: "platform-admin",
				Description:  "A glocal platform user",
				FirstLogin:   true,
				EncryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3z" +
					"c4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
		}, nil
	})
	return patches
}

// TestFuyaoPasswordAuthenticator_AuthenticatePassword_Success tests successful password authentication
func TestFuyaoPasswordAuthenticator_AuthenticatePassword_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := createUserInfoPatches()
	defer patches.Reset()

	response, ok, err := authenticator.AuthenticatePassword(nil, "admin", []byte("Soup4@LL"))

	if err != nil {
		t.Errorf("AuthenticatePassword() error = %v", err)
		return
	}
	if !ok {
		t.Error("AuthenticatePassword() ok = false, want true")
	}
	if response == nil {
		t.Error("AuthenticatePassword() response = nil")
	}
	if response.User.GetName() != "admin" {
		t.Errorf("AuthenticatePassword() username = %v, want admin", response.User.GetName())
	}
}

// TestFuyaoPasswordAuthenticator_AuthenticatePassword_WrongPassword tests authentication with wrong password
func TestFuyaoPasswordAuthenticator_AuthenticatePassword_WrongPassword(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := createUserInfoPatches()
	defer patches.Reset()

	_, ok, err := authenticator.AuthenticatePassword(nil, "admin", []byte("wrongpassword"))

	if err != nil {
		t.Errorf("AuthenticatePassword() error = %v", err)
		return
	}
	if ok {
		t.Error("AuthenticatePassword() ok = true, want false")
	}
}

// TestFuyaoPasswordAuthenticator_AuthenticatePassword_UserNotFound tests authentication with non-existent user
func TestFuyaoPasswordAuthenticator_AuthenticatePassword_UserNotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, _ string) (*fuyaouser.User, error) {
		return nil, errors.New("user not found")
	})
	defer patches.Reset()

	_, ok, err := authenticator.AuthenticatePassword(nil, "nonexistent", []byte("password"))

	if err == nil {
		t.Error("AuthenticatePassword() error = nil, want error")
	}
	if ok {
		t.Error("AuthenticatePassword() ok = true, want false")
	}
}

// TestFuyaoPasswordAuthenticator_ConfirmPassword_Success tests successful password confirmation on first login
func TestFuyaoPasswordAuthenticator_ConfirmPassword_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	// Mock GetUserInfo to return first login user
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: username},
			Spec: fuyaouser.UserSpec{
				Username:         username,
				FirstLogin:       true,
				EncryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3z" +
					"c4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
		}, nil
	})
	// Mock PatchUserInfo to succeed
	patches.ApplyFunc(fuyaouser.PatchUserInfo, func(_ dynamic.Interface, username string, data []byte) error {
		return nil
	})
	defer patches.Reset()

	err := authenticator.ConfirmPassword(nil, "admin", []byte("NewPass@123"))

	if err != nil {
		t.Errorf("ConfirmPassword() error = %v", err)
	}
}

// TestFuyaoPasswordAuthenticator_ConfirmPassword_SamePassword tests confirmation with same password
func TestFuyaoPasswordAuthenticator_ConfirmPassword_SamePassword(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := createUserInfoPatches()
	defer patches.Reset()

	err := authenticator.ConfirmPassword(nil, "admin", []byte("Soup4@LL"))

	if err != fuyaoerrors.ErrPasswordSame {
		t.Errorf("ConfirmPassword() error = %v, want ErrPasswordSame", err)
	}
}

// TestFuyaoPasswordAuthenticator_ConfirmPassword_WeakPassword tests confirmation with weak password
func TestFuyaoPasswordAuthenticator_ConfirmPassword_WeakPassword(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := createUserInfoPatches()
	defer patches.Reset()

	err := authenticator.ConfirmPassword(nil, "admin", []byte("weak"))

	if err != fuyaoerrors.ErrPasswordTooWeak {
		t.Errorf("ConfirmPassword() error = %v, want ErrPasswordTooWeak", err)
	}
}

// TestFuyaoPasswordAuthenticator_ConfirmPassword_NotFirstLogin tests confirmation when not first login
func TestFuyaoPasswordAuthenticator_ConfirmPassword_NotFirstLogin(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	// Mock GetUserInfo to return user with FirstLogin = false
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: username},
			Spec: fuyaouser.UserSpec{
				Username:         username,
				FirstLogin:       false,
				EncryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3z" +
					"c4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
		}, nil
	})
	defer patches.Reset()

	err := authenticator.ConfirmPassword(nil, "admin", []byte("NewPass@123"))

	if err != fuyaoerrors.ErrNotFirstLogin {
		t.Errorf("ConfirmPassword() error = %v, want ErrNotFirstLogin", err)
	}
}

// TestFuyaoPasswordAuthenticator_ResetPassword_Success tests successful password reset
func TestFuyaoPasswordAuthenticator_ResetPassword_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	// Mock GetUserInfo to return existing user
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: username},
			Spec: fuyaouser.UserSpec{
				Username:         username,
				FirstLogin:       false,
				EncryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3z" +
					"c4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
		}, nil
	})
	// Mock PatchUserInfo to succeed
	patches.ApplyFunc(fuyaouser.PatchUserInfo, func(_ dynamic.Interface, username string, data []byte) error {
		return nil
	})
	defer patches.Reset()

	err := authenticator.ResetPassword(nil, "admin", []byte("Soup4@LL"), []byte("NewPass@123"))

	if err != nil {
		t.Errorf("ResetPassword() error = %v", err)
	}
}

// TestFuyaoPasswordAuthenticator_ResetPassword_SamePassword tests reset with same password
func TestFuyaoPasswordAuthenticator_ResetPassword_SamePassword(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	err := authenticator.ResetPassword(nil, "admin", []byte("samepass"), []byte("samepass"))

	if err != fuyaoerrors.ErrPasswordSame {
		t.Errorf("ResetPassword() error = %v, want ErrPasswordSame", err)
	}
}

// TestFuyaoPasswordAuthenticator_ResetPassword_WrongOldPassword tests reset with wrong old password
func TestFuyaoPasswordAuthenticator_ResetPassword_WrongOldPassword(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := createUserInfoPatches()
	defer patches.Reset()

	err := authenticator.ResetPassword(nil, "admin", []byte("wrongpassword"), []byte("NewPass@123"))

	if err != fuyaoerrors.ErrPasswordResetFailed {
		t.Errorf("ResetPassword() error = %v, want ErrPasswordResetFailed", err)
	}
}

// TestFuyaoPasswordAuthenticator_ResetPassword_WeakPassword tests reset with weak new password
func TestFuyaoPasswordAuthenticator_ResetPassword_WeakPassword(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := createUserInfoPatches()
	defer patches.Reset()

	err := authenticator.ResetPassword(nil, "admin", []byte("Soup4@LL"), []byte("weak"))

	if err != fuyaoerrors.ErrPasswordTooWeak {
		t.Errorf("ResetPassword() error = %v, want ErrPasswordTooWeak", err)
	}
}

// TestFuyaoPasswordAuthenticator_savePassword_Success tests successful password save
func TestFuyaoPasswordAuthenticator_savePassword_Success(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: username},
			Spec: fuyaouser.UserSpec{
				Username:   username,
				FirstLogin: true,
			},
		}, nil
	})
	patches.ApplyFunc(fuyaouser.PatchUserInfo, func(_ dynamic.Interface, username string, data []byte) error {
		return nil
	})
	defer patches.Reset()

	err := authenticator.savePassword("admin", []byte("NewPass@123"), true)

	if err != nil {
		t.Errorf("savePassword() error = %v", err)
	}
}

// TestFuyaoPasswordAuthenticator_savePassword_UserNotFound tests save with non-existent user
func TestFuyaoPasswordAuthenticator_savePassword_UserNotFound(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return nil, errors.New("user not found")
	})
	defer patches.Reset()

	err := authenticator.savePassword("admin", []byte("NewPass@123"), true)

	if err != fuyaoerrors.ErrPasswordAuthenticationFailed {
		t.Errorf("savePassword() error = %v, want ErrPasswordAuthenticationFailed", err)
	}
}

// TestFuyaoPasswordAuthenticator_savePassword_NotFirstLogin tests save when not first login
func TestFuyaoPasswordAuthenticator_savePassword_NotFirstLogin(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: username},
			Spec: fuyaouser.UserSpec{
				Username:   username,
				FirstLogin: false,
			},
		}, nil
	})
	defer patches.Reset()

	err := authenticator.savePassword("admin", []byte("NewPass@123"), true)

	if err != fuyaoerrors.ErrNotFirstLogin {
		t.Errorf("savePassword() error = %v, want ErrNotFirstLogin", err)
	}
}

// TestFuyaoPasswordAuthenticator_savePassword_PatchFailed tests save when PatchUserInfo fails
func TestFuyaoPasswordAuthenticator_savePassword_PatchFailed(t *testing.T) {
	scheme := runtime.NewScheme()
	fakeClient := dynamicfake.NewSimpleDynamicClient(scheme)
	authenticator := NewFuyaoPasswordAuthenticator(fakeClient)

	patches := gomonkey.NewPatches()
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: username},
			Spec: fuyaouser.UserSpec{
				Username:   username,
				FirstLogin: true,
			},
		}, nil
	})
	patches.ApplyFunc(fuyaouser.PatchUserInfo, func(_ dynamic.Interface, username string, data []byte) error {
		return fuyaoerrors.ErrFailToPatchSecret
	})
	defer patches.Reset()

	err := authenticator.savePassword("admin", []byte("NewPass@123"), true)

	if err != fuyaoerrors.ErrFailToPatchSecret {
		t.Errorf("savePassword() error = %v, want ErrFailToPatchSecret", err)
	}
}
