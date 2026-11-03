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

package auth

import (
	"errors"
	"net/http"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"

	"console-service/pkg/constant"
	"console-service/pkg/utils/util"
)

var testSessionSecrets = []runtime.Object{
	&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "session1",
			Namespace: "session-secret",
		},
		Data: map[string][]byte{
			"AccessExpiry":  []byte("0"),
			"RefreshExpiry": []byte("0"),
			"SessionID":     []byte("11111111"),
			"AccessToken":   []byte("-11111111"),
			"RefreshToken":  []byte("-11111111"),
		},
		Type: corev1.SecretTypeOpaque,
	},
	&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "session2",
			Namespace: "session-secret",
		},
		Data: map[string][]byte{
			"AccessExpiry":  []byte("0"),
			"RefreshExpiry": []byte("0"),
			"SessionID":     []byte("22222222"),
			"AccessToken":   []byte("22222222"),
			"RefreshToken":  []byte("-22222222"),
		},
		Type: corev1.SecretTypeOpaque,
	},
	&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "session3",
			Namespace: "session-secret",
		},
		Data: map[string][]byte{
			"AccessExpiry":  []byte("0"),
			"RefreshExpiry": []byte("0"),
			"SessionID":     []byte("33333333"),
			"AccessToken":   []byte("33333333"),
			"RefreshToken":  []byte("33333333"),
		},
		Type: corev1.SecretTypeOpaque,
	},
	&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "session4",
			Namespace: "session-secret",
		},
		Data: map[string][]byte{
			"AccessExpiry":  []byte("0"),
			"RefreshExpiry": []byte(strconv.FormatInt(time.Now().Unix()+100, intBase10)),
			"SessionID":     []byte("44444444"),
			"AccessToken":   []byte("44444444"),
			"RefreshToken":  []byte("44444444"),
		},
		Type: corev1.SecretTypeOpaque,
	},
	&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "session5",
			Namespace: "session-secret",
		},
		Data: map[string][]byte{
			"AccessExpiry":  []byte(strconv.FormatInt(time.Now().Unix()+100, intBase10)),
			"RefreshExpiry": []byte(strconv.FormatInt(time.Now().Unix()+100, intBase10)),
			"SessionID":     []byte("55555555"),
			"AccessToken":   []byte("55555555"),
			"RefreshToken":  []byte("55555555"),
		},
		Type: corev1.SecretTypeOpaque,
	},
}

func getTestGetSessionClient() *fake.Clientset {
	return fake.NewSimpleClientset(testSessionSecrets...)
}

func patchCrypto(t *testing.T) {
	patchDecrypt := gomonkey.ApplyFunc(util.Decrypt, func(cipherText, key []byte) ([]byte, error) {
		if cipherText[0] != '-' {
			return cipherText, nil
		} else {
			return nil, errors.New("test error")
		}
	})

	patchEncrypt := gomonkey.ApplyFunc(util.Encrypt, func(plainText, key []byte) ([]byte, error) {
		if plainText[0] != '-' {
			return plainText, nil
		} else {
			return nil, errors.New("test error")
		}
	})

	patchSymKey := gomonkey.ApplyFunc(util.GetSecretSymmetricEncryptKey, func(clientset kubernetes.Interface,
		secretName string) ([]byte, error) {
		return []byte{}, nil
	})

	t.Cleanup(func() {
		defer patchDecrypt.Reset()
		defer patchEncrypt.Reset()
		patchSymKey.Reset()
	})
}

func TestNewStoreSession(t *testing.T) {
	token := &AccessRefreshToken{
		AccessToken:        "12345678",
		AccessTokenExpiry:  time.Now(),
		RefreshToken:       "12345678",
		RefreshTokenExpiry: time.Now(),
	}
	want := &SessionStore{
		accessTokenName:        []byte("12345678"),
		accessTokenExpiryName:  []byte(strconv.FormatInt(time.Now().Unix(), intBase10)),
		refreshTokenName:       []byte("12345678"),
		refreshTokenExpiryName: []byte(strconv.FormatInt(time.Now().Unix(), intBase10)),
	}
	got, err := NewStoreSession(token)
	delete(*got, sessionIDName)
	if err != nil {
		t.Errorf("NewStoreSession() error = %v, wantErr nil", err)
		return
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("NewStoreSession() got = %v, want %v", got, want)
	}
}

func TestGetSessionFailed(t *testing.T) {
	testClient := getTestGetSessionClient()
	t.Run("TestGetSymmetricKeyFailed", func(t *testing.T) {
		if _, err := GetSession(testClient, "non-existing"); err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	patchCrypto(t)

	tests := []struct {
		name      string
		sessionID string
		want      *SessionStore
		wantErr   bool
	}{
		{
			name:      "TestGetNonExistingSession",
			sessionID: "NonExisting",
			want:      nil,
			wantErr:   true,
		},
		{
			name:      "TestDecryptAccessFailed",
			sessionID: "11111111",
			want:      nil,
			wantErr:   true,
		},
		{
			name:      "TestDecryptRefreshFailed",
			sessionID: "22222222",
			want:      nil,
			wantErr:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSession(testClient, tt.sessionID)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSession() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetSession() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetSessionSucceeded(t *testing.T) {
	testClient := getTestGetSessionClient()
	patchCrypto(t)
	sessionID := "33333333"
	want := &SessionStore{
		"AccessExpiry":  []byte("0"),
		"RefreshExpiry": []byte("0"),
		"SessionID":     []byte("33333333"),
		"AccessToken":   []byte("33333333"),
		"RefreshToken":  []byte("33333333"),
	}
	got, err := GetSession(testClient, sessionID)
	if err != nil {
		t.Errorf("Should return nil error, but got %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("GetSession() got = %v, want %v", got, want)
	}
}

func TestStoreSession(t *testing.T) {
	testClient := getTestGetSessionClient()

	session := &SessionStore{
		"AccessExpiry":  []byte("0"),
		"RefreshExpiry": []byte("0"),
		"SessionID":     []byte("-33333333"),
		"AccessToken":   []byte("-33333333"),
		"RefreshToken":  []byte("-33333333"),
	}

	t.Run("GetSymKeyFailed", func(t *testing.T) {
		err := StoreSession(testClient, session, true)
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	patchCrypto(t)

	t.Run("UpdateNonExisting", func(t *testing.T) {
		err := StoreSession(testClient, session, true)
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	// 该用例结束后， session 被删除，后面不更新直接创建 (isUpdate=false)
	(*session)[sessionIDName] = []byte("33333333")
	t.Run("EncryptAccessFailed", func(t *testing.T) {
		err := StoreSession(testClient, session, true)
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	(*session)[accessTokenName] = []byte("33333333")
	t.Run("EncryptRefreshFailed", func(t *testing.T) {
		err := StoreSession(testClient, session, false)
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	(*session)[refreshTokenName] = []byte("33333333")
	t.Run("EncryptUpdateSuccess", func(t *testing.T) {
		err := StoreSession(testClient, session, false)
		if err != nil {
			t.Errorf("Should return nil error, get %v", err)
		}
	})
}

func TestUpdateSession(t *testing.T) {
	testClient := getTestGetSessionClient()
	patchCrypto(t)

	token := &AccessRefreshToken{
		AccessToken:        "12345678",
		AccessTokenExpiry:  time.Now(),
		RefreshToken:       "12345678",
		RefreshTokenExpiry: time.Now(),
	}

	t.Run("NonExistingSession", func(t *testing.T) {
		_, err := UpdateSession(testClient, "non-existing", token)
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	t.Run("ExistingSession", func(t *testing.T) {
		_, err := UpdateSession(testClient, "33333333", token)
		if err != nil {
			t.Errorf("Should return nil error, but got %v", err)
		}
	})
}

func TestGetTokenFromSessionID(t *testing.T) {
	testClient := getTestGetSessionClient()
	t.Run("TestGetSessionFailed", func(t *testing.T) {
		gotAccess, gotRefresh, err := GetTokenFromSessionID(testClient, "non-existing")
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
		if gotAccess != "" || gotRefresh != "" {
			t.Errorf("Expecting empty string, but get accessToken = %v, refreshToken = %v", gotAccess, gotRefresh)
		}
	})

	patchCrypto(t)

	t.Run("TestRefreshExpired", func(t *testing.T) {
		gotAccess, gotRefresh, err := GetTokenFromSessionID(testClient, "33333333")
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
		if gotAccess != "" || gotRefresh != "" {
			t.Errorf("Expecting empty string, but get accessToken = %v, refreshToken = %v",
				gotAccess, gotRefresh)
		}
	})

	t.Run("TestAccessExpired", func(t *testing.T) {
		gotAccess, gotRefresh, err := GetTokenFromSessionID(testClient, "44444444")
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
		if gotAccess != "44444444" || gotRefresh != "44444444" {
			t.Errorf(`Expecting string "44444444", but get accessToken = %v, refreshToken = %v`,
				gotAccess, gotRefresh)
		}
	})

	t.Run("TestValidToken", func(t *testing.T) {
		gotAccess, gotRefresh, err := GetTokenFromSessionID(testClient, "55555555")
		if err != nil {
			t.Errorf("Should return nil error, but got %v", err)
		}
		if gotAccess != "55555555" || gotRefresh != "55555555" {
			t.Errorf(`Expecting string "55555555", but get accessToken = %v, refreshToken = %v`,
				gotAccess, gotRefresh)
		}
	})
}

func TestGetTokenFromOpenFuyaoAuthHeader(t *testing.T) {
	tests := []struct {
		name       string
		authHeader string
		want       string
		wantErr    bool
	}{
		{
			name:       "TesFailToFetchAuthHeader",
			authHeader: "bad header",
			want:       "",
			wantErr:    true,
		},
		{
			name:       "TestAuthHeaderFetched",
			authHeader: "Bearer 12345678",
			want:       "12345678",
			wantErr:    false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &http.Request{
				Header: map[string][]string{},
			}
			req.Header.Set(constant.OpenFuyaoAuthHeader, tt.authHeader)
			got, err := GetTokenFromOpenFuyaoAuthHeader(req)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetTokenFromOpenFuyaoAuthHeader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("GetTokenFromOpenFuyaoAuthHeader() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDeleteSession(t *testing.T) {
	testClient := getTestGetSessionClient()

	t.Run("GetSymKeyFailed", func(t *testing.T) {
		err := DeleteSession(testClient, "11111111")
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	patchCrypto(t)

	t.Run("DeleteNonExistingSession", func(t *testing.T) {
		err := DeleteSession(testClient, "non-existing")
		if err == nil {
			t.Errorf("Should return error, but got nil")
		}
	})

	t.Run("DeleteNonExistingSession", func(t *testing.T) {
		err := DeleteSession(testClient, "33333333")
		if err != nil {
			t.Errorf("Should return nil error, but got %v", err)
		}
	})

}

func TestCheckExpiry(t *testing.T) {
	tests := []struct {
		name         string
		timestampStr string
		want         bool
	}{
		{
			"TestBadTimestamp",
			"notatimestamp",
			true,
		},
		{
			"TestExpiredTime",
			"1", // 1970/01/01 12:00 a.m.
			true,
		},
		{
			"TestNonExpiredTime",
			strconv.FormatInt(time.Now().Unix()+100, intBase10), // 10 seconds later than test time
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := checkExpiry(tt.timestampStr); got != tt.want {
				t.Errorf("checkExpiry() = %v, want %v", got, tt.want)
			}
		})
	}
}
