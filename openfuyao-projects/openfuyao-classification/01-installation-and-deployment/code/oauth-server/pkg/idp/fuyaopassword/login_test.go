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

package fuyaopassword

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"openfuyao/oauth-server/pkg/authenticators"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaostore"
	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/sessions"
)

// TestLoginHandlerGetSucceed tests the successful condition for getting login page
func TestLoginHandlerGetSucceed(t *testing.T) {
	req, err := http.NewRequest("GET", constants.FuyaoLoginEndpoint+"?then=/oauth2/oauth/authorize?test", nil)
	if err != nil {
		t.Fatal(err)
	}

	testLogin := mockLoginStruct()

	rr := httptest.NewRecorder()
	testLogin.LoginHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code %d; got %d", http.StatusOK, rr.Code)
	}
}

// TestLoginHandlerPostSucceed tests the successful condition for logging in
func TestLoginHandlerPostSucceed(t *testing.T) {
	reqBody := LoginRequest{
		Username: "admin",
		Password: []byte("Soup4@LL"),
		Then: "/oauth2/oauth/authorize?client_id=console&identity_provider=fuyaoPasswordProvider&redirect_uri" +
			"=https%3A%2F%2F192.168.100.48%3A31616%2Frest%2Fauth%2Fcallback&response_type=code&state=d7e6a4b3",
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	testLogin := mockLoginStruct()
	testLogin.LoginHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}
}

// TestLoginHandlerPostFail tests the failed condition for logging in
func TestLoginHandlerPostFail(t *testing.T) {
	then := "/oauth2/oauth/authorize?client_id=console&identity_provider=fuyaoPasswordProvider&redirect_uri=https%3A" +
		"%2F%2F192.168.100.48%3A31616%2Frest%2Fauth%2Fcallback&response_type=code&state=d7e6a4b3"
	reqBody := LoginRequest{
		Username: "admin",
		Password: []byte("Soup4@LL"),
		Then:     then,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", constants.FuyaoLoginEndpoint, bytes.NewReader(jsonBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	testLogin := mockLoginStruct()

	rr := httptest.NewRecorder()
	testLogin.LoginHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}

	if rr.Header().Get("Location")[:20] != "/oauth2/auth/login/fuyaoPasswordProvider"[:20] {
		t.Errorf("Expected Location %s; got %s", then, rr.Header().Get("Location"))
	}
}

// TestLoginHandlerUnknownMethod tests the successful condition for logging in
func TestLoginHandlerUnknownMethod(t *testing.T) {
	req, err := http.NewRequest("PATCH", constants.FuyaoLoginEndpoint, nil)
	if err != nil {
		t.Fatal(err)
	}

	testLogin := mockLoginStruct()

	rr := httptest.NewRecorder()
	testLogin.LoginHandler(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status code %d; got %d", http.StatusMethodNotAllowed, rr.Code)
	}
}

// TestLoginPasswordConfirmHandlerGetSucceed tests the successful condition for password confirmation
func TestLoginPasswordConfirmHandlerGetSucceed(t *testing.T) {
	req, err := http.NewRequest("GET", constants.FuyaoPasswordConfirmEndpoint+`?then=/oauth2/oauth/authorize?test`,
		nil)
	if err != nil {
		t.Fatal(err)
	}

	patches, err := createLoginTestPatches()
	if err != nil {
		t.Fatal(err)
	}
	defer patches.Reset()

	testLogin := mockLoginStruct()
	rr := httptest.NewRecorder()
	testLogin.PasswordConfirmHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code %d; got %d", http.StatusOK, rr.Code)
	}
}

// TestLoginPasswordConfirmHandlerPostSucceed tests the successful condition for password confirmation
func TestLoginPasswordConfirmHandlerPostSucceed(t *testing.T) {
	then := "/oauth2/oauth/authorize?client_id=console&identity_provider=fuyaoPasswordProvider&redir" +
		"ect_uri=https%3A%2F%2F192.168.100.48%3A31616%2Frest%2Fauth%2Fcallback&response_type=code&state=d7e6a4b3"

	reqBody := PasswordConfirmRequest{
		NewPassword: []byte("soup4@LL"),
		Then:        then,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("POST", constants.FuyaoPasswordConfirmEndpoint, bytes.NewReader(jsonBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	testLogin := mockLoginStruct()
	patches, err := createLoginTestPatches()
	if err != nil {
		t.Fatal(err)
	}
	defer patches.Reset()

	rr := httptest.NewRecorder()
	testLogin.PasswordConfirmHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}

	if rr.Header().Get("Set-Cookie") == "" {
		t.Errorf("Expected Set-Cookie key in header but it does show up")
	}

	if rr.Header().Get("Location") != then {
		t.Errorf("Expected Location %s; got %s", then, rr.Header().Get("Location"))
	}
}

func createLoginTestPatches() (*gomonkey.Patches, error) {
	patches := gomonkey.NewPatches()
	sessionMap := make(map[string][]string)
	sessionMap[constants.UserFirstLogin] = []string{"true"}
	jsonExtra, err := json.Marshal(sessionMap)
	if err != nil {
		return nil, err
	}
	patches.ApplyMethod(reflect.TypeOf(&sessions.CookieStore{}), "Get",
		func(_ *sessions.CookieStore, _ *http.Request) sessions.Values {
			return sessions.Values{
				constants.UserName:  "admin",
				constants.UserExtra: jsonExtra,
			}
		})
	patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, name string) (*fuyaouser.User, error) {
		return &fuyaouser.User{
			ObjectMeta: metav1.ObjectMeta{Name: "admin"},
			Spec: fuyaouser.UserSpec{
				Username:     "admin",
				PlatformRole: "platform-admin",
				Description:  "A platform user",
				FirstLogin:   true,
				EncryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3" +
					"zc4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
			},
			Status: fuyaouser.UserStatus{
				LockStatus:      "",
				LockedTimestamp: nil,
				RemainAttempts:  5,
			},
		}, nil
	})
	patches.ApplyFunc(fuyaouser.PatchUserInfo, func(_ dynamic.Interface, name string, data []byte) error {
		return nil
	})
	return patches, nil
}

func mockLoginStruct() *Login {
	fakeClient := fake.NewSimpleClientset()
	fakeTokenStore := fuyaostore.NewK8sSecretStore(fakeClient, "oauth-code-token")
	fakeAuthenticator := authenticators.NewFuyaoPasswordAuthenticator(fakeDynamicClient)
	fakeIdpLoginStore := sessions.NewSessionStore("idpLogin", loginStoreMaxAge, []byte("auth"),
		[]byte("encrypt123123123"))

	testLogin := &Login{
		Provider:       "fuyaoPasswordProvider",
		K8sClient:      fakeClient,
		TokenStore:     fakeTokenStore,
		Authenticator:  fakeAuthenticator,
		IdpLoginStore:  fakeIdpLoginStore,
		LoginProtector: fakeLoginProtector,
	}
	return testLogin
}

// TestLoginPasswordResetHandlerPostSucceed tests the successful condition for password reset
func TestLoginPasswordResetHandlerPostSucceed(t *testing.T) {
	// 构造请求体
	requestBody := PasswordResetRequest{
		Username:         "admin",
		OriginalPassword: []byte("Soup4@LL"),
		NewPassword:      []byte("soup4@LL"),
	}
	requestBodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		t.Fatalf("failed to marshal request body: %v", err)
	}
	req, err := http.NewRequest("POST", constants.FuyaoPasswordModifyEndpoint, bytes.NewBuffer(requestBodyBytes))
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	// 设置请求头
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-test")

	fakeClient := fake.NewSimpleClientset()
	// 定义自定义的反应函数，模拟 TokenReview 的 Create 方法
	fakeClient.Fake.PrependReactor("create", "tokenreviews",
		func(action k8stesting.Action) (bool, runtime.Object, error) {
			createAction, ok := action.(k8stesting.CreateAction)
			if !ok {
				t.Fatalf("failed to assert typer of action")
			}
			tokenReview, ok := createAction.GetObject().(*authenticationv1.TokenReview)
			if !ok {
				t.Fatalf("failed to assert typer of createAction")
			}
			// 模拟返回结果
			tokenReview.Status = authenticationv1.TokenReviewStatus{
				Authenticated: true,
				User: authenticationv1.UserInfo{
					Username: "admin",
					UID:      "12345",
				},
			}
			return true, tokenReview, nil
		})

	patches, err := createLoginTestPatches()
	if err != nil {
		t.Fatal(err)
	}
	defer patches.Reset()

	fakeTokenStore := fuyaostore.NewK8sSecretStore(fakeClient, "oauth-code-token")
	fakeAuthenticator := authenticators.NewFuyaoPasswordAuthenticator(fakeDynamicClient)
	fakeIdpLoginStore := sessions.NewSessionStore("idpLogin", loginStoreMaxAge,
		[]byte("auth"), []byte("encrypt123123123"))

	testLogin := &Login{
		Provider:       "fuyaoPasswordProvider",
		K8sClient:      fakeClient,
		TokenStore:     fakeTokenStore,
		Authenticator:  fakeAuthenticator,
		IdpLoginStore:  fakeIdpLoginStore,
		LoginProtector: fakeLoginProtector,
	}
	rr := httptest.NewRecorder()
	testLogin.PasswordResetHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}
}

// TestLoginPasswordConfirmHandlerRevertSucceed tests the reverting condition for password confirmation
func TestLoginPasswordConfirmHandlerRevertSucceed(t *testing.T) {
	req, err := http.NewRequest("DELETE", constants.FuyaoPasswordConfirmEndpoint+`?then=/`, nil)
	if err != nil {
		t.Fatal(err)
	}

	testLogin := mockLoginStruct()

	rr := httptest.NewRecorder()
	testLogin.PasswordConfirmHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}
}
