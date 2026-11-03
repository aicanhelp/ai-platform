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

package oauth2

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"golang.org/x/oauth2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaostore"
	"openfuyao/oauth-server/pkg/sessions"
)

func createSessionMockPatches(firstLogin string) (*gomonkey.Patches, error) {
	patches := gomonkey.NewPatches()
	sessionMap := make(map[string][]string)
	sessionMap[constants.UserFirstLogin] = []string{firstLogin}
	jsonExtra, err := json.Marshal(sessionMap)
	if err != nil {
		return nil, err
	}
	patches.ApplyMethod(reflect.TypeOf(&sessions.CookieStore{}), "Get",
		func(_ *sessions.CookieStore, _ *http.Request) sessions.Values {
			return sessions.Values{
				constants.UserName:   "admin",
				constants.UserExtra:  jsonExtra,
				constants.UserUID:    "test",
				constants.UserGroups: []string{"system:authenticated"},
			}
		})
	return patches, nil
}

// TestFuyaoAuthorizeServerOAuthAuthorizeHandlerSucceed test http handler for /oauth/authorize
func TestFuyaoAuthorizeServerOAuthAuthorizeHandlerSucceed(t *testing.T) {
	// prepare query parameters
	redirectUri := "http://example.test.com/rest/auth/callback"
	req := createOAuthAuthorizeTestRequest(t, redirectUri)

	patches, err := createSessionMockPatches("false")
	if err != nil {
		t.Fatal(err)
	}
	defer patches.Reset()

	testFuyaoOAuthServer, _, _ := createTestFuyaoOAuthServer(nil)

	rr := httptest.NewRecorder()
	testFuyaoOAuthServer.OAuthAuthorizeHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}

	location := redirectUri + "?code="
	if !strings.HasPrefix(rr.Header().Get("Location"), location) {
		t.Errorf("Expected location prefix %s; get %s", location, rr.Header().Get("Location"))
	}
}

// TestFuyaoAuthorizeServerOAuthAuthorizeHandlerNoSession test http handler for /oauth/authorize
func TestFuyaoAuthorizeServerOAuthAuthorizeHandlerNoSession(t *testing.T) {
	// prepare query parameters
	redirectUri := "http://example.test.com/rest/auth/callback"
	req := createOAuthAuthorizeTestRequest(t, redirectUri)

	testFuyaoOAuthServer, _, _ := createTestFuyaoOAuthServer(nil)

	rr := httptest.NewRecorder()
	testFuyaoOAuthServer.OAuthAuthorizeHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}

	cookieSet := rr.Header().Get("Set-Cookie")
	if !strings.HasPrefix(cookieSet, "idpLogin=") {
		t.Errorf("Expected cookie set but it didn't")
	}

	location := constants.FuyaoLoginEndpoint + "?then=" + url.QueryEscape(req.URL.String())
	if rr.Header().Get("Location") != location {
		t.Errorf("Expected locatin %s; get %s", location, rr.Header().Get("Location"))
	}
}

// TestFuyaoAuthorizeServerOAuthAuthorizeHandlerFirstLogin test http handler for /oauth/authorize
func TestFuyaoAuthorizeServerOAuthAuthorizeHandlerFirstLogin(t *testing.T) {
	// prepare query parameters
	redirectUri := "http://example.test.com/rest/auth/callback"
	req := createOAuthAuthorizeTestRequest(t, redirectUri)

	testFuyaoOAuthServer, _, _ := createTestFuyaoOAuthServer(nil)

	rr := httptest.NewRecorder()
	testFuyaoOAuthServer.OAuthAuthorizeHandler(rr, req)

	if rr.Code != http.StatusFound {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}

	location := constants.FuyaoLoginEndpoint + "?then=" + url.QueryEscape(req.URL.String())
	if rr.Header().Get("Location") != location {
		t.Errorf("Expected location %s; get %s", location, rr.Header().Get("Location"))
	}
}

// TestFuyaoAuthorizeServerOAuthTokenHandlerCodeExpired test http handler for /oauth/token
func TestFuyaoAuthorizeServerOAuthTokenHandlerCodeExpired(t *testing.T) {
	// with correct code
	// prepare form parameters
	testCode := "zjhjoddimwutyzjkni0zzwe3lwfhngutothkmwy0ngzjnmjj"
	req := createOAuthTokenTestRequest(t, testCode)

	// prepare authcode secret
	testCodeSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constants.CodePrefix + testCode,
			Namespace: "oauth-code-token",
		},
		Data: map[string][]byte{
			"userinfo": []byte(`{"ClientID":"oauth-proxy","UserID":"admin","RedirectURI":"https://example.test.com` +
				`/oauth/callback","Scope":"user:info user:check-access","Code":"zjhjoddimwutyzjkni0zzwe3lwfhngut` +
				`othkmwy0ngzjnmjj","CodeChallenge":"","CodeChallengeMethod":"","CodeCreateAt":"2024-05-27T10:34:51.` +
				`973738633+08:00","CodeExpiresIn":300000000000,"Access":"","AccessCreateAt":"0001-01-01T00:00:00` +
				`Z","AccessExpiresIn":0,"Refresh":"","RefreshCreateAt":"0001-01-01T00:00:00Z","RefreshExpiresIn":0}`),
		},
	}
	testFuyaoOAuthServer, _, _ := createTestFuyaoOAuthServer(testCodeSecret)

	// run test
	rr := httptest.NewRecorder()
	testFuyaoOAuthServer.OAuthTokenHandler(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("Expected status code %d; got %d", http.StatusUnauthorized, rr.Code)
	}

	body, err := ioutil.ReadAll(rr.Body)
	if err != nil {
		t.Fatal(err)
	}
	bodyStr := string(body)
	expectedBodyStr := `{"error":"invalid_grant","error_description":"The provided authorization grant (e.g., autho` +
		`rization code, resource owner credentials) or refresh token is invalid, expired, revoked, does not match` +
		` the redirection URI used in the authorization request, or was issued to another client"}` + "\n"
	if bodyStr != expectedBodyStr {
		t.Errorf("Expected return body %s; got %s", expectedBodyStr, bodyStr)
	}
}

// TestFuyaoAuthorizeServerOAuthTokenHandlerSucceed test http handler for /oauth/token
func TestFuyaoAuthorizeServerOAuthTokenHandlerSucceed(t *testing.T) {
	// prepare form parameters
	testCode := "oda1mthhmtutmgfknc0zyjdllwe5ztetzmi5mwm0owuyzjbm"
	req := createOAuthTokenTestRequest(t, testCode)

	// prepare authcode secret
	testCodeSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constants.CodePrefix + testCode,
			Namespace: "oauth-code-token",
		},
		Data: map[string][]byte{
			"userinfo": []byte(`{"ClientID":"oauth-proxy","UserID":"admin","RedirectURI":"https://` +
				`example.test.com/oauth/callback","Scope":"user:info user:check-access","Code":"oda1mthhmt` +
				`utmgfknc0zyjdllwe5ztetzmi5mwm0owuyzjbm","CodeChallenge":"","CodeChallengeMethod":"","CodeCrea` +
				`teAt":"2024-05-27T11:29:35.378162227+08:00","CodeExpiresIn":315360000000000000,"Access":"","A` +
				`ccessCreateAt":"0001-01-01T00:00:00Z","AccessExpiresIn":0,"Refresh":"","RefreshCreateAt":"0001` +
				`-01-01T00:00:00Z","RefreshExpiresIn":0}`),
		},
	}
	testFuyaoOAuthServer, _, _ := createTestFuyaoOAuthServer(testCodeSecret)

	// run test
	rr := httptest.NewRecorder()
	testFuyaoOAuthServer.OAuthTokenHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status code %d; got %d", http.StatusOK, rr.Code)
	}

	body, err := io.ReadAll(rr.Body)
	if err != nil {
		t.Fatal(err)
	}
	var returnedToken oauth2.Token
	if err = json.Unmarshal(body, &returnedToken); err != nil {
		t.Fatal(err)
	}

	if returnedToken.AccessToken == "" || returnedToken.TokenType != "Bearer" {
		t.Errorf("Expected return token is invalid")
	}
}

func createOAuthTokenTestRequest(t *testing.T, code string) *http.Request {
	form := url.Values{}
	form.Add("code", code)
	form.Add("grant_type", "authorization_code")
	form.Add("logout_endpoint", "https://example.test.com/oauth/logout")
	form.Add("redirect_uri", "https://example.test.com/oauth/callback")
	form.Add("session_id", "testsessionidtryme")
	form.Add("client_id", "oauth-proxy")
	form.Add("client_secret", "SECRETTS")

	// prepare request
	req, err := http.NewRequest("POST", constants.FuyaoOAuthTokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return req
}

// TestFuyaoAuthorizeServerSingleLogoutHandlerSucceed test http handler for /auth/logout
func TestFuyaoAuthorizeServerSingleLogoutHandlerSucceed(t *testing.T) {
	// prepare form parameters
	redirectUri := "https://example.test.com/rest/auth/login"
	query := url.Values{}
	query.Add("redirect_uri", redirectUri)

	// prepare request
	req, err := http.NewRequest("POST", constants.FuyaoLogoutEndpoint, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.URL.RawQuery = query.Encode()

	patches, err := createSessionMockPatches("false")
	if err != nil {
		t.Fatal(err)
	}
	defer patches.Reset()

	testFuyaoOAuthServer, _, _ := createTestFuyaoOAuthServer(nil)

	// run test
	rr := httptest.NewRecorder()
	testFuyaoOAuthServer.SingleLogoutHandler(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Errorf("Expected status code %d; got %d", http.StatusFound, rr.Code)
	}
}

// TestNewOAuthServer test initializing OAuthServer
func TestNewOAuthServer(t *testing.T) {
	type args struct {
		idpLoginStore *sessions.CookieStore
		tokenStore    *fuyaostore.K8sSecretStore
		cfg           *config.OAuthServerConfig
		csrf          string
	}

	tgt, fakeTokenStore, cfg := createTestFuyaoOAuthServer(nil)

	tests := []struct {
		name string
		args args
		want *FuyaoAuthorizeServer
	}{
		{
			"successfully initialized",
			args{
				idpLoginStore: tgt.idpLoginStore,
				tokenStore:    fakeTokenStore,
				cfg:           cfg,
				csrf:          "csrf",
			},
			tgt,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewOAuthServer(tt.args.idpLoginStore, tt.args.tokenStore, tt.args.cfg,
				tt.args.csrf); !reflect.DeepEqual(
				got.Config.TokenType, tt.want.Config.TokenType) {
				t.Errorf("NewOAuthServer() = %v, want %v", got, tt.want)
			}
		})
	}
}

func createOAuthAuthorizeTestRequest(t *testing.T, rediretUri string) *http.Request {
	query := url.Values{}
	query.Add("client_id", "console")
	query.Add("identity_provider", "fuyaoPasswordProvider")
	query.Add("redirect_uri", rediretUri)
	query.Add("response_type", "code")
	query.Add("state", "10a4d3a9")
	req, err := http.NewRequest("GET", constants.FuyaoOAuthAuthorizeEndpoint, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.URL.RawQuery = query.Encode()
	return req
}

func createTestFuyaoOAuthServer(obj runtime.Object) (*FuyaoAuthorizeServer, *fuyaostore.K8sSecretStore,
	*config.OAuthServerConfig) {
	fakeClient := fake.NewSimpleClientset()
	if obj != nil {
		fakeClient = fake.NewSimpleClientset(obj)
	}
	const authCodeExp = 8760
	const accessTokenExp = 2
	const refreshTokenExp = 2
	const maxAge = 300
	fakeTokenStore := fuyaostore.NewK8sSecretStore(fakeClient, "oauth-code-token")
	fakeIdpLoginStore := sessions.NewSessionStore("idpLogin", maxAge, []byte("auth"), []byte("encrypt123123123"))
	cfg := &config.OAuthServerConfig{
		CodeTokenNamespace: "oauth-code-token",
		AuthCodeExp:        time.Hour * authCodeExp,
		AccessTokenExp:     time.Hour * accessTokenExp,
		RefreshTokenExp:    time.Hour * refreshTokenExp,
		IsGenerateRefresh:  false,
		JWTKeyID:           "access_token_sign_key",
		JWTPrivateKey:      []byte("i_am_the_secrets"),
		ClientMapper: map[string]string{
			"console":     "console-password",
			"oauth-proxy": "SECRETTS",
		},
	}
	testFuyaoOAuthServer := NewOAuthServer(fakeIdpLoginStore, fakeTokenStore, cfg, "csrf")
	return testFuyaoOAuthServer, fakeTokenStore, cfg
}
