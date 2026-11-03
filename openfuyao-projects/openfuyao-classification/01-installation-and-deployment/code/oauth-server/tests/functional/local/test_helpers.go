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

package local

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/gorilla/mux"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	dfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/authenticators"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaostore"
	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/idp/fuyaopassword"
	"openfuyao/oauth-server/pkg/oauth2"
	"openfuyao/oauth-server/pkg/protector"
	"openfuyao/oauth-server/pkg/sessions"
)

// Test configuration constants
const (
	TestFailTimes        = 5
	TestFailMinute       = 5
	TestLockMinute       = 20
	TestLoginStoreMaxAge = 300
	TestValidToken       = "valid-access-token-for-testing"
	TestNamespace        = "oauth-code-token"
)

// TestUser represents a test user configuration
type TestUser struct {
	Username          string
	Password          string
	EncryptedPassword []byte
	FirstLogin        bool
	LockStatus        string
	LockedTimestamp   *metav1.Time
	RemainAttempts    int
}

// TestContext holds all the shared test context
type TestContext struct {
	// HTTP test components
	Router   *mux.Router
	Recorder *httptest.ResponseRecorder
	Request  *http.Request

	// Handlers
	LoginHandler *fuyaopassword.Login
	OAuthServer  *oauth2.FuyaoAuthorizeServer

	// Mock clients
	FakeClient        *fake.Clientset
	FakeDynamicClient *dfake.FakeDynamicClient

	// Protector
	LoginProtector *protector.LoginUserProtector

	// Stores
	TokenStore    *fuyaostore.K8sSecretStore
	IdpLoginStore *sessions.CookieStore

	// Test data
	TestUser       *TestUser
	FirstLoginUser *TestUser

	// Patches for gomonkey
	Patches *gomonkey.Patches

	// Session cookie for testing (created via Put, read via Get)
	SessionCookie *http.Cookie

	// Test URLs
	ValidThenURL     string
	ValidRedirectURI string
}

// NewTestContext creates a new test context with initialized components
func NewTestContext() *TestContext {
	scheme := runtime.NewScheme()
	fakeDynamicClient := dfake.NewSimpleDynamicClient(scheme)
	fakeClient := fake.NewSimpleClientset()

	// Create login protector
	loginProtector := protector.NewLoginUserProtector(fakeDynamicClient, &config.IPProtectorConfig{
		FailTimes:    TestFailTimes,
		FailDuration: time.Minute * TestFailMinute,
		LockDuration: time.Minute * TestLockMinute,
	})

	// Create stores
	tokenStore := fuyaostore.NewK8sSecretStore(fakeClient, TestNamespace)
	// Note: signing key should be 64 bytes, encryption key should be 32 bytes (for AES-256)
	idpLoginStore := sessions.NewSessionStore("idpLogin", TestLoginStoreMaxAge,
		[]byte("signing-key-12345678901234567890123456789012345678901234567890123456"),
		[]byte("encrypt-key-12345678901234567890"))

	return &TestContext{
		Router:            mux.NewRouter(),
		Recorder:          httptest.NewRecorder(),
		FakeClient:        fakeClient,
		FakeDynamicClient: fakeDynamicClient,
		LoginProtector:    loginProtector,
		TokenStore:        tokenStore,
		IdpLoginStore:     idpLoginStore,
		TestUser:          CreateTestUser(false),
		FirstLoginUser:    CreateTestUser(true),
		ValidThenURL:      "/oauth2/oauth/authorize?client_id=console&identity_provider=fuyaoPasswordProvider&redirect_uri=https%3A%2F%2Fexample.com%2Frest%2Fauth%2Fcallback&response_type=code&state=test123",
		ValidRedirectURI:  "https://example.com/rest/auth/callback",
	}
}

// CreateTestUser creates a test user with specified firstLogin status
func CreateTestUser(firstLogin bool) *TestUser {
	return &TestUser{
		Username: "admin",
		Password: "Soup4@LL",
		// PBKDF2 encrypted password for "Soup4@LL"
		EncryptedPassword: []byte("IFp9vTCHQ5v0qgLrFNsD5oqNG7TS4LCs0P5IWRrAlYfFeZSk9xVm0KxRi4pOsOECvNaw3" +
			"zc4JXvEr4j4ldxlf541zErHyqRHE+I2ik7ww5M="),
		FirstLogin:     firstLogin,
		LockStatus:     "",
		RemainAttempts: TestFailTimes,
	}
}

// CreateMockUserInfo creates a mock fuyaouser.User from TestUser
func CreateMockUserInfo(testUser *TestUser) *fuyaouser.User {
	return &fuyaouser.User{
		ObjectMeta: metav1.ObjectMeta{
			Name: testUser.Username,
		},
		Spec: fuyaouser.UserSpec{
			Username:          testUser.Username,
			PlatformRole:      "platform-admin",
			Description:       "A test user",
			FirstLogin:        testUser.FirstLogin,
			EncryptedPassword: testUser.EncryptedPassword,
		},
		Status: fuyaouser.UserStatus{
			LockStatus:      testUser.LockStatus,
			LockedTimestamp: testUser.LockedTimestamp,
			RemainAttempts:  testUser.RemainAttempts,
		},
	}
}

// SetupLoginHandler creates and configures a Login handler for testing
func (ctx *TestContext) SetupLoginHandler(mockTokenReview bool) {
	if mockTokenReview {
		ctx.FakeClient.Fake.PrependReactor("create", "tokenreviews",
			func(action k8stesting.Action) (bool, runtime.Object, error) {
				createAction, ok := action.(k8stesting.CreateAction)
				if !ok {
					return false, nil, nil
				}
				tokenReview, ok := createAction.GetObject().(*authenticationv1.TokenReview)
				if !ok {
					return false, nil, nil
				}
				tokenReview.Status = authenticationv1.TokenReviewStatus{
					Authenticated: true,
					User: authenticationv1.UserInfo{
						Username: "admin",
						UID:      "12345",
					},
				}
				return true, tokenReview, nil
			})
	}

	// Create a new Login handler using the existing fields
	// Note: We create it as a pointer struct to allow setting exported fields
	ctx.LoginHandler = &fuyaopassword.Login{
		Provider:       constants.FuyaoIdpProvider,
		K8sClient:      ctx.FakeClient,
		TokenStore:     ctx.TokenStore,
		Authenticator:  authenticators.NewFuyaoPasswordAuthenticator(ctx.FakeDynamicClient),
		IdpLoginStore:  ctx.IdpLoginStore,
		LoginProtector: ctx.LoginProtector,
	}
}

// SetupLoginHandlerWithExpiredToken creates Login handler with expired token reactor
func (ctx *TestContext) SetupLoginHandlerWithExpiredToken() {
	// Setup reactor to simulate expired/invalid token by returning an error from TokenReview API
	// This simulates what happens when K8s API server rejects an expired token
	ctx.FakeClient.Fake.PrependReactor("create", "tokenreviews",
		func(action k8stesting.Action) (bool, runtime.Object, error) {
			// Return an error to simulate token validation failure at K8s API level
			// This triggers the err != nil path in authenticateByWebhook
			return true, nil, fmt.Errorf("Unauthorized: token is expired or invalid")
		})

	ctx.LoginHandler = &fuyaopassword.Login{
		Provider:       constants.FuyaoIdpProvider,
		K8sClient:      ctx.FakeClient,
		TokenStore:     ctx.TokenStore,
		Authenticator:  authenticators.NewFuyaoPasswordAuthenticator(ctx.FakeDynamicClient),
		IdpLoginStore:  ctx.IdpLoginStore,
		LoginProtector: ctx.LoginProtector,
	}
}

// SetupOAuthServer creates and configures an OAuth server for testing
func (ctx *TestContext) SetupOAuthServer() {
	oauthConfig := &config.OAuthServerConfig{
		CodeTokenNamespace: TestNamespace,
		AuthCodeExp:        time.Minute * 5,
		AccessTokenExp:     time.Hour * 2,
		RefreshTokenExp:    time.Hour * 24,
		IsGenerateRefresh:  true,
		JWTKeyID:           "test_key_id",
		JWTPrivateKey:      []byte("test-jwt-private-key-for-testing"),
		ClientMapper: map[string]string{
			"console": "console-password",
		},
	}

	ctx.OAuthServer = oauth2.NewOAuthServer(
		ctx.IdpLoginStore,
		ctx.TokenStore,
		oauthConfig,
		"csrf",
	)
}

// SetupUserInfoMock sets up gomonkey patches for user info functions
func (ctx *TestContext) SetupUserInfoMock(mockUser *fuyaouser.User) {
	ctx.Patches = gomonkey.NewPatches()

	// Mock GetUserInfo
	ctx.Patches.ApplyFunc(fuyaouser.GetUserInfo, func(_ dynamic.Interface, name string) (*fuyaouser.User, error) {
		if name == mockUser.Spec.Username {
			return mockUser, nil
		}
		return nil, fmt.Errorf("user %s not found", name)
	})

	// Mock PatchUserInfo
	ctx.Patches.ApplyFunc(fuyaouser.PatchUserInfo, func(_ dynamic.Interface, name string, data []byte) error {
		return nil
	})

	// Mock UpdateUserInfo
	ctx.Patches.ApplyFunc(fuyaouser.UpdateUserInfo, func(_ dynamic.Interface, userCR *fuyaouser.User) error {
		return nil
	})
}

// CreateSessionCookie creates a real session cookie by calling Put
// This allows testing the full session flow without mocking internal logic
func (ctx *TestContext) CreateSessionCookie(sessionValues sessions.Values) *http.Cookie {
	// Create a response recorder to capture the cookie
	recorder := httptest.NewRecorder()

	// Use Put to write the session cookie
	err := ctx.IdpLoginStore.Put(recorder, sessionValues)
	if err != nil {
		return nil
	}

	// Extract the cookie from the response
	cookies := recorder.Result().Cookies()
	for _, cookie := range cookies {
		if cookie.Name == "idpLogin" {
			return cookie
		}
	}
	return nil
}

// AddSessionCookieToRequest adds the session cookie to an HTTP request
func (ctx *TestContext) AddSessionCookieToRequest(req *http.Request, cookie *http.Cookie) {
	if cookie != nil {
		req.AddCookie(cookie)
	}
}

// SetupSession creates a session cookie and stores it in context for reuse
// Returns the cookie for adding to requests
func (ctx *TestContext) SetupSession(sessionValues sessions.Values) *http.Cookie {
	ctx.SessionCookie = ctx.CreateSessionCookie(sessionValues)
	return ctx.SessionCookie
}

// NewRequestWithSession creates a new HTTP request with the stored session cookie
func (ctx *TestContext) NewRequestWithSession(method, url string, body *bytes.Reader) (*http.Request, error) {
	if ctx == nil {
		return nil, fmt.Errorf("test context is nil")
	}
	var req *http.Request
	var err error
	if body != nil {
		req, err = http.NewRequest(method, url, body)
	} else {
		req, err = http.NewRequest(method, url, nil)
	}
	if err != nil {
		return nil, err
	}
	if req == nil {
		return nil, fmt.Errorf("failed to create request")
	}
	if ctx.SessionCookie != nil {
		req.AddCookie(ctx.SessionCookie)
	}
	return req, nil
}

// ClearSession clears the session cookie from context
func (ctx *TestContext) ClearSession() {
	ctx.SessionCookie = nil
}

// ResetPatches resets all gomonkey patches
func (ctx *TestContext) ResetPatches() {
	if ctx.Patches != nil {
		ctx.Patches.Reset()
		ctx.Patches = nil
	}
}

// ResetRecorder resets the response recorder
func (ctx *TestContext) ResetRecorder() {
	ctx.Recorder = httptest.NewRecorder()
}

// ResetProtector resets the login protector state
func (ctx *TestContext) ResetProtector() {
	scheme := runtime.NewScheme()
	ctx.FakeDynamicClient = dfake.NewSimpleDynamicClient(scheme)
	ctx.LoginProtector = protector.NewLoginUserProtector(ctx.FakeDynamicClient, &config.IPProtectorConfig{
		FailTimes:    TestFailTimes,
		FailDuration: time.Minute * TestFailMinute,
		LockDuration: time.Minute * TestLockMinute,
	})
}

// CreateLoggedInSession creates a session value representing a logged-in user
func CreateLoggedInSession(username string, firstLogin bool) sessions.Values {
	sessionValues := sessions.Values{
		constants.UserName:   username,
		constants.UserUID:    "test-uid-12345",
		constants.UserGroups: []string{"system:authenticated"},
	}

	extra := make(map[string][]string)
	if firstLogin {
		extra[constants.UserFirstLogin] = []string{"true"}
	} else {
		extra[constants.UserFirstLogin] = []string{"false"}
	}
	extra[constants.OAuthServerSessionID] = []string{"test-session-id"}

	jsonExtra, err := json.Marshal(extra)
	if err != nil {
		fmt.Print("this should never happen")
	}
	sessionValues[constants.UserExtra] = jsonExtra

	return sessionValues
}

// CreateBrokenSession creates an incomplete session with missing required fields
func CreateBrokenSession() sessions.Values {
	return sessions.Values{
		constants.UserName: "admin",
		// Missing UserUID, UserGroups, UserExtra
	}
}

// ExtractErrorFromLocation extracts the error parameter from a redirect location URL
func ExtractErrorFromLocation(location string) string {
	u, err := url.Parse(location)
	if err != nil {
		return ""
	}
	return u.Query().Get("error")
}

// ExtractCodeFromLocation extracts the code parameter from a redirect location URL
func ExtractCodeFromLocation(location string) string {
	u, err := url.Parse(location)
	if err != nil {
		return ""
	}
	return u.Query().Get("code")
}
