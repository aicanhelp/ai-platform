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
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	"golang.org/x/oauth2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	"console-service/pkg/constant"
)

const (
	loginEndpoint         = "http://example.com/rest/auth/login"
	loginCallbackEndpoint = "http://example.com/rest/auth/callback"
	oauth2RedirectURL     = "https://example.com/oauth2/oauth/authorize"
)

// oAuth2Mock mocks oauth2.config for test
type oAuth2Mock struct{}

func (o *oAuth2Mock) AuthCodeURL(state string, opts ...oauth2.AuthCodeOption) string {
	return oauth2RedirectURL
}

// Exchange takes the code and returns a token.
func (o *oAuth2Mock) Exchange(ctx context.Context, code string, opts ...oauth2.AuthCodeOption) (*oauth2.Token, error) {
	if code == "error-auth-code" {
		return nil, errors.New("auth code error")
	}
	// Return token with refresh_token_expires_in for successful exchange
	// The value is a float64 (seconds) as expected by retrieveRefreshTokenExpiry
	token := &oauth2.Token{
		AccessToken:  "token-for-123456789",
		Expiry:       time.Unix(1, 1), // Keep original expiry for existing tests
		RefreshToken: "refresh-token-for-123456789",
	}
	// Add extra field for refresh_token_expires_in
	token = token.WithExtra(map[string]interface{}{
		"refresh_token_expires_in": float64(7200),
	})
	return token, nil
}

type mockTokenSource struct {
	t *oauth2.Token
}

func (s *mockTokenSource) Token() (*oauth2.Token, error) {
	return s.t, nil
}

func (o *oAuth2Mock) TokenSource(ctx context.Context, t *oauth2.Token) oauth2.TokenSource {
	return &mockTokenSource{
		t: t,
	}
}

func newFakeClientSet() *fake.Clientset {
	return fake.NewSimpleClientset(
		&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "session-123456789",
				Namespace: "session-secret",
			},
			Data: map[string][]byte{
				"Expiry":      []byte("0"),
				"SessionID":   []byte("123456789"),
				"AccessToken": []byte("token-for-123456789"),
			},
			Type: corev1.SecretTypeOpaque,
		},
		&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "session-777777777",
				Namespace: "session-secret",
			},
			Data: map[string][]byte{
				"Expiry":      []byte(strconv.FormatInt(time.Now().Unix()+100, intBase10)),
				"SessionID":   []byte("777777777"),
				"AccessToken": []byte("token-for-777777777"),
			},
			Type: corev1.SecretTypeOpaque,
		},
	)
}

func newFakeClientSetWithTokenKey() *fake.Clientset {
	return fake.NewSimpleClientset(
		&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "console-service-token-key",
				Namespace: "openfuyao-system",
			},
			Data: map[string][]byte{
				"console-service-symmetric-key": []byte("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
			},
			Type: corev1.SecretTypeOpaque,
		},
	)
}

func TestNewHandler(t *testing.T) {
	err := os.Setenv(clientIDEnv, "client-id")
	if err != nil {
		return
	}
	err = os.Setenv(clientSecretEnv, "client-secret")
	if err != nil {
		return
	}
	defer os.Unsetenv(clientIDEnv)
	defer os.Unsetenv(clientSecretEnv)

	_, err = NewHandler(&rest.Config{})
	if err != nil {
		t.Errorf("NewHandler() should return nil error, got %v", err)
		return
	}
}

func newTestAuthHandler() Handler {
	return Handler{
		kubeConfig:  &rest.Config{},
		clientset:   newFakeClientSet(),
		oauthConfig: &oAuth2Mock{},
	}
}

// login tests
func TestHandlerLoginHandler(t *testing.T) {
	h := newTestAuthHandler()
	tests := []struct {
		name              string
		sessionID         string
		wantSetCookie     bool
		wantSetCookieName string
	}{
		{
			name:          "TestWithValidSession",
			sessionID:     "777777777",
			wantSetCookie: false,
		},
		{
			name:              "TestWithInvalidSession",
			sessionID:         "invalid-sessionID",
			wantSetCookie:     true,
			wantSetCookieName: "sessionID",
		},
		{
			name:              "TestLogin",
			wantSetCookie:     true,
			wantSetCookieName: "state",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := restful.NewRequest(httptest.NewRequest("GET", loginEndpoint, nil))
			if tt.sessionID != "" {
				req.Request.AddCookie(&http.Cookie{
					Name:  "sessionID",
					Value: tt.sessionID,
				})
			}
			recorder := httptest.NewRecorder()
			resp := restful.NewResponse(recorder)
			h.loginHandler(req, resp)
			if recorder.Code != http.StatusFound {
				t.Errorf("loginHandler() code = %v, want %v", recorder.Code, http.StatusFound)
			}
			if val := recorder.Header().Get("Set-Cookie"); tt.wantSetCookie !=
				strings.HasPrefix(val, tt.wantSetCookieName+"=") {
				t.Errorf("loginHandler() set cookie expect: %v, target cookie: %s, actual Set-Cookie: %s",
					tt.wantSetCookie, tt.wantSetCookieName, val)
			}
			if val := recorder.Header().Get("Location"); !strings.HasPrefix(val, oauth2RedirectURL) {
				t.Errorf("loginHandler() redirect to wrong location: %s, should start with %s", val, oauth2RedirectURL)
			}
		})
	}
}

func TestHandlerCheckSessionCookie(t *testing.T) {
	tests := []struct {
		name   string
		cookie *http.Cookie
		want   bool
	}{
		{
			"TestNonExistingSessionID",
			&http.Cookie{
				Name:  "sessionID",
				Value: "non-existing-sessionID",
			},
			false,
		},
		{
			"TestExistingSessionID",
			&http.Cookie{
				Name:  "sessionID",
				Value: "777777777",
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newTestAuthHandler()
			if got := h.checkSessionCookie(tt.cookie); got != tt.want {
				t.Errorf("checkSessionCookie() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCreateLoginState(t *testing.T) {
	double := 2
	tests := []struct {
		name    string
		length  int
		wantErr bool
	}{
		{
			"TestLoginStateLength",
			1,
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := createLoginState(tt.length)
			if (err != nil) != tt.wantErr {
				t.Errorf("createLoginState() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			// login state is random, only checking length, StringLength = 2*ByteLength
			if len(got) != double*tt.length {
				t.Errorf("createLoginState() got = %v, want length %v", got, double*tt.length)
			}
		})
	}
}

// callback tests
func generateMockRequest(method string, target string) *restful.Request {
	return restful.NewRequest(
		httptest.NewRequest(method, target, nil),
	)
}

func TestHandlerCallbackHandlerFailure(t *testing.T) {
	handler := newTestAuthHandler()
	tests := []struct {
		name           string
		reqQuery       string
		state          string
		wantHttpStatus int
		wantHTML       string
	}{
		{
			name:           "TestWithError",
			reqQuery:       "error=TEST_ERROR&error_description=This%20is%20a%20test%20error",
			wantHttpStatus: http.StatusOK,
			wantHTML:       fmt.Sprintf(responseTemplate, "zh-CN", "TEST_ERROR", "This is a test error"),
		},
		{
			name:           "TestParseQueryFailure",
			reqQuery:       "state=123456&other=values",
			wantHttpStatus: http.StatusUnauthorized,
		},
		{
			name:           "TestCheckLoginStateFailure",
			reqQuery:       "state=123456&code=acdefg",
			wantHttpStatus: http.StatusSeeOther,
		},
	}

	for _, tt := range tests {
		req := generateMockRequest("GET", loginCallbackEndpoint+"?"+tt.reqQuery)
		if tt.state != "" {
			req.Request.AddCookie(&http.Cookie{Name: "state", Value: "123456"})
		}
		recorder := httptest.NewRecorder()
		resp := restful.NewResponse(recorder)
		handler.callbackHandler(req, resp)
		if recorder.Code != tt.wantHttpStatus {
			t.Errorf("callbackHandler() code = %v, want %v", recorder.Code, tt.wantHttpStatus)
		}
		if tt.wantHTML != "" {
			got := removeEmptyCharHTML(recorder.Body.String())
			want := removeEmptyCharHTML(tt.wantHTML)
			if got != want {
				t.Errorf("callbackHandler() got = %v, want %v", got, want)
			}
		}
	}
}

func createReqForErropRespTest(reqQuery string) *restful.Request {
	return &restful.Request{
		Request: &http.Request{
			URL: &url.URL{RawQuery: reqQuery},
		},
	}
}

func TestGenerateErrorRespText(t *testing.T) {
	tests := []struct {
		name     string
		reqQuery string
		want     string
		wantErr  bool
	}{
		{
			"TestNoError",
			"code=123&state=456",
			"",
			false,
		},
		{
			"TestWithErrorCode",
			"error=TEST_ERROR&other=value",
			fmt.Sprintf(responseTemplate, "en-US", "TEST_ERROR", ""),
			true,
		},
		{
			"TestWithErrorDesc",
			"error_description=This is a test error&other=value",
			fmt.Sprintf(responseTemplate, "en-US", "", "This is a test error"),
			true,
		},
		{
			"TestWithErrorCodeAndDesc",
			"error=TEST_ERROR&error_description=This is a test error&other=value",
			fmt.Sprintf(responseTemplate, "en-US", "TEST_ERROR", "This is a test error"),
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := createReqForErropRespTest(tt.reqQuery)
			got, err := generateErrorRespText(req, "en-US")
			if (err != nil) != tt.wantErr {
				t.Errorf("generateErrorRespText() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			// remove \n & \t characters, which don't matter in html
			got = removeEmptyCharHTML(got)
			tt.want = removeEmptyCharHTML(tt.want)
			if got != tt.want {
				t.Errorf("generateErrorRespText() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func createTestReqForParseCallbackQuery(query string) *restful.Request {
	return restful.NewRequest(
		httptest.NewRequest("GET", loginCallbackEndpoint+"?"+query, nil),
	)
}

func TestParseCallbackQuery(t *testing.T) {
	tests := []struct {
		name           string
		reqQuery       string
		wantAuthCode   string
		wantLoginState string
		wantErr        bool
	}{
		{
			name:     "TesNoAuthCode",
			reqQuery: "state=123456",
			wantErr:  true,
		},
		{
			name:     "TestNoAuthCode",
			reqQuery: "code=qwerty",
			wantErr:  true,
		},
		{
			name:           "TestCorrectQuery",
			reqQuery:       "state=123456&code=qwerty",
			wantAuthCode:   "qwerty",
			wantLoginState: "123456",
			wantErr:        false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := createTestReqForParseCallbackQuery(tt.reqQuery)
			authCode, loginState, err := parseCallbackQuery(req)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseCallbackQuery() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if authCode != tt.wantAuthCode {
				t.Errorf("parseCallbackQuery() authCode = %v, want %v", authCode, tt.wantAuthCode)
			}
			if loginState != tt.wantLoginState {
				t.Errorf("parseCallbackQuery() loginState = %v, want %v", loginState, tt.wantLoginState)
			}
		})
	}
}

func TestCheckLoginState(t *testing.T) {
	tests := []struct {
		name       string
		req        *restful.Request
		loginState string
		wantErr    bool
	}{
		{
			"TestNoCookie",
			&restful.Request{
				Request: &http.Request{},
			},
			"123456",
			true,
		},
		{
			"TestWrongCookie",
			&restful.Request{
				Request: &http.Request{
					Header: http.Header{
						"Cookie": []string{"state=654321"},
					},
				},
			},
			"123456",
			true,
		},
		{
			"TestRightCookie",
			&restful.Request{
				Request: &http.Request{
					Header: http.Header{
						"Cookie": []string{"state=123456"},
					},
				},
			},
			"123456",
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := checkLoginState(tt.req, tt.loginState); (err != nil) != tt.wantErr {
				t.Errorf("checkLoginState() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestHandlerExchangeCodeForToken(t *testing.T) {
	tests := []struct {
		name     string
		authCode string
		want     string
		want1    time.Time
		wantErr  bool
	}{
		{
			"TestExchangeFailure",
			"error-auth-code",
			"",
			time.Time{},
			true,
		},
		{
			"TestExchangeSuccess",
			"",
			"token-for-123456789",
			time.Unix(1, 1),
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newTestAuthHandler()
			got, err := h.exchangeCodeForToken(tt.authCode)
			if (err != nil) != tt.wantErr {
				t.Errorf("exchangeCodeForToken() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if (err != nil) && got == nil {
				return
			}
			if got.AccessToken != tt.want {
				t.Errorf("exchangeCodeForToken() got = %v, want %v", got.AccessToken, tt.want)
			}
			if !reflect.DeepEqual(got.AccessTokenExpiry, tt.want1) {
				t.Errorf("exchangeCodeForToken() got1 = %v, want %v", got.AccessTokenExpiry, tt.want1)
			}
		})
	}
}

// logout tests
func TestHandlerLogout(t *testing.T) {
	handler := newTestAuthHandler()

	t.Run("TestLogoutWithoutSession", func(t *testing.T) {
		req := httptest.NewRequest(
			"POST",
			"http://example.com/rest/auth/logout",
			nil,
		)
		recorder := httptest.NewRecorder()
		resp := restful.NewResponse(recorder)

		handler.logoutHandler(&restful.Request{Request: req}, resp)

		if recorder.Code != http.StatusNoContent {
			t.Errorf("logoutCore()() code = %v, want %v", recorder.Code, http.StatusNoContent)
		}
	})

	t.Run("TestLogoutWithSession", func(t *testing.T) {
		req := httptest.NewRequest(
			"POST",
			"http://example.com/rest/auth/logout",
			nil,
		)
		req.AddCookie(&http.Cookie{
			Name:  "sessionID",
			Value: "123456789",
		})
		recorder := httptest.NewRecorder()
		resp := restful.NewResponse(recorder)

		handler.logoutHandler(&restful.Request{Request: req}, resp)

		if recorder.Code != http.StatusNoContent {
			t.Errorf("logoutCore() code = %v, want %v", recorder.Code, http.StatusNoContent)
		}
		if val := recorder.Header().Get("Set-Cookie"); val == "" {
			t.Errorf("loginHandler() failed to set cookie")
		}
		wantLocation := ""
		if val := recorder.Header().Get("Location"); val != wantLocation {
			t.Errorf("loginHandler() redirect to wrong location: %s, should be %s", val, wantLocation)
		}
	})
}

// helpers
func removeEmptyCharHTML(str string) string {
	str = strings.ReplaceAll(str, "\n", "")
	str = strings.ReplaceAll(str, "\t", "")
	return str
}

// callbackHandler referrer redirect tests
func TestHandlerCallbackHandlerReferrerRedirect(t *testing.T) {
	handler := Handler{
		kubeConfig:  &rest.Config{},
		clientset:   newFakeClientSetWithTokenKey(),
		oauthConfig: &oAuth2Mock{},
	}
	tests := []struct {
		name              string
		referrerCookie    string
		stateCookie       string
		reqQuery          string
		wantLocation      string
		wantClearReferrer bool
	}{
		{
			name:              "redirect_to_referrer_with_valid_cookie",
			referrerCookie:    "/container_platform/workload/pods",
			stateCookie:       "correctstate",
			reqQuery:          "code=validcode&state=correctstate",
			wantLocation:      "/container_platform/workload/pods",
			wantClearReferrer: true,
		},
		{
			name:              "redirect_to_root_without_referrer_cookie",
			referrerCookie:    "",
			stateCookie:       "correctstate",
			reqQuery:          "code=validcode&state=correctstate",
			wantLocation:      "/",
			wantClearReferrer: false,
		},
		{
			name:              "redirect_to_full_url_with_referrer",
			referrerCookie:    "http://example.com/workload/pods?param=value",
			stateCookie:       "correctstate",
			reqQuery:          "code=validcode&state=correctstate",
			wantLocation:      "http://example.com/workload/pods?param=value",
			wantClearReferrer: true,
		},
		{
			name:              "redirect_to_root_with_login_page_referrer",
			referrerCookie:    "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc",
			stateCookie:       "correctstate",
			reqQuery:          "code=validcode&state=correctstate",
			wantLocation:      "/",
			wantClearReferrer: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := generateMockRequest("GET", loginCallbackEndpoint+"?"+tt.reqQuery)
			req.Request.AddCookie(&http.Cookie{Name: "state", Value: tt.stateCookie})
			if tt.referrerCookie != "" {
				req.Request.AddCookie(&http.Cookie{Name: constant.CookieNameReferrer, Value: tt.referrerCookie})
			}

			recorder := httptest.NewRecorder()
			resp := restful.NewResponse(recorder)
			handler.callbackHandler(req, resp)

			assertCallbackRedirect(t, recorder, tt.wantLocation, tt.wantClearReferrer)
		})
	}
}

// findCookie returns the cookie with the given name from the cookie list,
// or nil when no such cookie is present.
func findCookie(cookies []*http.Cookie, name string) *http.Cookie {
	for _, cookie := range cookies {
		if cookie.Name == name {
			return cookie
		}
	}
	return nil
}

// assertCallbackRedirect asserts the status code, Location header and referrer
// cookie clearing of a callback handler response.
func assertCallbackRedirect(t *testing.T, recorder *httptest.ResponseRecorder, wantLocation string, wantClearReferrer bool) {
	t.Helper()
	if recorder.Code != http.StatusFound {
		t.Errorf("callbackHandler() code = %v, want %v", recorder.Code, http.StatusFound)
		return
	}

	if location := recorder.Header().Get("Location"); location != wantLocation {
		t.Errorf("callbackHandler() Location = %v, want %v", location, wantLocation)
	}

	if !wantClearReferrer {
		return
	}
	// Check if referrer cookie is cleared when expected
	if cookie := findCookie(recorder.Result().Cookies(), constant.CookieNameReferrer); cookie != nil && cookie.Value != "" {
		t.Errorf("callbackHandler() should clear referrer cookie, but got value %v", cookie.Value)
	}
}
