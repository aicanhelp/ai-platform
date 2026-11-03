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

// Package fuyaopassword implements fuyaoidp login interfaces
package fuyaopassword

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"text/template"

	"github.com/gorilla/csrf"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	"openfuyao/oauth-server/assets/templates"
	overallconfigs "openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/authenticators"
	"openfuyao/oauth-server/pkg/config"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/fuyaostore"
	"openfuyao/oauth-server/pkg/httpserver"
	"openfuyao/oauth-server/pkg/i18n"
	"openfuyao/oauth-server/pkg/idp"
	"openfuyao/oauth-server/pkg/protector"
	"openfuyao/oauth-server/pkg/sessions"
	"openfuyao/oauth-server/pkg/utils"
	"openfuyao/oauth-server/pkg/zlog"
)

// LoginForm contains the fields used by fuyao login
type LoginForm struct {
	Action      string
	Then        string
	CSRFToken   string
	Base64Image string
	UserName    string
	Error       string
	I18n        map[string]string
}

// OutputHTML writes the contents back to web
func (l *LoginForm) OutputHTML(w http.ResponseWriter, tpl string, name string) {
	tplForm, err := template.New(name).Parse(tpl)
	if err != nil {
		http.Error(w, fuyaoerrors.ErrStrFailToDisplayLogin, http.StatusInternalServerError)
		return
	}
	if err = tplForm.Execute(w, l); err != nil {
		http.Error(w, fuyaoerrors.ErrStrFailToDisplayLogin, http.StatusInternalServerError)
		return
	}
}

// Login works for fuyao login, implement the login interfaces
type Login struct {
	Provider string
	// CSRF csrf.CSRF
	K8sClient      kubernetes.Interface
	dynamicClient  dynamic.Interface
	TokenStore     *fuyaostore.K8sSecretStore
	Authenticator  authenticators.PasswordAuthenticator
	IdpLoginStore  *sessions.CookieStore
	LoginProtector *protector.LoginUserProtector
}

// NewLogin returns the fuyao Login instance
func NewLogin(
	idpLoginStore *sessions.CookieStore,
	tokenStore *fuyaostore.K8sSecretStore,
	loginUserProtector *protector.LoginUserProtector,
	cfg *overallconfigs.OAuthServerAPIServerConfig,
) *Login {
	k8sClient := config.GetKubernetesClient(cfg.K8sConfig)
	dynamicClient := config.GetDynamicClient(cfg.K8sConfig)
	return &Login{
		Provider:       cfg.LoginConfig.Provider,
		K8sClient:      k8sClient,
		dynamicClient:  dynamicClient,
		TokenStore:     tokenStore,
		Authenticator:  authenticators.NewFuyaoPasswordAuthenticator(dynamicClient),
		IdpLoginStore:  idpLoginStore,
		LoginProtector: loginUserProtector,
	}
}

// LoginHandler deals with both GET/POST requests
func (l *Login) LoginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		// deal with GET
		l.handleLoginForm(w, r)
	} else if r.Method == http.MethodPost {
		// deal with POST
		l.processLogin(w, r)
	} else {
		http.Error(w, fuyaoerrors.ErrStrRequestMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}
}

// PasswordConfirmHandler works when the user login for the first time
func (l *Login) PasswordConfirmHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		// deal with GET
		l.handlePasswordConfirmForm(w, r)
	} else if r.Method == http.MethodPost {
		// deal with POST
		l.processPasswordConfirm(w, r)
	} else if r.Method == http.MethodDelete {
		l.revertPasswordConfirm(w, r)
	} else {
		http.Error(w, fuyaoerrors.ErrStrRequestMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}
}

func (l *Login) handlePasswordConfirmForm(w http.ResponseWriter, r *http.Request) {
	// 生成 uri
	uri, err := idp.GetBaseURL(r)
	if err != nil {
		zlog.LogErrorf("unable to fetch password confirm requestURL, err: %v", err)
		http.Error(w, "unable to fetch requestURL", http.StatusInternalServerError)
		return
	}

	// fetch then from r
	then := r.URL.Query().Get(constants.ThenParam)
	if !isValidThenURL(then) {
		http.Redirect(w, r, getConsoleServiceHost(r), http.StatusFound)
		return
	}

	// get error from r
	errString := r.URL.Query().Get(constants.ErrorParam)

	// read userinfo from session-store
	loginData := l.IdpLoginStore.Get(r)
	username, ok := loginData.GetString(constants.UserName)
	if !ok {
		zlog.LogErrorf("cannot enter password confirmation process, %s", fuyaoerrors.ErrStrNotLogin)
		http.Redirect(w, r, getConsoleServiceHost(r), http.StatusFound)
		return
	}

	// read image
	imageData := readLoginBackground()

	// get locale from request
	locale := i18n.GetLocaleFromRequest(r)

	// 生成loginForm
	loginForm := LoginForm{
		Action:      uri.String(),
		Then:        html.EscapeString(then),
		UserName:    username,
		Base64Image: imageData,
		CSRFToken:   string(csrf.TemplateField(r)),
		Error:       html.EscapeString(errString),
		I18n:        buildPasswordConfirmI18n(locale),
	}

	// render form
	loginForm.OutputHTML(w, templates.DefaultPasswordConfirmTemplateString, constants.PasswordConfirmFormTemplate)
}

func (l *Login) processPasswordConfirm(w http.ResponseWriter, r *http.Request) {
	// read userinfo from session-store
	loginData := l.IdpLoginStore.Get(r)
	username, ok := loginData.GetString(constants.UserName)
	if !ok {
		zlog.LogErrorf("Password confirmation fail for user %s, err: the idpLogin cookie is missing", username)
		http.Redirect(w, r, getConsoleServiceHost(r), http.StatusFound)
		return
	}

	// read params from r.url
	var requestBody PasswordConfirmRequest
	if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
		zlog.LogErrorf("Password Confirm failed, error: %v", err)
		httpserver.RespondWithStatusMsg(w, http.StatusBadRequest, 0, fuyaoerrors.ErrStrFailToUnmarshalData)
		return
	}

	byteNewPassword := requestBody.NewPassword
	then := requestBody.Then
	defer destroyBytes(byteNewPassword)
	if len(byteNewPassword) == 0 {
		zlog.LogErrorf("Password confirmation fail for user %s, err: %v", username,
			fuyaoerrors.ErrUsernameOrPasswordMissing)
		redirectGetMethodWithError(w, r, fuyaoerrors.ErrStrUsernameOrPasswordMissing, then)
		return
	}
	if !isValidThenURL(then) {
		redirectGetMethodWithError(w, r, i18n.T(i18n.GetLocaleFromRequest(r), "error.invalid_then"), then)
		return
	}

	// password confirmation logic
	locale := i18n.GetLocaleFromRequest(r)
	if err := l.Authenticator.ConfirmPassword(context.Background(), username, byteNewPassword); err != nil {
		zlog.LogErrorf("Password confirmation fail for user %s, err: %v", username, err)
		var errString string
		switch {
		case errors.Is(err, fuyaoerrors.ErrPasswordSame):
			errString = i18n.T(locale, "error.password_same")
		default:
			errString = err.Error()
		}
		redirectGetMethodWithError(w, r, errString, then)
		return
	}

	// set first-login to false in the oauth-session
	// if fail in the following 8 lines only idpLogin cookie first-login is tainted, we can still redirect safely
	ok = loginData.SetLoggedIn()
	if !ok {
		zlog.LogError("fail to set first-login state to false, probably due to web modification")
	} else if err := l.IdpLoginStore.Put(w, loginData); err != nil {
		zlog.LogError("fail to store loginState to session")
	} else {
		zlog.LogInfo("Successfully set idpLogin state for password confirmation.")
	}
	zlog.LogInfof("Password confirmation succeed for user: %s", username)

	// redirect normally
	http.Redirect(w, r, then, http.StatusFound)
}

func (l *Login) revertPasswordConfirm(w http.ResponseWriter, r *http.Request) {
	// delete loginState
	if err := l.IdpLoginStore.Put(w, make(sessions.Values)); err != nil {
		zlog.LogErrorf("cannot delete the loginstore used in authorization, err: %v", err)
	}

	// redirect to console-service host
	redirect := getConsoleServiceHost(r)

	http.Redirect(w, r, redirect, http.StatusFound)
}

func getConsoleServiceHost(r *http.Request) string {
	return "/"
}

// PasswordResetHandler resets the password
func (l *Login) PasswordResetHandler(w http.ResponseWriter, r *http.Request) {
	// check http method
	if r.Method != http.MethodPost {
		httpserver.RespondWithStatusMsg(w, http.StatusMethodNotAllowed, 0,
			fuyaoerrors.ErrStrRequestMethodNotAllowed)
		return
	}

	// add an access token validation, since all the services are required to expose in this version
	if !l.webhookAuthentication(w, r) {
		return
	}

	// read params from r.url
	var requestBody PasswordResetRequest
	if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
		zlog.LogErrorf("Password Reset failed, error: %v", err)
		httpserver.RespondWithStatusMsg(w, http.StatusBadRequest, 0, fuyaoerrors.ErrStrFailToUnmarshalData)
		return
	}

	username := utils.EscapeSpecialChars(requestBody.Username)
	byteOldPassword := requestBody.OriginalPassword
	byteNewPassword := requestBody.NewPassword
	defer destroyBytes(byteOldPassword)
	defer destroyBytes(byteNewPassword)
	if len(username) == 0 || len(byteOldPassword) == 0 || len(byteNewPassword) == 0 {
		zlog.LogErrorf("Password Reset failed, error: %v",
			fuyaoerrors.ErrUsernameOrPasswordMissing)
		httpserver.RespondWithStatusMsg(w, http.StatusBadRequest, 0,
			fuyaoerrors.ErrStrUsernameOrPasswordMissing)
		return
	}

	ipAddress := getIPAddress(r)
	zlog.LogInfof("Password reset from %s", ipAddress)

	if err := l.Authenticator.ResetPassword(context.Background(), username, byteOldPassword, byteNewPassword); err != nil {
		zlog.LogErrorf("Password Reset failed, error: %v", err)
		// add failed records and check blocking
		if errors.Is(err, fuyaoerrors.ErrPasswordResetFailed) {
			blocked, errString := l.checkForUserBlocking(username, i18n.GetLocaleFromRequest(r))
			if blocked {
				httpserver.RespondWithStatusMsg(w, http.StatusFound, 0, errString)
			} else {
				httpserver.RespondWithStatusMsg(w, fuyaoerrors.ErrStatusCode[err], 0, errString)
			}
			return
		}
		// translate known user-facing errors
		locale := i18n.GetLocaleFromRequest(r)
		var errString string
		switch {
		case errors.Is(err, fuyaoerrors.ErrPasswordSame):
			errString = i18n.T(locale, "error.password_same")
		default:
			errString = err.Error()
		}
		httpserver.RespondWithStatusMsg(w, fuyaoerrors.ErrStatusCode[err], 0, errString)
		return
	}
	zlog.LogInfof("Password Reset succeed for user: %s", username)

	httpserver.RespondWithStatusMsg(w, http.StatusOK, 0, "Password Reset OK")
	return
}

func (l *Login) webhookAuthentication(w http.ResponseWriter, r *http.Request) bool {
	accessToken, err := l.getAccessToken(r)
	if err != nil {
		zlog.LogErrorf("Password Reset failed, error: %v", err)
		httpserver.RespondWithStatusMsg(w, http.StatusUnauthorized, 0, fuyaoerrors.ErrStrNotLogin)
		return false
	}

	loggedIn, err := l.authenticateByWebhook(accessToken)
	if !loggedIn || err != nil {
		zlog.LogErrorf("Password Reset failed, error: %v", err)
		httpserver.RespondWithStatusMsg(w, fuyaoerrors.ErrStatusCode[err], 0, err.Error())
		return false
	}
	return true
}

func (l *Login) getAccessToken(r *http.Request) (string, error) {
	// blindly get access-token from header
	authHeader := r.Header.Get("Authorization")
	if authHeader != "" {
		// Check if the Authorization header starts with "Bearer "
		if strings.HasPrefix(authHeader, "Bearer ") {
			// Extract the access token
			accessToken := strings.TrimPrefix(authHeader, "Bearer ")
			return accessToken, nil
		}
	}

	return "", fuyaoerrors.ErrNotLogin
}

func (l *Login) authenticateByWebhook(accessToken string) (bool, error) {
	tokenReview := &authenticationv1.TokenReview{
		Spec: authenticationv1.TokenReviewSpec{Token: accessToken},
	}

	tokenReviewResponse, err := l.K8sClient.AuthenticationV1().TokenReviews().Create(
		context.TODO(), tokenReview, metav1.CreateOptions{})
	if err != nil {
		zlog.LogErrorf("cannot post tokenReview to k8s, err: %v", err)
		return false, fuyaoerrors.ErrNotLogin
	}

	return tokenReviewResponse.Status.Authenticated, nil
}

func (l *Login) handleLoginForm(w http.ResponseWriter, r *http.Request) {
	// 生成 uri
	uri, err := idp.GetBaseURL(r)
	if err != nil {
		zlog.LogErrorf("unable to fetch login requestURL, err: %v", err)
		http.Error(w, "unable to fetch requestURL", http.StatusInternalServerError)
		return
	}

	then := r.URL.Query().Get(constants.ThenParam)
	if !isValidThenURL(then) {
		http.Redirect(w, r, getConsoleServiceHost(r), http.StatusFound)
		return
	}

	// get error from r
	errString := r.URL.Query().Get(constants.ErrorParam)

	// read image
	imageData := readLoginBackground()

	// get locale from request
	locale := i18n.GetLocaleFromRequest(r)

	// 生成loginForm
	loginForm := LoginForm{
		Action:      uri.String(),
		Then:        html.EscapeString(then),
		Base64Image: imageData,
		CSRFToken:   string(csrf.TemplateField(r)),
		Error:       html.EscapeString(errString),
		I18n:        buildLoginI18n(locale),
	}

	// render form
	loginForm.OutputHTML(w, templates.DefaultLoginTemplateString, constants.LoginFormTemplate)
}

func (l *Login) processLogin(w http.ResponseWriter, r *http.Request) {
	// read params from r.url
	var requestBody LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
		httpserver.RespondWithStatusMsg(w, http.StatusBadRequest, 0, fuyaoerrors.ErrStrFailToUnmarshalData)
		return
	}
	username := utils.EscapeSpecialChars(requestBody.Username)
	bytePassword := requestBody.Password
	defer destroyBytes(bytePassword)
	then := requestBody.Then

	// check form value
	if len(username) == 0 || len(bytePassword) == 0 {
		redirectGetMethodWithError(w, r, fuyaoerrors.ErrStrUsernameOrPasswordMissing, then)
		return
	}
	if !isValidThenURL(then) {
		redirectGetMethodWithError(w, r, i18n.T(i18n.GetLocaleFromRequest(r), "error.invalid_then"), then)
		return
	}

	// login devastation check
	ipAddress := getIPAddress(r)
	zlog.LogInfof("Login request from %s: Username: %s\n", ipAddress, username)
	// 这里变成直接查询userStatus
	if locked, errString := l.LoginProtector.CheckLocked(username, i18n.GetLocaleFromRequest(r)); locked {
		redirectGetMethodWithError(w, r, errString, then)
		return
	}

	// verify the password
	response, ok, err := l.Authenticator.AuthenticatePassword(context.Background(), username, bytePassword)

	// service internal error
	if err != nil && !errors.Is(err, fuyaoerrors.ErrPasswordAuthenticationFailed) {
		zlog.LogErrorf("Login request fail, error: %v", err)
		redirectGetMethodWithError(w, r, fuyaoerrors.ErrStrLoginServiceDown, then)
		return
	}

	// password authentication error
	if !ok {
		// 这里给对应用户的blocker加一
		_, errString := l.checkForUserBlocking(username, i18n.GetLocaleFromRequest(r))
		redirectGetMethodWithError(w, r, errString, then)
		return
	}

	// successfully login, erase user block flags
	l.LoginProtector.Unlock(username)

	// redirect if already logged in
	if _, ok = l.IdpLoginStore.Get(r).GetString(constants.UserName); ok {
		http.Redirect(w, r, then, http.StatusFound)
		return
	}

	if err = l.saveLoginStateToSession(response.User, w); err != nil {
		zlog.LogErrorf("Login request fail, error: %v", err)
		redirectGetMethodWithError(w, r, fuyaoerrors.ErrStrLoginServiceDown, then)
		return
	}

	zlog.LogInfof("Successfully logging in with %s", response.User.GetName())
	// 重定向回到 /oauth/authorize
	http.Redirect(w, r, then, http.StatusFound)
}

func (l *Login) checkForUserBlocking(username, locale string) (bool, string) {
	// current ip failed times +1
	remainingAttempt := l.LoginProtector.AddFailedLogin(username)

	// still got login attempts
	if remainingAttempt > 0 {
		errString := i18n.TFormattable(locale, "error.username_or_password_count",
			strconv.FormatInt(int64(remainingAttempt), constants.Decimal))
		zlog.LogErrorf("Login request fail, error: %s", errString)
		return false, errString
	}

	// trigger user blocking
	lockDuration := int64(l.LoginProtector.LockDuration.Minutes())
	errString := i18n.TFormattable(locale, "error.username_or_password_locked",
		strconv.FormatInt(lockDuration, constants.Decimal))
	zlog.LogErrorf("Login request fail and trigger user blocking for %s", username)
	return true, errString
}

func (l *Login) saveLoginStateToSession(user user.Info, w http.ResponseWriter) error {
	values := sessions.Values{}
	values[constants.UserName] = user.GetName()
	values[constants.UserUID] = user.GetUID()
	values[constants.UserGroups] = user.GetGroups()

	// serialize extra (map[string][]string)
	extra := user.GetExtra()

	// add the web-oauthserver session-id
	const sessionIDLength = 32
	sessionID, err := generateSessionID(sessionIDLength)
	extra[constants.OAuthServerSessionID] = []string{sessionID}

	// save the extra information
	jsonExtra, err := json.Marshal(extra)
	if err != nil {
		zlog.LogErrorf("cannot marshal data, err: %v", err)
		return fuyaoerrors.ErrFailToMarshalData
	}
	values[constants.UserExtra] = jsonExtra
	return l.IdpLoginStore.Put(w, values)
}

// ---- util functions ----
func getIPAddress(r *http.Request) string {
	// first fetch from X-Forwarded-For header
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		// X-Forwarded-For may have multiple ips
		ips := strings.Split(xff, ",")
		// return the first ip and trim it
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// use RemoteAddr instead
	ip := r.RemoteAddr
	// remove the port if existed
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}

	return ip
}

func destroyBytes(bt []byte) {
	for i := range bt {
		bt[i] = 0
	}
}

func isServerRelatedURL(uri string) bool {
	// check whether it is empty
	if len(uri) == 0 {
		return false
	}

	// check whether it follows url pattern
	u, err := url.Parse(uri)
	if err != nil {
		return false
	}

	return strings.HasPrefix(u.Path, "/") && len(u.Scheme) == 0 && len(u.Host) == 0
}

func isValidThenURL(uri string) bool {
	if !strings.HasPrefix(uri, constants.FuyaoOAuthAuthorizeEndpoint+"?") {
		return false
	}

	const validLen = 2
	parts := strings.Split(uri, "?")
	if len(parts) != validLen {
		return false
	}

	return true
}

func readBase64Image(filePath string) (string, error) {
	imageData, err := os.ReadFile(filePath)
	if err != nil {
		zlog.LogErrorf("failed to read image file: %v", err)
		return "", err
	}

	return base64.StdEncoding.EncodeToString(imageData), nil
}

// readLoginBackground reads the login background image and returns a CSS url()
// value. On any read error it falls back to the default background constant.
func readLoginBackground() string {
	imageData, err := readBase64Image(constants.LoginBackgroundPath)
	if err != nil {
		return constants.DefaultLoginBackground
	}
	return fmt.Sprintf(`url("data:image/png;base64,%s")`, imageData)
}

func redirectGetMethodWithError(w http.ResponseWriter, r *http.Request, errString, then string) {
	// build redirect url
	encodedErrString := url.QueryEscape(errString)
	encodedThen := url.QueryEscape(then)
	redirect := fmt.Sprintf("%s?then=%s&error=%s", r.URL.String(), encodedThen, encodedErrString)

	// redirect to GET handleLogin
	http.Redirect(w, r, redirect, http.StatusFound)
}

func generateSessionID(length int) (string, error) {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	sessionID := make([]byte, length)

	for i := 0; i < length; i++ {
		num, err := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
		if err != nil {
			zlog.LogErrorf("Cannot generate random char, err: %v", err)
			return "", err
		}
		sessionID[i] = charset[num.Int64()]
	}

	return string(sessionID), nil
}

// buildLoginI18n builds the i18n map for the login page
func buildLoginI18n(locale string) map[string]string {
	return map[string]string{
		"pageTitle":        i18n.T(locale, "page.title"),
		"welcome":          i18n.T(locale, "login.welcome"),
		"username":         i18n.T(locale, "login.username"),
		"password":         i18n.T(locale, "login.password"),
		"loginBtn":         i18n.T(locale, "login.button"),
		"usernameRequired": i18n.T(locale, "login.username_required"),
		"passwordRequired": i18n.T(locale, "login.password_required"),
	}
}

// buildPasswordConfirmI18n builds the i18n map for the password confirm page
func buildPasswordConfirmI18n(locale string) map[string]string {
	return map[string]string{
		"pageTitle":       i18n.T(locale, "page.title"),
		"title":           i18n.T(locale, "password_confirm.title"),
		"prompt":          i18n.T(locale, "password_confirm.prompt"),
		"newPassword":     i18n.T(locale, "password_confirm.new_password"),
		"confirmPassword": i18n.T(locale, "password_confirm.confirm"),
		"lengthRule":      i18n.T(locale, "password_confirm.length_rule"),
		"complexity":      i18n.T(locale, "password_confirm.complexity"),
		"sameAsAccount":   i18n.T(locale, "password_confirm.same_as_account"),
		"mismatch":        i18n.T(locale, "password_confirm.mismatch"),
		"cancel":          i18n.T(locale, "password_confirm.cancel"),
		"confirmBtn":      i18n.T(locale, "password_confirm.confirm_btn"),
	}
}
