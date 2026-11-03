/*
 * Copyright OAuth2_proxy Authors.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://opensource.org/licenses/MIT
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * NEW COPYRIGHT AND LICENSE
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 * functions modified:
 * type FuyaoProvider struct
 * func (p *FuyaoProvider) Complete (rearrange funcs)
 * functions added:
 * func (o recordsByPath) MatchAndSetVerb
 * func matchPathByRe
 * func (p *FuyaoProvider) ValidateMultiClusterRequest
 */

// Package openfuyao defines the openfuyao provider configs
package openfuyao

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/bitly/go-simplejson"
	"golang.org/x/oauth2"
	authenticationv1 "k8s.io/api/authentication/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"openfuyao/oauth-proxy/constants"
	"openfuyao/oauth-proxy/cookie"
	"openfuyao/oauth-proxy/providers"
)

// FuyaoProvider is the oauth provider interface for oauth-server
type FuyaoProvider struct {
	*providers.ProviderData

	AuthenticationOptions DelegatingAuthenticationOptions
	AuthorizationOptions  DelegatingAuthorizationOptions
	KubeClientOptions     KubeClientOptions

	authenticator authenticator.Request
	authorizer    authorizer.Authorizer

	kubeClient         *kubernetes.Clientset
	defaultRecord      authorizer.AttributesRecord
	reviews            []string
	paths              recordsByPath
	hostreviews        map[string][]string
	logoutURL          string
	sessionIDNamespace string
}

// New init the FuyaoProvider
func New() *FuyaoProvider {
	p := &FuyaoProvider{}
	p.AuthenticationOptions.SkipInClusterLookup = true
	p.AuthenticationOptions.CacheTTL = constants.AuthnCacheTTL * time.Minute
	p.AuthorizationOptions.AllowCacheTTL = constants.AuthzAllowCacheTTL * time.Minute
	p.AuthorizationOptions.DenyCacheTTL = constants.AuthzDenyCacheTTL * time.Second
	return p
}

// SetClientCAFile set the client ca
func (p *FuyaoProvider) SetClientCAFile(file string) {
	p.AuthenticationOptions.ClientCert.ClientCA = file // 这里的clientCA是访问k8s的吧，其他地方都是作为server验证客户端的
}

// Bind add the authn cmdline param to flgas
func (p *FuyaoProvider) Bind(flags *flag.FlagSet) {
	p.KubeClientOptions.AddFlags(flags)
	p.AuthenticationOptions.AddFlags(flags)
	p.AuthorizationOptions.AddFlags(flags)
	flags.StringVar(&p.logoutURL, "delete-token-url", p.logoutURL, "oauth-server logout endpoint")
	flags.StringVar(&p.sessionIDNamespace, "sessionid-namespace", p.sessionIDNamespace,
		"namespace to store the sessionid secret in k8s")
}

// LoadDefaults accepts configuration and loads defaults from the environment, or returns an error.
// The provider may partially initialize config for subsequent calls.
func (p *FuyaoProvider) LoadDefaults(reviewJSON, reviewByHostJSON, resources string) (*providers.ProviderData, error) {
	if len(resources) > 0 {
		log.Printf("openfuyao-delegatae-urls: %s", resources)
		paths, err := parseResources(resources)
		if err != nil {
			return nil, err
		}
		p.paths = paths
	}
	reviews, err := parseSubjectAccessReviews(reviewJSON)
	if err != nil {
		return nil, err
	}
	p.reviews = reviews

	hostreviews, err := parseSubjectAccessReviewsByHost(reviewByHostJSON)
	if err != nil {
		return nil, err
	}
	p.hostreviews = hostreviews

	defaults := &providers.ProviderData{
		Scope: "user:info user:check-access",
	}

	// provide default URLs
	defaults.ValidateURL = getKubeAPIURLWithPath("/apis/user.openfuyao.io/v1/users/~")

	return defaults, nil
}

// encodeSARWithScope adds a "scopes" array to the SAR if it does not have one already, and outputs
// encoded bytes.
func encodeSARWithScope(json *simplejson.Json) ([]byte, error) {
	if len(json.Get("scopes").MustArray()) == 0 {
		json.Set("scopes", []interface{}{})
	}
	return json.Encode()
}

func parseSubjectAccessReviewsByHost(review string) (map[string][]string, error) {
	if len(review) == 0 {
		return nil, nil
	}
	json, err := simplejson.NewJson([]byte(review))
	if err != nil {
		return nil, fmt.Errorf("unable to decode review: %v", err)
	}

	reviews := make(map[string][]string)
	for k := range json.MustMap() {
		data, err := json.Get(k).EncodePretty()
		if err != nil {
			return nil, err
		}
		r, err := parseSubjectAccessReviews(string(data))
		if err != nil {
			return nil, err
		}
		reviews[k] = r
	}
	return reviews, nil
}

// parseSubjectAccessReviews parses a list of SAR records and ensures they are properly scoped.
func parseSubjectAccessReviews(review string) ([]string, error) {
	review = strings.TrimSpace(review)
	if len(review) == 0 {
		return nil, nil
	}

	// Convert to a json array to simplify encoding later
	if review[0] != '[' && review[len(review)-1] != ']' {
		review = "[" + review + "]"
	}

	json, err := simplejson.NewJson([]byte(review))
	if err != nil {
		return nil, fmt.Errorf("unable to decode review: %v", err)
	}

	var reviews []string
	for i := range json.MustArray() {
		data, err := encodeSARWithScope(json.GetIndex(i))
		if err != nil {
			return nil, fmt.Errorf("unable to encode modified review: %v (%#v)", err, json)
		}
		reviews = append(reviews, string(data))
	}
	return reviews, nil
}

type pathRecord struct {
	path   string
	record authorizer.AttributesRecord
}

type recordsByPath []pathRecord

func (o recordsByPath) Len() int { return len(o) }
func (o recordsByPath) Swap(i, j int) {
	if i >= 0 && i <= len(o) && j >= 0 && j <= len(o) {
		o[i], o[j] = o[j], o[i]
	}
}
func (o recordsByPath) Less(i, j int) bool {
	if i < 0 || j < 0 || i >= len(o) || j >= len(o) {
		return false
	}
	// match longest paths first
	if len(o[j].path) < len(o[i].path) {
		return true
	}
	// match in lexographic order otherwise
	return o[i].path < o[j].path
}

func (o recordsByPath) Match(path string) (pathRecord, bool) {
	for i := range o {
		if strings.HasPrefix(path, o[i].path) {
			return o[i], true
		}
	}
	return pathRecord{}, false
}

var methodToVerb = map[string]string{
	"GET":    "get",
	"POST":   "create",
	"PUT":    "create",
	"PATCH":  "create",
	"DELETE": "delete",
}

func (o recordsByPath) MatchAndSetVerb(req *http.Request) (pathRecord, bool) {
	path := req.URL.Path
	for _, rc := range o {
		if matchPathByRe(path, rc.path) {
			// set the record verb according to req.Method
			if verb, ok := methodToVerb[req.Method]; ok {
				log.Printf("successfully match path %s with verb %s", rc.path, verb)
				rc.record.Verb = verb
			}
			return rc, true
		}
	}
	return pathRecord{}, false
}

// matchPathByRe matches request path and record path by regex
func matchPathByRe(reqPath, recordPath string) bool {
	if !strings.Contains(recordPath, "{") || !strings.Contains(recordPath, "}") {
		return strings.HasPrefix(reqPath, recordPath)
	}
	pattern := buildPattern(recordPath)
	log.Printf("generated match path pattern: %s", pattern)

	re := regexp.MustCompile(pattern)
	return re.MatchString(reqPath)
}

// buildPattern Function to convert the URL with {variables} to a regular expression
func buildPattern(url string) string {
	// Replace all {xxx} with regex pattern to match any value (e.g., "[^/]+")
	re := regexp.MustCompile(`\{[^}]+\}`)
	pattern := re.ReplaceAllString(url, `[^/]+`)

	// Escape special regex characters
	escapedPattern := regexp.QuoteMeta(pattern)

	// Replace the escaped `[^/]+` to make it a valid regex
	escapedPattern = strings.ReplaceAll(escapedPattern, `\[\^/\]\+`, `[^/]+`)

	return "^" + escapedPattern + "$"
}

// parseResources creates a map of path prefixes (the keys in the provided input) to
// SubjectAccessReview ResourceAttributes (the keys) and returns the records ordered
// by longest path first, or an error.
func parseResources(resources string) (recordsByPath, error) {
	defaults := authorizer.AttributesRecord{
		Verb:            "proxy",
		ResourceRequest: true,
	}
	var paths recordsByPath
	mappings := make(map[string]authorizationv1.ResourceAttributes)
	if err := json.Unmarshal([]byte(resources), &mappings); err != nil {
		return nil, fmt.Errorf("resources must be a JSON map of paths to authorizationv1beta1.ResourceAttributes: %v", err)
	}
	for path, attrs := range mappings {
		r := defaults
		if len(attrs.Verb) > 0 {
			r.Verb = attrs.Verb
		}
		if len(attrs.Group) > 0 {
			r.APIGroup = attrs.Group
		}
		if len(attrs.Version) > 0 {
			r.APIVersion = attrs.Version
		}
		if len(attrs.Resource) > 0 {
			r.Resource = attrs.Resource
		}
		if len(attrs.Subresource) > 0 {
			r.Subresource = attrs.Subresource
		}
		if len(attrs.Namespace) > 0 {
			r.Namespace = attrs.Namespace
		}
		if len(attrs.Name) > 0 {
			r.Name = attrs.Name
		}
		paths = append(paths, pathRecord{
			path:   path,
			record: r,
		})
	}
	sort.Sort(paths)
	return paths, nil
}

// Complete performs final setup on the provider or returns an error.
func (p *FuyaoProvider) Complete(data *providers.ProviderData) error {
	p.ProviderData = data

	kubeClient, err := p.KubeClientOptions.ToKubeClientConfig()
	if err != nil {
		return err
	}

	p.kubeClient = kubeClient

	if p.sessionIDNamespace == "" {
		p.sessionIDNamespace = "oauth-sessionid"
	}

	log.Printf("Delegation of authentication and authorization to OpenFuyao is " +
		"enabled for bearer tokens and client certificates.")

	authenticator, err := p.initAuthenticator()
	if err != nil {
		return err
	}

	authorizer, err := p.initAuthorizer()
	if err != nil {
		return err
	}

	p.authenticator, _, err = authenticator.New()
	if err != nil {
		return fmt.Errorf("unable to configure authenticator: %v", err)
	}

	p.authorizer, err = authorizer.New()
	if err != nil {
		return fmt.Errorf("unable to configure authorizer: %v", err)
	}

	return nil
}

func (p *FuyaoProvider) initAuthorizer() (authorizerfactory.DelegatingAuthorizerConfig, error) {
	authorizer, err := p.AuthorizationOptions.ToAuthorizationConfig()
	if err != nil {
		return authorizerfactory.DelegatingAuthorizerConfig{},
			fmt.Errorf("unable to configure authorizer: %v", err)
	}
	// check whether we have access to perform authentication review
	if authorizer.SubjectAccessReviewClient != nil {
		const interval = 2
		const timeout = 10
		wait.PollImmediate(interval*time.Second, timeout*time.Second, func() (bool, error) {
			_, err := authorizer.SubjectAccessReviewClient.SubjectAccessReviews().Create(context.TODO(),
				&authorizationv1.SubjectAccessReview{
					Spec: authorizationv1.SubjectAccessReviewSpec{
						User: "TEST",
						ResourceAttributes: &authorizationv1.ResourceAttributes{
							Resource: "TEST",
							Verb:     "TEST",
						},
					},
				}, metav1.CreateOptions{})
			if err != nil {
				log.Printf("unable to retrieve authorization information for users")
				return false, nil
			}
			return true, nil
		})
	}
	return authorizer, nil
}

func (p *FuyaoProvider) initAuthenticator() (authenticatorfactory.DelegatingAuthenticatorConfig, error) {
	authenticator, err := p.AuthenticationOptions.ToAuthenticationConfig()
	if err != nil {
		return authenticatorfactory.DelegatingAuthenticatorConfig{},
			fmt.Errorf("unable to configure authenticator: %v", err)
	}
	// check whether we have access to perform authentication review
	if authenticator.TokenAccessReviewClient != nil {
		const interval = 2
		const timeout = 10
		wait.PollImmediate(interval*time.Second, timeout*time.Second, func() (bool, error) {
			_, err := authenticator.TokenAccessReviewClient.TokenReviews().Create(context.TODO(),
				&authenticationv1.TokenReview{
					Spec: authenticationv1.TokenReviewSpec{
						Token: "TEST",
					},
				},
				metav1.CreateOptions{})
			if err != nil {
				log.Printf("unable to retrieve authentication information for tokens")
				return false, nil
			}
			return true, nil
		})
	}
	return authenticator, nil
}

// AuthenticateByWebhook use webhook request to validate the accessToken
func (p *FuyaoProvider) AuthenticateByWebhook(accessToken string) (*authenticator.Response, bool, error) {
	tokenReview := &authenticationv1.TokenReview{
		Spec: authenticationv1.TokenReviewSpec{Token: accessToken},
	}

	tokenReviewResponse, err := p.kubeClient.AuthenticationV1().TokenReviews().Create(
		context.TODO(), tokenReview, metav1.CreateOptions{})
	if err != nil {
		log.Printf("cannot post tokenReview to k8s")
		return nil, false, err
	}

	authenticated := tokenReviewResponse.Status.Authenticated
	if !authenticated {
		return nil, false, nil
	}
	userInfo := tokenReviewResponse.Status.User

	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name:   userInfo.Username,
			UID:    userInfo.UID,
			Groups: userInfo.Groups,
			Extra:  convertToMapOfStringSlice(userInfo.Extra),
		},
		Audiences: tokenReviewResponse.Status.Audiences,
	}, true, nil
}

func convertToMapOfStringSlice(extraMap map[string]authenticationv1.ExtraValue) map[string][]string {
	result := make(map[string][]string)

	for key, extra := range extraMap {
		var extraSlice []string
		for _, value := range extra {
			extraSlice = append(extraSlice, value)
		}
		result[key] = extraSlice
	}

	return result
}

// ValidateSessionAuth use webhook authn and authz to validate user and its RBAC
func (p *FuyaoProvider) ValidateSessionAuth(session *providers.SessionState, req *http.Request) (bool, error) {
	// authn
	response, ok, err := p.AuthenticateByWebhook(session.AccessToken)
	if !ok || err != nil {
		return ok, err
	}

	// authz
	err = p.ReviewUser(response.User.GetName(), req.Host)
	if err != nil {
		return false, err
	}

	session.User = response.User.GetName()
	return true, nil
}

// ValidateRequest deals with the request from other components in cluster instead of web user
func (p *FuyaoProvider) ValidateRequest(req *http.Request) (*providers.SessionState, error) {
	// no authenticator is registered
	if p.authenticator == nil {
		return nil, nil
	}

	// authenticate
	auth := req.Header.Get("Authorization")
	response, ok, err := p.authenticator.AuthenticateRequest(req)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}
	userName := response.User.GetName()
	log.Printf("validate request get Authorization header: %s, username: %s", auth, userName)
	if userName == "system:anonymous" {
		return nil, nil
	}

	// authorize
	if len(p.paths) > 0 {
		log.Printf("------ check authz by SAR ------")
		// find a match
		record, ok := p.paths.MatchAndSetVerb(req)
		if !ok {
			log.Printf("no resource mapped path")
			return nil, nil
		}

		record.record.User = response.User
		decision, _, err := p.authorizer.Authorize(context.Background(), record.record)
		if err != nil {
			return nil, err
		}
		if decision != authorizer.DecisionAllow {
			return nil, nil
		}

	}

	const partNum = 2
	parts := strings.SplitN(auth, " ", partNum)
	session := &providers.SessionState{User: userName, Email: userName + "@cluster.local"}
	if parts[0] == "Bearer" {
		session.AccessToken = parts[len(parts)-1]
	}
	return session, nil
}

// ValidateMultiClusterRequest validate requests from the host cluster
func (p *FuyaoProvider) ValidateMultiClusterRequest(req *http.Request) (*providers.SessionState, error) {
	token := req.Header.Get(constants.OpenFuyaoAuthHeaderKey)
	if !strings.HasPrefix(token, "Bearer ") {
		return nil, nil
	}

	// decode the token info
	token = strings.TrimPrefix(token, "Bearer ")
	extractedUser, err := ExtractUserFromJWT(token)
	if err != nil {
		return nil, err
	}
	log.Printf("request from user: %s", extractedUser.GetName())
	req = req.WithContext(context.WithValue(req.Context(), "openFuyao-user", extractedUser))

	// find a match (bypass authz if length p.paths is 0)
	if len(p.paths) > 0 {
		log.Printf("------ check authz by SAR ------")
		record, ok := p.paths.MatchAndSetVerb(req)
		if !ok {
			log.Printf("no resource mapped path")
			return nil, nil
		}
		record.record.User = extractedUser
		log.Printf("the rbac authz record: %v", record.record)

		decision, reason, err := p.authorizer.Authorize(context.Background(), record.record)
		if err != nil {
			return nil, err
		}
		if decision != authorizer.DecisionAllow {
			log.Printf("rbac failed, reason: %s", reason)
			return nil, nil
		}
	}

	const partNum = 2
	parts := strings.SplitN(token, " ", partNum)
	session := &providers.SessionState{User: extractedUser.GetName(), Email: extractedUser.GetName() + "@cluster.local"}
	if parts[0] == "Bearer" {
		session.AccessToken = parts[len(parts)-1]
	}
	return session, nil
}

// GetUserName return the username
func (p *FuyaoProvider) GetUserName(accessToken string) (string, error) {
	response, ok, err := p.AuthenticateByWebhook(accessToken)
	if !ok || err != nil {
		log.Printf("cannot get userinfo from accessToken")
		return "", err
	}
	return response.User.GetName(), nil
}

// GetEmailAddress return the email address
func (p *FuyaoProvider) GetEmailAddress(s *providers.SessionState) (string, error) {
	return "*", nil
}

// ReviewUser reviews the RBAC of the current user
func (p *FuyaoProvider) ReviewUser(username, host string) error {
	var tocheck []string

	hostreviews, ok := p.hostreviews[host]
	if ok {
		tocheck = append(tocheck, hostreviews...)
	}
	if len(p.reviews) > 0 {
		tocheck = append(tocheck, p.reviews...)
	}

	// 这里使用k8s client-set去访问吧，直接传入解析出来的username
	for _, review := range tocheck {
		record, err := buildAttributeRecords(username, review)
		if err != nil {
			log.Printf("failed building request %s", err)
			return err
		}
		// 在这里调用delegate
		decision, reason, err := p.authorizer.Authorize(context.Background(), record)
		if err != nil {
			log.Printf("review user sar requst faild: %v", err)
			return err
		}
		if decision != authorizer.DecisionAllow {
			log.Printf("Permission denied for %s for check %s, the reasons are %s", username, review, reason)
			return providers.ErrPermissionDenied
		}
	}
	return nil
}

func buildAttributeRecords(username, resources string) (*authorizer.AttributesRecord, error) {
	var record authorizer.AttributesRecord
	var resourceAttributes authorizationv1.ResourceAttributes
	var nonResourceAttributes authorizationv1.NonResourceAttributes

	// parse ResourceAttributes and nonResourceAttributes
	if err := json.Unmarshal([]byte(resources), &resourceAttributes); err != nil {
		if err = json.Unmarshal([]byte(resources), &nonResourceAttributes); err != nil {
			return nil, fmt.Errorf("resources must be a JSON string of authorizationv1.ResourceAttributes or "+
				"authorizationv1.NonResourceAttributes: %v", err)
		}
		// build nonResourceAttributes
		record.Path = nonResourceAttributes.Path
		record.Verb = nonResourceAttributes.Verb
	} else {
		// build ResourceAttributes
		record.Name = resourceAttributes.Name
		record.Namespace = resourceAttributes.Namespace
		record.Verb = resourceAttributes.Verb
		record.APIVersion = resourceAttributes.Version
		record.APIGroup = resourceAttributes.Group
		record.Subresource = resourceAttributes.Subresource
		record.Resource = resourceAttributes.Resource
	}
	record.User = &user.DefaultInfo{Name: username}

	return &record, nil
}

// SessionFromCookie decodes the session from cookie
func (p *FuyaoProvider) SessionFromCookie(v string, c *cookie.Cipher) (*providers.SessionState, error) {
	session, err := providers.DecodeSessionState(v, c)
	if err != nil {
		log.Printf("cannot decode session from cookie")
		return nil, err
	}
	// fetch accessToken
	accessToken, err := p.getAccessTokenFromSessionID(session.SessionID)
	if err != nil {
		return nil, err
	}
	session.AccessToken = accessToken

	return session, nil
}

// getAccessTokenFromSessionID retrieves accessToken from sessionID in the backend
func (p *FuyaoProvider) getAccessTokenFromSessionID(sessionID string) (string, error) {
	sessionInfo, err := p.kubeClient.CoreV1().Secrets(p.sessionIDNamespace).Get(
		context.TODO(), sessionID, metav1.GetOptions{})
	if err != nil {
		log.Printf("cannot fetch sessionid secret")
		return "", err
	}
	accessTokenBytes, err := base64.StdEncoding.DecodeString(string(sessionInfo.Data["access-token"]))
	if err != nil {
		log.Printf("cannot decode token")
		return "", err
	}
	accessToken := string(accessTokenBytes)

	return accessToken, nil
}

// Redeem exchanges for the oauth token
func (p *FuyaoProvider) Redeem(redirectURL, logoutURL, code string) (s *providers.SessionState, err error) {
	if code == "" {
		err = errors.New("missing code")
		return
	}
	// prepare oauth2Config
	ctx := context.Background()
	config := oauth2.Config{
		ClientID:     p.ClientID,
		ClientSecret: p.ClientSecret,
		Endpoint: oauth2.Endpoint{
			TokenURL: p.RedeemURL.String(),
		},
		RedirectURL: redirectURL,
	}

	// generate sessionID
	sessionID, err := p.generateSessionID(constants.SessionIDLength)
	if err != nil {
		return nil, fmt.Errorf("error when generating sessionID: %v", err)
	}

	// add sessionID as extra param to oauth-server to perform single-logout
	sessionIDOption := oauth2.SetAuthURLParam(constants.SessionIDParam, sessionID)
	logoutEndpointOption := oauth2.SetAuthURLParam(constants.LogoutEndpointParam, logoutURL)
	token, err := config.Exchange(ctx, code, sessionIDOption, logoutEndpointOption)
	if err != nil {
		return nil, fmt.Errorf("error when exchanging code for token: %v", err)
	}

	// save sessionID -> token secret mapper then never return accessToken
	if err = p.saveSessionIDSecret(sessionID, token.AccessToken); err != nil {
		return
	}
	s = &providers.SessionState{SessionID: sessionID, AccessToken: token.AccessToken, ExpiresOn: token.Expiry}

	return
}

func (p *FuyaoProvider) saveSessionIDSecret(sessionID, accessToken string) error {
	// save the secret
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sessionID,
			Namespace: p.sessionIDNamespace,
		},
		Data: map[string][]byte{
			"access-token": []byte(base64.StdEncoding.EncodeToString([]byte(accessToken))),
		},
	}
	_, err := p.kubeClient.CoreV1().Secrets(p.sessionIDNamespace).Create(context.TODO(), secret, metav1.CreateOptions{})
	if err != nil {
		log.Printf("cannot save sessionID -> accessToken secret mapper")
		return err
	}
	return nil
}

// DeleteSession delete the sessionID <-> accessToken secret
func (p *FuyaoProvider) DeleteSession(sessionID string) error {
	// delete the sessionID<->accessToken secret
	err := p.kubeClient.CoreV1().Secrets(p.sessionIDNamespace).Delete(context.TODO(), sessionID, metav1.DeleteOptions{})
	if err != nil {
		// add error code here
		log.Printf("cannot delete session secret")
	}

	return nil
}

func (p *FuyaoProvider) generateSessionID(length int) (string, error) {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	sessionID := make([]byte, length)

	for {
		for i := 0; i < length; i++ {
			num, err := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
			if err != nil {
				log.Printf("Cannot generate random char, err: %v", err)
				return "", err
			}
			sessionID[i] = charset[num.Int64()]
		}

		// check the sessionID does not exists
		_, err := p.kubeClient.CoreV1().Secrets(p.sessionIDNamespace).Get(context.TODO(), string(sessionID),
			metav1.GetOptions{})
		if err != nil {
			break
		}
	}

	return string(sessionID), nil
}

// GetLoginURL with typical oauth parameters
func (p *FuyaoProvider) GetLoginURL(redirectURI, state string) string {
	var a url.URL
	a = *p.LoginURL
	params, err := url.ParseQuery(a.RawQuery)
	if err != nil {
		params = url.Values{}
	}
	params.Add("scope", p.Scope)
	params.Set("client_id", p.ClientID)
	params.Set("redirect_uri", redirectURI)
	params.Set("response_type", "code")
	params.Add("state", state)
	params.Set("identity_provider", "fuyaoPasswordProvider")
	a.RawQuery = params.Encode()
	return a.String()
}

func getKubeAPIURLWithPath(path string) *url.URL {
	ret := &url.URL{
		Scheme: "https",
		Host:   "kubernetes.default.svc",
		Path:   path,
	}

	if host := os.Getenv("KUBERNETES_SERVICE_HOST"); len(host) > 0 {
		// assume IPv6 if host contains colons
		if strings.IndexByte(host, ':') != -1 {
			host = "[" + host + "]"
		}
		ret.Host = host
	}

	return ret
}

// GetClientConfig get kube rest config from kubeconfigfile
func GetClientConfig(remoteKubeConfigFile string) (*rest.Config, error) {
	clientConfig := &rest.Config{}
	var err error
	if len(remoteKubeConfigFile) > 0 {
		loadingRules := &clientcmd.ClientConfigLoadingRules{ExplicitPath: remoteKubeConfigFile}
		loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

		clientConfig, err = loader.ClientConfig()

	} else {
		// without the remote kubeconfig file, try to use the in-cluster config.  Most addon API servers will
		// use this path
		clientConfig, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, err
	}

	// set high qps/burst limits since this will effectively limit API server responsiveness
	clientConfig.QPS = 200
	clientConfig.Burst = 400

	return clientConfig, nil
}
