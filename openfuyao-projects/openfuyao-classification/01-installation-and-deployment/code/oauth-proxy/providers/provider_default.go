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
 * func (p *ProviderData) Redeem
 * functions added:
 * func (p *ProviderData) prepareRedeemParams
 * func (p *ProviderData) ValidateMultiClusterRequest
 * func (p *ProviderData) SetRedeemURL
 * func (p *ProviderData) SetLoginURL
 */

package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"

	"openfuyao/oauth-proxy/cookie"
)

// Redeem exchanges for the oauth token
func (p *ProviderData) Redeem(redirectURL, logoutURL, code string) (s *SessionState, err error) {
	if code == "" {
		err = errors.New("missing code")
		return
	}

	params := p.prepareRedeemParams(redirectURL, code)

	var req *http.Request
	req, err = http.NewRequest("POST", p.RedeemURL.String(), bytes.NewBufferString(params.Encode()))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	var resp *http.Response
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	var body []byte
	body, err = ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return
	}

	if resp.StatusCode != 200 {
		err = fmt.Errorf("got %d from %q %s", resp.StatusCode, p.RedeemURL.String(), body)
		return
	}

	// blindly try json and x-www-form-urlencoded
	var jsonResponse struct {
		AccessToken string `json:"access_token"`
	}
	err = json.Unmarshal(body, &jsonResponse)
	if err == nil {
		s = &SessionState{
			AccessToken: jsonResponse.AccessToken,
		}
		return
	}

	var v url.Values
	v, err = url.ParseQuery(string(body))
	if err != nil {
		return
	}
	if a := v.Get("access_token"); a != "" {
		s = &SessionState{AccessToken: a}
	} else {
		err = fmt.Errorf("no access token found %s", body)
	}
	return
}

func (p *ProviderData) prepareRedeemParams(redirectURL string, code string) url.Values {
	params := url.Values{}
	params.Add("redirect_uri", redirectURL)
	params.Add("client_id", p.ClientID)
	params.Add("client_secret", p.ClientSecret)
	params.Add("code", code)
	params.Add("grant_type", "authorization_code")
	if p.ProtectedResource != nil && p.ProtectedResource.String() != "" {
		params.Add("resource", p.ProtectedResource.String())
	}
	return params
}

// GetLoginURL with typical oauth parameters
func (p *ProviderData) GetLoginURL(redirectURI, state string) string {
	var a url.URL
	a = *p.LoginURL
	params, err := url.ParseQuery(a.RawQuery)
	if err != nil {
		params = url.Values{}
	}
	params.Set("redirect_uri", redirectURI)
	params.Add("scope", p.Scope)
	params.Set("client_id", p.ClientID)
	params.Set("response_type", "code")
	params.Add("state", state)
	a.RawQuery = params.Encode()
	return a.String()
}

// CookieForSession serializes a session state for storage in a cookie
func (p *ProviderData) CookieForSession(s *SessionState, c *cookie.Cipher) (string, error) {
	return s.EncodeSessionState(c)
}

// SessionFromCookie deserializes a session from a cookie value
func (p *ProviderData) SessionFromCookie(v string, c *cookie.Cipher) (s *SessionState, err error) {
	return DecodeSessionState(v, c)
}

// GetEmailAddress abstract func
func (p *ProviderData) GetEmailAddress(s *SessionState) (string, error) {
	return "", errors.New("not implemented")
}

// GetUserName abstract func
func (p *ProviderData) GetUserName(accessToken string) (string, error) {
	return "", errors.New("not implemented")
}

// ReviewUser reviews the RBAC of the current user
func (p *ProviderData) ReviewUser(name, host string) error {
	return nil
}

// DeleteSession delete the sessionID <-> accessToken secret
func (p *ProviderData) DeleteSession(sessionID string) error {
	return nil
}

// ValidateGroup validates that the provided email exists in the configured provider
// email group(s).
func (p *ProviderData) ValidateGroup(email string) bool {
	return true
}

// ValidateRequest abstract func
func (p *ProviderData) ValidateRequest(_ *http.Request) (*SessionState, error) {
	return nil, nil
}

// ValidateMultiClusterRequest abstruct func
func (p *ProviderData) ValidateMultiClusterRequest(*http.Request) (*SessionState, error) {
	return nil, nil
}

// ValidateSessionState validate the sessionState in default provider
func (p *ProviderData) ValidateSessionState(s *SessionState) bool {
	return validateToken(p, s.AccessToken, nil)
}

// ValidateSessionAuth abstract func
func (p *ProviderData) ValidateSessionAuth(_ *SessionState, _ *http.Request) (bool, error) {
	return false, fmt.Errorf("session auth is not implemented")
}

// RefreshSessionIfNeeded abstract func
func (p *ProviderData) RefreshSessionIfNeeded(s *SessionState) (bool, error) {
	return false, nil
}

// SetRedeemURL sets the redeem url
func (p *ProviderData) SetRedeemURL(req *http.Request, scheme string) {
	p.RedeemURL.Host = req.Host
	p.RedeemURL.Scheme = scheme
}

// SetLoginURL sets the oauth/authorize endpoint
func (p *ProviderData) SetLoginURL(req *http.Request, scheme string) {
	p.LoginURL.Host = req.Host
	p.LoginURL.Scheme = scheme
}
