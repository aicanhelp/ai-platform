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
 * type Option struct
 * func NewOptions()
 * func (o *Options) Validate
 * func (o *Options) validateProvider
 * functions added:
 * func ResolveOptions
 * func hasArg
 */

package main

import (
	"crypto"
	"crypto/tls"
	"encoding/base64"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"reflect"
	"regexp"
	"strings"
	"time"

	oscrypto "github.com/openshift/library-go/pkg/crypto"

	"openfuyao/oauth-proxy/constants"
	"openfuyao/oauth-proxy/providers"
	"openfuyao/oauth-proxy/providers/openfuyao"
)

const (
	// DefaultUpstreamTimeout is the maximum duration a network dial to a upstream server for a response.
	DefaultUpstreamTimeout = 600 * time.Second
)

// Configuration Options that can be set by Command Line Flag, or Config File
type Options struct {
	ProxyPrefix      string        `flag:"proxy-prefix" cfg:"proxy-prefix"`
	HttpAddress      string        `flag:"http-address" cfg:"http_address"`
	HttpsAddress     string        `flag:"https-address" cfg:"https_address"`
	UpstreamFlush    time.Duration `flag:"upstream-flush" cfg:"upstream_flush"`
	RedirectURL      string        `flag:"redirect-url" cfg:"redirect_url"`
	ClientID         string        `flag:"client-id" cfg:"client_id" env:"OAUTH2_PROXY_CLIENT_ID"`
	ClientSecret     string        `flag:"client-secret" cfg:"client_secret" env:"OAUTH2_PROXY_CLIENT_SECRET"`
	ClientSecretFile string        `flag:"client-secret-file" cfg:"client_secret_file" env:"OAUTH2_PROXY_CLIENT_SECRET_FILE"`
	TLSCertFile      string        `flag:"tls-cert" cfg:"tls_cert_file"`
	TLSKeyFile       string        `flag:"tls-key" cfg:"tls_key_file"`
	TLSClientCAFile  string        `flag:"tls-client-ca" cfg:"tls_client_ca"`

	AuthenticatedEmailsFile string   `flag:"authenticated-emails-file" cfg:"authenticated_emails_file"`
	EmailDomains            []string `flag:"email-domain" cfg:"email_domains"`
	CustomTemplatesDir      string   `flag:"custom-templates-dir" cfg:"custom_templates_dir"`
	Footer                  string   `flag:"footer" cfg:"footer"`

	OpenFuyaoSAR          string `flag:"openfuyao-sar" cfg:"openfuyao_sar"`
	OpenFuyaoSARByHost    string `flag:"openfuyao-sar-by-host" cfg:"openfuyao_sar_by_host"`
	OpenFuyaoDelegateURLs string `flag:"openfuyao-delegate-urls" cfg:"openfuyao_delegate_urls"`
	// RootPrefix is the endpoint where callback will redirect if oauth2 succeeds, need to mandatorily set
	RootPrefix string `flag:"root-prefix" cfg:"root-prefix"`

	CookieName       string        `flag:"cookie-name" cfg:"cookie_name" env:"OAUTH2_PROXY_COOKIE_NAME"`
	CookieSecret     string        `flag:"cookie-secret" cfg:"cookie_secret" env:"OAUTH2_PROXY_COOKIE_SECRET"`
	CookieSecretFile string        `flag:"cookie-secret-file" cfg:"cookie_secret_file" env:"OAUTH2_PROXY_COOKIE_SECRET_FILE"`
	CookieDomain     string        `flag:"cookie-domain" cfg:"cookie_domain" env:"OAUTH2_PROXY_COOKIE_DOMAIN"`
	CookieExpire     time.Duration `flag:"cookie-expire" cfg:"cookie_expire" env:"OAUTH2_PROXY_COOKIE_EXPIRE"`
	CookieRefresh    time.Duration `flag:"cookie-refresh" cfg:"cookie_refresh" env:"OAUTH2_PROXY_COOKIE_REFRESH"`
	CookieSecure     bool          `flag:"cookie-secure" cfg:"cookie_secure"`
	CookieHttpOnly   bool          `flag:"cookie-httponly" cfg:"cookie_httponly"`
	CookieSameSite   string        `flag:"cookie-samesite" cfg:"cookie_samesite" env:"OAUTH2_PROXY_COOKIE_SAMESITE"`

	Upstreams             []string `flag:"upstream" cfg:"upstreams"`
	BypassAuthExceptRegex []string `flag:"bypass-auth-except-for" cfg:"bypass_auth_except_for"`
	SkipAuthRegex         []string `flag:"skip-auth-regex" cfg:"skip_auth_regex"`
	PassBasicAuth         bool     `flag:"pass-basic-auth" cfg:"pass_basic_auth"`
	BasicAuthPassword     string   `flag:"basic-auth-password" cfg:"basic_auth_password"`
	PassAccessToken       bool     `flag:"pass-access-token" cfg:"pass_access_token"`
	PassHostHeader        bool     `flag:"pass-host-header" cfg:"pass_host_header"`
	PassUserHeaders       bool     `flag:"pass-user-headers" cfg:"pass_user_headers"`
	SSLInsecureSkipVerify bool     `flag:"ssl-insecure-skip-verify" cfg:"ssl_insecure_skip_verify"`
	SetXAuthRequest       bool     `flag:"set-xauthrequest" cfg:"set_xauthrequest"`
	SkipAuthPreflight     bool     `flag:"skip-auth-preflight" cfg:"skip_auth_preflight"`

	// These options allow for other providers besides Google, with
	// potential overrides.
	Provider       string `flag:"provider" cfg:"provider"`
	LoginURL       string `flag:"login-url" cfg:"login_url"`   // 要提供
	RedeemURL      string `flag:"redeem-url" cfg:"redeem_url"` // 要提供
	ProfileURL     string `flag:"profile-url" cfg:"profile_url"`
	ValidateURL    string `flag:"validate-url" cfg:"validate_url"`
	Scope          string `flag:"scope" cfg:"scope"`
	RequestLogging bool   `flag:"request-logging" cfg:"request_logging"`

	SignatureKey string   `flag:"signature-key" cfg:"signature_key" env:"OAUTH2_PROXY_SIGNATURE_KEY"`
	UpstreamCAs  []string `flag:"upstream-ca" cfg:"upstream_ca"`

	// An optional, absolute URL to redirect web browsers to after logging out of
	// the console. If not specified, it will redirect to the default login page.
	// This is required when using an identity provider that supports single
	// sign-on (SSO) such as:
	// - OpenID (Keycloak, Azure)
	// - RequestHeader (GSSAPI, SSPI, SAML)
	// - OAuth (GitHub, GitLab, Google)
	// Logging out of the console will destroy the user's token. The logoutRedirect
	// provides the user the option to perform single logout (SLO) through the identity
	// provider to destroy their single sign-on session.
	LogoutRedirectURL string `flag:"logout-redirect-url" cfg:"logout_redirect_url"` // 这个应该设置为登陆的主页

	// Timeout is the maximum duration the server will wait for a response from the upstream server.
	// Defaults to 30 seconds.
	Timeout time.Duration `flag:"upstream-timeout" cfg:"upstream_timeout"`

	// internal values that are set after config validation
	proxyBaseURL      *url.URL
	proxyURLs         []*url.URL
	CompiledAuthRegex []*regexp.Regexp
	CompiledSkipRegex []*regexp.Regexp
	provider          providers.Provider
	signatureData     *SignatureData
}

type SignatureData struct {
	hash crypto.Hash
	key  string
}

func NewOptions() *Options {
	return &Options{
		ProxyPrefix:       "/oauth",
		HttpAddress:       "127.0.0.1:4180",
		HttpsAddress:      ":443",
		UpstreamFlush:     time.Duration(constants.DefaultUpstreamFlush) * time.Millisecond,
		CookieName:        "_oauth2_proxy",
		CookieSecure:      true,
		CookieHttpOnly:    true,
		CookieSameSite:    "strict",
		CookieExpire:      time.Duration(constants.DefaultCookieExpireMin) * time.Minute,
		CookieRefresh:     time.Duration(0),
		SetXAuthRequest:   false,
		SkipAuthPreflight: false,
		PassBasicAuth:     true,
		PassUserHeaders:   true,
		PassAccessToken:   false,
		PassHostHeader:    true,
		RequestLogging:    true,
		Timeout:           DefaultUpstreamTimeout,
	}
}

func parseURL(to_parse string, urltype string, msgs []string) (*url.URL, []string) {
	parsed, err := url.Parse(to_parse)
	if err != nil {
		return nil, append(msgs, fmt.Sprintf(
			"error parsing %s-url=%q %s", urltype, to_parse, err))
	}
	return parsed, msgs
}

func (o *Options) Validate(p providers.Provider) error {
	msgs := make([]string, 0)

	// allow the provider to default some values
	switch provider := p.(type) {
	case *openfuyao.FuyaoProvider:
		defaults, err := provider.LoadDefaults(o.OpenFuyaoSAR, o.OpenFuyaoSARByHost, o.OpenFuyaoDelegateURLs)
		if err != nil {
			return err
		}
		if len(o.ClientID) == 0 {
			o.ClientID = defaults.ClientID
		}
		if len(o.ClientSecret) == 0 {
			o.ClientSecret = defaults.ClientSecret
		}
		if len(o.Scope) == 0 {
			o.Scope = defaults.Scope
		}
		if len(o.ValidateURL) == 0 && defaults.ValidateURL != nil {
			o.ValidateURL = defaults.ValidateURL.String()
		}
		if len(o.EmailDomains) == 0 {
			o.EmailDomains = []string{"*"}
		}
		if len(o.RedirectURL) == 0 {
			o.RedirectURL = "https:///"
		}
	}

	if o.CookieSecretFile != "" {
		if contents, err := ioutil.ReadFile(o.CookieSecretFile); err != nil {
			msgs = append(msgs, fmt.Sprintf("cannot read cookie-secret-file: %v", err))
		} else {
			o.CookieSecret = string(contents)
		}
	}
	if o.ClientSecretFile != "" {
		if contents, err := ioutil.ReadFile(o.ClientSecretFile); err != nil {
			msgs = append(msgs, fmt.Sprintf("cannot read client-secret-file: %v", err))
		} else {
			o.ClientSecret = string(contents)
		}
	}

	if len(o.Upstreams) < 1 {
		msgs = append(msgs, "missing setting: upstream")
	}
	if o.ClientID == "" {
		msgs = append(msgs, "missing setting: client-id")
	}
	if o.ClientSecret == "" {
		msgs = append(msgs, "missing setting: client-secret")
	}
	if o.AuthenticatedEmailsFile == "" && len(o.EmailDomains) == 0 {
		msgs = append(msgs, "missing setting for email validation: email-domain or authenticated-emails-file required.\n      use email-domain=* to authorize all email addresses")
	}

	o.proxyBaseURL, msgs = parseURL(o.RedirectURL, "redirect", msgs)

	o.proxyURLs = nil
	for _, u := range o.Upstreams {
		upstreamURL, err := url.Parse(u)
		if err != nil {
			msgs = append(msgs, fmt.Sprintf(
				"error parsing upstream=%q %s",
				upstreamURL, err))
			continue
		}
		if upstreamURL.Path == "" {
			upstreamURL.Path = "/"
		}
		o.proxyURLs = append(o.proxyURLs, upstreamURL)
	}
	if len(o.BypassAuthExceptRegex) != 0 && len(o.SkipAuthRegex) != 0 {
		msgs = append(msgs, "error: cannot set -skip-auth-regex and -bypass-auth-except-for together")
	}

	for _, u := range o.BypassAuthExceptRegex {
		CompiledRegex, err := regexp.Compile(u)
		if err != nil {
			msgs = append(msgs, fmt.Sprintf(
				"error compiling regex=%q %s", u, err))
		}
		o.CompiledAuthRegex = append(o.CompiledAuthRegex, CompiledRegex)
	}

	// Ensure paths under ProxyPrefix are still protected when using -bypass-auth-except-for
	if len(o.CompiledAuthRegex) > 0 {
		proxyRegex, err := regexp.Compile(o.ProxyPrefix + "*")
		if err != nil {
			msgs = append(msgs, fmt.Sprintf(
				"error compiling regex=%q %s", o.ProxyPrefix+"*", err))
		}
		o.CompiledAuthRegex = append(o.CompiledAuthRegex, proxyRegex)
	}

	for _, u := range o.SkipAuthRegex {
		CompiledRegex, err := regexp.Compile(u)
		if err != nil {
			msgs = append(msgs, fmt.Sprintf(
				"error compiling regex=%q %s", u, err))
		}
		o.CompiledSkipRegex = append(o.CompiledSkipRegex, CompiledRegex)
	}

	if o.PassAccessToken || (o.CookieRefresh != time.Duration(0)) {
		valid_cookie_secret_size := false
		for _, i := range []int{16, 24, 32} {
			if len(secretBytes(o.CookieSecret)) == i {
				valid_cookie_secret_size = true
			}
		}
		var decoded bool
		if string(secretBytes(o.CookieSecret)) != o.CookieSecret {
			decoded = true
		}
		if valid_cookie_secret_size == false {
			var suffix string
			if decoded {
				suffix = fmt.Sprintf(" note: cookie secret was base64 decoded from %q", o.CookieSecret)
			}
			msgs = append(msgs, fmt.Sprintf(
				"cookie_secret must be 16, 24, or 32 bytes "+
					"to create an AES cipher when "+
					"pass_access_token == true or "+
					"cookie_refresh != 0, but is %d bytes.%s",
				len(secretBytes(o.CookieSecret)), suffix))
		}
	}

	if o.CookieRefresh >= o.CookieExpire {
		msgs = append(msgs, fmt.Sprintf(
			"cookie_refresh (%s) must be less than "+
				"cookie_expire (%s)",
			o.CookieRefresh.String(),
			o.CookieExpire.String()))
	}

	if len(o.TLSClientCAFile) > 0 && len(o.TLSKeyFile) == 0 && len(o.TLSCertFile) == 0 {
		msgs = append(msgs, "tls-client-ca requires tls-key-file or tls-cert-file to be set to listen on tls")
	}

	switch o.CookieSameSite {
	case "", "none", "lax", "strict":
	default:
		msgs = append(msgs, fmt.Sprintf("cookie_samesite (%q) must be one of ['', 'lax', 'strict', 'none']", o.CookieSameSite))
	}

	msgs = validateCookieName(o, msgs)

	if o.SSLInsecureSkipVerify {
		insecureTransport := &http.Transport{
			TLSClientConfig: oscrypto.SecureTLSConfig(
				&tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS12}),
		}
		http.DefaultClient = &http.Client{Transport: insecureTransport}
	}

	msgs = append(msgs, o.validateProvider(p)...)
	if len(msgs) != 0 {
		return fmt.Errorf("Invalid configuration:\n  %s",
			strings.Join(msgs, "\n  "))
	}
	o.provider = p

	return nil
}

func (o *Options) validateProvider(provider providers.Provider) []string {
	var msgs []string
	data := &providers.ProviderData{
		Scope:        o.Scope,
		ClientID:     o.ClientID,
		ClientSecret: o.ClientSecret,
	}
	data.LoginURL, msgs = parseURL(o.LoginURL, "login", msgs)
	data.RedeemURL, msgs = parseURL(o.RedeemURL, "redeem", msgs)

	data.ProfileURL, msgs = parseURL(o.ProfileURL, "profile", msgs)
	data.ValidateURL, msgs = parseURL(o.ValidateURL, "validate", msgs)
	if len(msgs) != 0 {
		return msgs
	}

	switch p := provider.(type) {
	case *openfuyao.FuyaoProvider:
		if err := p.Complete(data); err != nil {
			msgs = append(msgs, fmt.Sprintf("unable to load openfuyao configuration: %v", err))
		}
	case *providers.ProviderData:
		p.Scope = data.Scope
		p.ClientID = data.ClientID
		p.ClientSecret = data.ClientSecret
		p.LoginURL = data.LoginURL
		p.RedeemURL = data.RedeemURL
		p.ProfileURL = data.ProfileURL
		p.ValidateURL = data.ValidateURL
	}
	return msgs
}

func validateCookieName(o *Options, msgs []string) []string {
	cookie := &http.Cookie{Name: o.CookieName}
	if cookie.String() == "" {
		return append(msgs, fmt.Sprintf("invalid cookie name: %q", o.CookieName))
	}
	return msgs
}

func addPadding(secret string) string {
	padding := len(secret) % 4
	switch padding {
	case 1:
		return secret + "==="
	case 2:
		return secret + "=="
	case 3:
		return secret + "="
	default:
		return secret
	}
}

// secretBytes attempts to base64 decode the secret, if that fails it treats the secret as binary
func secretBytes(secret string) []byte {
	b, err := base64.URLEncoding.DecodeString(addPadding(secret))
	if err == nil {
		return []byte(addPadding(string(b)))
	}
	return []byte(secret)
}

// ResolveOptions maps the parameters in flagset and cfg to options
func ResolveOptions(options interface{}, flagSet *flag.FlagSet, cfg map[string]interface{}) {
	val := reflect.ValueOf(options).Elem()
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		// pull out the struct tags:
		//    flag - the name of the command line flag
		//    deprecated - (optional) the name of the deprecated command line flag
		//    cfg - (optional, defaults to underscored flag) the name of the config file option
		field := typ.Field(i)

		// Recursively resolve embedded types.
		if field.Anonymous {
			var fieldPtr reflect.Value
			switch val.FieldByName(field.Name).Kind() {
			case reflect.Struct:
				fieldPtr = val.FieldByName(field.Name).Addr()
			case reflect.Ptr:
				fieldPtr = reflect.Indirect(val).FieldByName(field.Name)
			default:
			}
			if !fieldPtr.IsNil() {
				ResolveOptions(fieldPtr.Interface(), flagSet, cfg)
			}
		}

		flagName := field.Tag.Get("flag")
		cfgName := field.Tag.Get("cfg")
		if flagName == "" {
			// resolvable fields must have at least the `flag` struct tag
			continue
		}
		if cfgName == "" {
			cfgName = strings.Replace(flagName, "-", "_", -1)
		}

		// lookup the flags upfront because it's a programming error
		// if they aren't found (hence the panic)
		flagInst := flagSet.Lookup(flagName)
		if flagInst == nil {
			log.Panicf("ERROR: flag %q does not exist", flagName)
		}
		// resolve the flags according to priority
		var v interface{}
		if hasArg(flagSet, flagName) {
			v = flagInst.Value.(flag.Getter).Get()
		} else if cfgVal, ok := cfg[cfgName]; ok && !isNilInterface(cfgVal) {
			v = cfgVal
		} else if getter, ok := flagInst.Value.(flag.Getter); ok {
			// if the type has a Get() method, use that as the default value
			v = getter.Get()
		} else {
			// otherwise, use the struct's default value
			v = val.Field(i).Interface()
		}

		fieldVal := val.FieldByName(field.Name)
		if fieldVal.Type() != reflect.TypeOf(v) {
			log.Fatalf("ERROR: Resolve failed to coerce value %v (%+v) for field %s",
				v, fieldVal, field.Name)
		}
		fieldVal.Set(reflect.ValueOf(v))
	}
}

func hasArg(fs *flag.FlagSet, s string) bool {
	var found bool
	fs.Visit(func(flag *flag.Flag) {
		if flag.Name == s {
			found = true
		}
	})
	return found
}

// isNilInterface reports whether v is a nil interface or an interface holding a
// typed nil value (e.g. (*T)(nil) wrapped in an interface). A plain `v == nil`
// check does not catch the typed-nil case, so reflect is used to inspect it.
// This avoids comparing an interface value to nil directly, which would be
// unreliable for types whose zero value is not the empty interface.
func isNilInterface(v interface{}) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.Interface:
		return rv.IsNil()
	default:
		return false
	}
}
