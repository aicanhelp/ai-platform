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
 * func initOAuthProxy
 * func NewOAuthProxy
 * func (p *OAuthProxy) ErrorResponse
 * func (p *OAuthProxy) OAuthStart
 * func (p *OAuthProxy) Proxy
 * func (p *OAuthProxy) Authenticate
 * functions added:
 * func prepareProxy
 * func matchPathByRe
 * func (p *OAuthProxy) GetLogoutURI
 * func (p *OAuthProxy) MakeSessionCookie  MakeSessionCookie generate the encrypted cookie
 * func (p *OAuthProxy) MakeCSRFCookie  MakeCSRFCookie generates the encrypted csrf cookie
 * func (p *OAuthProxy) updateOAuth2URL
 * func (p *OAuthProxy) LogOut
 * func (p *OAuthProxy) getNonceAndRedirect
 * func accessLog
 */

package main

import (
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/openshift/library-go/pkg/crypto"
	"golang.org/x/net/http2"

	"openfuyao/oauth-proxy/constants"
	"openfuyao/oauth-proxy/cookie"
	"openfuyao/oauth-proxy/fuyaoerrors"
	"openfuyao/oauth-proxy/providers"
	"openfuyao/oauth-proxy/util"
)

const SignatureHeader = "GAP-Signature"

var SignatureHeaders []string = []string{
	"Content-Length",
	"Content-Md5",
	"Content-Type",
	"Date",
	"Authorization",
	"X-Forwarded-User",
	"X-Forwarded-Email",
	"X-Forwarded-Access-Token",
	"Cookie",
	"Gap-Auth",
}

type OAuthProxy struct {
	CookieSeed     string
	CookieName     string
	CSRFCookieName string
	CookieDomain   string
	CookieSecure   bool
	CookieHttpOnly bool
	CookieExpire   time.Duration
	CookieRefresh  time.Duration
	CookieSameSite string
	Validator      func(string) bool

	RobotsPath        string
	PingPath          string
	SignOutPath       string
	LogOutPath        string
	OAuthStartPath    string
	OAuthCallbackPath string

	LogoutRedirectURL string

	redirectURL       *url.URL // the url to receive requests at
	logoutURL         *url.URL
	rootPrefix        string
	provider          providers.Provider
	ProxyPrefix       string
	SignInMessage     string
	serveMux          http.Handler
	SetXAuthRequest   bool
	PassBasicAuth     bool
	PassUserHeaders   bool
	BasicAuthPassword string
	PassAccessToken   bool
	CookieCipher      *cookie.Cipher
	authRegex         []string
	skipAuthRegex     []string
	skipAuthPreflight bool
	compiledAuthRegex []*regexp.Regexp
	compiledSkipRegex []*regexp.Regexp
	Footer            string
}

type UpstreamProxy struct {
	upstream string
	handler  http.Handler
}

func (u *UpstreamProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("GAP-Upstream-Address", u.upstream)
	u.handler.ServeHTTP(w, r)
}

func NewReverseProxy(target *url.URL, opts *Options) (*httputil.ReverseProxy, error) {
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = opts.UpstreamFlush

	// Inherit default transport options from Go's stdlib
	transport := http.DefaultTransport.(*http.Transport).Clone()

	transport.MaxIdleConnsPerHost = 500
	transport.IdleConnTimeout = 1 * time.Minute

	// Change default duration for waiting for an upstream response
	transport.ResponseHeaderTimeout = opts.Timeout

	if len(opts.UpstreamCAs) > 0 {
		pool, err := util.GetCertPool(opts.UpstreamCAs, false)
		if err != nil {
			return nil, err
		}
		transport.TLSClientConfig = crypto.SecureTLSConfig(&tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12})
	}
	if err := http2.ConfigureTransport(transport); err != nil {
		if len(opts.UpstreamCAs) > 0 {
			return nil, err
		}
		log.Printf("WARN: Failed to configure http2 transport: %v", err)
	}

	// Apply the customized transport to our proxy before returning it
	proxy.Transport = transport

	return proxy, nil
}

func setProxyUpstreamHostHeader(proxy *httputil.ReverseProxy, target *url.URL) {
	director := proxy.Director
	proxy.Director = func(req *http.Request) {
		director(req)
		// use RequestURI so that we aren't unescaping encoded slashes in the request path
		req.Host = target.Host
		req.URL.Opaque = req.RequestURI
		req.URL.RawQuery = ""
	}
}
func setProxyDirector(proxy *httputil.ReverseProxy) {
	director := proxy.Director
	proxy.Director = func(req *http.Request) {
		director(req)
		// use RequestURI so that we aren't unescaping encoded slashes in the request path
		req.URL.Opaque = req.RequestURI
		req.URL.RawQuery = ""
	}
}
func NewFileServer(path string, filesystemPath string) (proxy http.Handler) {
	return http.StripPrefix(path, http.FileServer(http.Dir(filesystemPath)))
}

func NewWebSocketOrRestReverseProxy(u *url.URL, opts *Options) http.Handler {
	u.Path = ""
	proxy, err := NewReverseProxy(u, opts)
	if err != nil {
		log.Fatal("Failed to initialize Reverse Proxy: ", err)
	}
	if !opts.PassHostHeader {
		setProxyUpstreamHostHeader(proxy, u)
	} else {
		setProxyDirector(proxy)
	}

	return &UpstreamProxy{u.Host, proxy}
}

func NewOAuthProxy(opts *Options, validator func(string) bool) *OAuthProxy {
	serveMux := http.NewServeMux()
	prepareProxy(opts, serveMux)

	for _, u := range opts.CompiledAuthRegex {
		log.Printf("compiled auth-regex => %q", u)
	}

	for _, u := range opts.CompiledSkipRegex {
		log.Printf("compiled skip-auth-regex => %q", u)
	}

	log.Printf("OAuthProxy configured for %s Client ID: %s", opts.provider.Data().ProviderName, opts.ClientID)
	domain := opts.CookieDomain
	if domain == "" {
		domain = "<default>"
	}
	refresh := "disabled"
	if opts.CookieRefresh != time.Duration(0) {
		refresh = fmt.Sprintf("after %s", opts.CookieRefresh)
	}

	log.Printf("Cookie settings: name:%s secure(https):%v httponly:%v expiry:%s domain:%s samesite:%s refresh:%s",
		opts.CookieName, opts.CookieSecure, opts.CookieHttpOnly, opts.CookieExpire, domain, opts.CookieSameSite, refresh)

	var cipher *cookie.Cipher
	if opts.PassAccessToken || (opts.CookieRefresh != time.Duration(0)) {
		var err error
		cipher, err = cookie.NewCipher(secretBytes(opts.CookieSecret))
		if err != nil {
			log.Fatal("cookie-secret error: ", err)
		}
	}

	if opts.PassAccessToken {
		log.Printf("WARN: Configured to pass client specified bearer token upstream.\n" +
			"      Only use this option if you're sure that the upstream will not leak or abuse the given token.\n" +
			"      Bear in mind that the token could be a long lived token or hard to revoke.")
	}

	return initOAuthProxy(opts, validator, serveMux, cipher)
}

func initOAuthProxy(
	opts *Options,
	validator func(string) bool,
	serveMux *http.ServeMux,
	cipher *cookie.Cipher,
) *OAuthProxy {
	// prepare oauth-proxy self-urls
	redirectURL := &url.URL{}
	logoutURL := &url.URL{}
	*redirectURL = *opts.proxyBaseURL
	*logoutURL = *opts.proxyBaseURL
	// OAuth2.0 callback endpoint
	redirectURL.Path = path.Join(redirectURL.Path, opts.RootPrefix, opts.ProxyPrefix, "callback")
	// oauth-proxy logout url for openfuyao
	logoutURL.Path = path.Join(logoutURL.Path, opts.RootPrefix, opts.ProxyPrefix, "logout")

	return &OAuthProxy{
		CookieName:     opts.CookieName,
		CSRFCookieName: fmt.Sprintf("%v_%v", opts.CookieName, "csrf"),
		CookieSeed:     opts.CookieSecret,
		CookieDomain:   opts.CookieDomain,
		CookieSecure:   opts.CookieSecure,
		CookieHttpOnly: opts.CookieHttpOnly,
		CookieExpire:   opts.CookieExpire,
		CookieRefresh:  opts.CookieRefresh,
		CookieSameSite: opts.CookieSameSite,
		Validator:      validator,

		RobotsPath:        "/robots.txt",
		PingPath:          fmt.Sprintf("%s/healthz", opts.ProxyPrefix),
		SignOutPath:       fmt.Sprintf("%s/sign_out", opts.ProxyPrefix),
		LogOutPath:        fmt.Sprintf("%s/logout", opts.ProxyPrefix),
		OAuthStartPath:    fmt.Sprintf("%s/start", opts.ProxyPrefix),
		OAuthCallbackPath: fmt.Sprintf("%s/callback", opts.ProxyPrefix),

		LogoutRedirectURL: opts.LogoutRedirectURL,

		ProxyPrefix:       opts.ProxyPrefix,
		provider:          opts.provider,
		serveMux:          serveMux,
		redirectURL:       redirectURL,
		logoutURL:         logoutURL,
		rootPrefix:        opts.RootPrefix,
		authRegex:         opts.BypassAuthExceptRegex,
		skipAuthRegex:     opts.SkipAuthRegex,
		skipAuthPreflight: opts.SkipAuthPreflight,
		compiledAuthRegex: opts.CompiledAuthRegex,
		compiledSkipRegex: opts.CompiledSkipRegex,
		SetXAuthRequest:   opts.SetXAuthRequest,
		PassBasicAuth:     opts.PassBasicAuth,
		PassUserHeaders:   opts.PassUserHeaders,
		BasicAuthPassword: opts.BasicAuthPassword,
		PassAccessToken:   opts.PassAccessToken,
		CookieCipher:      cipher,
		Footer:            opts.Footer,
	}
}

func prepareProxy(opts *Options, serveMux *http.ServeMux) {
	for _, u := range opts.proxyURLs {
		path := u.Path
		switch u.Scheme {
		case "http", "https":
			log.Printf("mapping path %q => upstream %q", path, u)
			proxy := NewWebSocketOrRestReverseProxy(u, opts)
			serveMux.Handle(path, proxy)

		case "file":
			if u.Fragment != "" {
				path = u.Fragment
			}
			log.Printf("mapping path %q => file system %q", path, u.Path)
			proxy := NewFileServer(path, u.Path)
			serveMux.Handle(path, &UpstreamProxy{upstream: path, handler: proxy})
		default:
			panic(fmt.Sprintf("unknown upstream protocol %s", u.Scheme))
		}
	}
}

func (p *OAuthProxy) GetRedirectURI(host string) string {
	// default to the request Host if not set
	if p.redirectURL.Host != "" {
		return p.redirectURL.String()
	}
	var u url.URL
	u = *p.redirectURL
	if u.Scheme == "" {
		if p.CookieSecure {
			u.Scheme = "https"
		} else {
			u.Scheme = "http"
		}
	}
	u.Host = host
	return u.String()
}

// GetLogoutURI returns the oauth-proxy logout endpoint
func (p *OAuthProxy) GetLogoutURI(host string) string {
	if p.logoutURL.Host != "" {
		return p.logoutURL.String()
	}
	var u url.URL
	u = *p.logoutURL
	if u.Scheme == "" {
		if p.CookieSecure {
			u.Scheme = "https"
		} else {
			u.Scheme = "http"
		}
	}
	u.Host = host
	u.Path = path.Join("/", p.rootPrefix, p.LogOutPath)
	return u.String()
}

func (p *OAuthProxy) redeemCode(host, code string) (s *providers.SessionState, err error) {
	if code == "" {
		return nil, errors.New("missing code")
	}
	redirectURI := p.GetRedirectURI(host)
	logoutURI := p.GetLogoutURI(host)

	if err != nil {
		return s, err
	}
	s, err = p.provider.Redeem(redirectURI, logoutURI, code)
	if err != nil {
		return
	}

	if s.Email == "" {
		s.Email, err = p.provider.GetEmailAddress(s)
		if err != nil {
			return
		}
	}
	username, err := p.provider.GetUserName(s.AccessToken)
	if err != nil {
		return nil, err
	}
	err = p.provider.ReviewUser(username, host)
	if err != nil {
		return nil, err
	}

	// eliminate access_token
	s.AccessToken = ""
	return
}

// MakeSessionCookie generate the encrypted cookie
func (p *OAuthProxy) MakeSessionCookie(
	req *http.Request,
	value string,
	expiration time.Duration,
	now time.Time,
) *http.Cookie {
	rawValue := cookie.RawValue(value, now)
	return p.makeCookie(req, p.CookieName, rawValue, expiration, now)
}

// MakeCSRFCookie generates the encrypted csrf cookie
func (p *OAuthProxy) MakeCSRFCookie(
	req *http.Request,
	value string,
	expiration time.Duration,
	now time.Time,
) *http.Cookie {
	return p.makeCookie(req, p.CSRFCookieName, value, expiration, now)
}

func (p *OAuthProxy) makeCookie(
	req *http.Request,
	name string,
	value string,
	expiration time.Duration,
	now time.Time,
) *http.Cookie {
	domain := req.Host
	if h, _, err := net.SplitHostPort(domain); err == nil {
		domain = h
	}
	if p.CookieDomain != "" {
		if !strings.HasSuffix(domain, p.CookieDomain) {
			log.Printf("Warning: request host is %q but using configured cookie domain of %q", domain, p.CookieDomain)
		}
		domain = p.CookieDomain
	}

	return &http.Cookie{
		Name:     name,
		Value:    value,
		Path:     "/",
		Domain:   domain,
		HttpOnly: p.CookieHttpOnly,
		Secure:   p.CookieSecure,
		Expires:  now.Add(expiration),
		SameSite: parseSameSite(p.CookieSameSite),
	}
}

func (p *OAuthProxy) ClearCSRFCookie(rw http.ResponseWriter, req *http.Request) {
	http.SetCookie(rw, p.MakeCSRFCookie(req, "", time.Hour*-1, time.Now()))
}

func (p *OAuthProxy) SetCSRFCookie(rw http.ResponseWriter, req *http.Request, val string) {
	http.SetCookie(rw, p.MakeCSRFCookie(req, val, p.CookieExpire, time.Now()))
}

func (p *OAuthProxy) ClearSessionCookie(rw http.ResponseWriter, req *http.Request) {
	http.SetCookie(rw, p.MakeSessionCookie(req, "", time.Hour*-1, time.Now()))
}

func (p *OAuthProxy) SetSessionCookie(rw http.ResponseWriter, req *http.Request, val string) {
	http.SetCookie(rw, p.MakeSessionCookie(req, val, p.CookieExpire, time.Now()))
}

func (p *OAuthProxy) LoadCookiedSession(req *http.Request) (*providers.SessionState, time.Duration, error) {
	var age time.Duration
	c, err := req.Cookie(p.CookieName)
	if err != nil {
		// always http.ErrNoCookie
		return nil, age, err
	}
	val, timestamp, ok := cookie.Fetch(c, fmt.Sprintf("%s%s", p.CookieSeed, req.Host), p.CookieExpire)
	if !ok {
		return nil, age, errors.New("Cookie Signature not valid")
	}

	session, err := p.provider.SessionFromCookie(val, p.CookieCipher)
	if err != nil {
		return nil, age, err
	}

	age = time.Now().Truncate(time.Second).Sub(timestamp)
	return session, age, nil
}

func (p *OAuthProxy) SaveSession(rw http.ResponseWriter, req *http.Request, s *providers.SessionState) error {
	value, err := p.provider.CookieForSession(s, p.CookieCipher)
	if err != nil {
		return err
	}
	p.SetSessionCookie(rw, req, value)
	return nil
}

func (p *OAuthProxy) RobotsTxt(rw http.ResponseWriter) {
	rw.WriteHeader(http.StatusOK)
	fmt.Fprintf(rw, "User-agent: *\nDisallow: /")
}

func (p *OAuthProxy) PingPage(rw http.ResponseWriter) {
	rw.WriteHeader(http.StatusOK)
	fmt.Fprintf(rw, "OK")
}

// ErrorResponse respond the error through http response
func (p *OAuthProxy) ErrorResponse(rw http.ResponseWriter, code int, title string, message string) {
	log.Printf("Response error: %d %s %s", code, title, message)
	rw.WriteHeader(code)

	errorResponse := struct {
		Code    int    `json:"code"`
		Title   string `json:"title"`
		Message string `json:"message"`
	}{
		Code:    code,
		Title:   title,
		Message: message,
	}

	jsonResponse, err := json.Marshal(errorResponse)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}

	rw.Header().Set("Content-Type", "application/json")

	_, err = rw.Write(jsonResponse)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
	}
}

func (p *OAuthProxy) GetRedirect(req *http.Request) (string, error) {
	redirect := path.Join(p.rootPrefix, req.RequestURI)
	return redirect, nil
}

func (p *OAuthProxy) IsWhitelistedRequest(req *http.Request) (ok bool) {
	isPreflightRequestAllowed := p.skipAuthPreflight && req.Method == "OPTIONS"
	return isPreflightRequestAllowed || (!p.IsProtectedPath(req.URL.Path) || p.IsWhitelistedPath(req.URL.Path))
}

// IsProtectedPath returns true if auth-regex matches the path, false otherwise. Defaults to true if no auth-regex is given.
func (p *OAuthProxy) IsProtectedPath(path string) bool {
	match := true
	for _, u := range p.compiledAuthRegex {
		ok := u.MatchString(path)
		if ok {
			return true
		}
		match = false
	}
	return match
}

func (p *OAuthProxy) IsWhitelistedPath(path string) (ok bool) {
	for _, u := range p.compiledSkipRegex {
		ok = u.MatchString(path)
		if ok {
			return
		}
	}
	return
}

func getRemoteAddr(req *http.Request) (s string) {
	s = req.RemoteAddr
	if req.Header.Get("X-Real-IP") != "" {
		s += fmt.Sprintf(" (%q)", req.Header.Get("X-Real-IP"))
	}
	return
}

func (p *OAuthProxy) updateOAuth2URL(req *http.Request) {
	host := req.Host
	scheme := "http"
	if req.TLS != nil {
		scheme = "https"
	}
	if p.redirectURL.Host != host {
		p.redirectURL.Host = host
		p.redirectURL.Scheme = scheme
		p.logoutURL.Host = host
		p.logoutURL.Scheme = scheme
		p.provider.SetLoginURL(req, scheme)
		p.provider.SetRedeemURL(req, scheme)
	}
	fmt.Printf("updated redirect url: %s", p.redirectURL.String())
}

func (p *OAuthProxy) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	// set redirect base url here
	p.updateOAuth2URL(req)

	switch path := req.URL.Path; {
	case path == p.LogOutPath:
		p.LogOut(rw, req)
	case path == p.OAuthStartPath:
		p.OAuthStart(rw, req)
	case path == p.OAuthCallbackPath:
		p.OAuthCallback(rw, req)
	default:
		p.Proxy(rw, req)
	}
}

// LogOut will not call the oauth-server logout interface, only passive calls are received by oauth-proxy.SignOut
func (p *OAuthProxy) LogOut(rw http.ResponseWriter, req *http.Request) {
	// fetch sessionID from req.Form
	encryptedSessionID := req.FormValue(constants.SessionIDParam)
	if encryptedSessionID == "" {
		log.Printf(fuyaoerrors.ErrStrMissingSessionIDParams)
		http.Error(rw, fuyaoerrors.ErrStrMissingSessionIDParams, http.StatusBadRequest)
		return
	}

	// convert to the real sessionID
	session, err := p.provider.SessionFromCookie(encryptedSessionID, p.CookieCipher)
	if err != nil {
		log.Printf("the encrypted cookie does not contain the sessionID")
		http.Error(rw, fuyaoerrors.ErrStrSessionIDDeleteFailed, http.StatusInternalServerError)
		return
	}

	// delete the session<->secret here
	if err := p.provider.DeleteSession(session.SessionID); err != nil {
		log.Printf("cannot delete cookied session <-> sessionid mapper secret")
		http.Error(rw, fuyaoerrors.ErrStrSessionIDDeleteFailed, http.StatusInternalServerError)
		return
	}

	// successfully delete the session
	rw.WriteHeader(http.StatusOK)
	_, err = rw.Write([]byte("Delete session OK"))
	if err != nil {
		log.Printf("write header back to reponse failed")
		return
	}

	log.Printf("successfully log out")
	return
}

func (p *OAuthProxy) SignOut(rw http.ResponseWriter, req *http.Request) {
	// delete the session<->secret here
	session, _, err := p.LoadCookiedSession(req)
	if err != nil && !errors.Is(err, http.ErrNoCookie) {
		log.Printf("loading cookied session gets err: %s", err)
	}
	if err != nil {
		if err = p.provider.DeleteSession(session.SessionID); err != nil {
			log.Printf("cannot delete cookied session <-> sessionid mapper secret")
		}
	}

	// redirect
	p.ClearSessionCookie(rw, req)
	redirectURL := "/"
	if len(p.LogoutRedirectURL) > 0 {
		redirectURL = p.LogoutRedirectURL
	}
	http.Redirect(rw, req, redirectURL, http.StatusFound)
}

func (p *OAuthProxy) OAuthStart(rw http.ResponseWriter, req *http.Request) {
	redirectURI := p.GetRedirectURI(req.Host)

	// reuse the nonce and redirect if set
	nonce, redirect := p.getNonceAndRedirect(req)
	if nonce != "" && redirect != "" {
		http.Redirect(rw, req, p.provider.GetLoginURL(redirectURI,
			fmt.Sprintf("%v:%v", nonce, redirect)), http.StatusFound)
	}

	// generate new nonce and redirect
	nonce, err := cookie.Nonce()
	if err != nil {
		p.ErrorResponse(rw, http.StatusInternalServerError, "Internal Error", err.Error())
		return
	}
	p.SetCSRFCookie(rw, req, nonce)
	redirect, err = p.GetRedirect(req)
	if err != nil {
		p.ErrorResponse(rw, http.StatusInternalServerError, "Internal Error", err.Error())
		return
	}

	http.Redirect(rw, req, p.provider.GetLoginURL(redirectURI,
		fmt.Sprintf("%v:%v", nonce, redirect)), http.StatusFound)
}

func (p *OAuthProxy) getNonceAndRedirect(req *http.Request) (string, string) {
	const parts = 2
	s := strings.SplitN(req.Form.Get("state"), ":", parts)
	if len(s) != parts {
		return "", ""
	}

	nonce := s[0]
	redirect := s[parts-1]
	c, err := req.Cookie(p.CSRFCookieName)
	if err != nil {
		return "", ""
	}
	if c.Value != nonce {
		return "", ""
	}

	return nonce, redirect
}

func (p *OAuthProxy) OAuthCallback(rw http.ResponseWriter, req *http.Request) {
	remoteAddr := getRemoteAddr(req)

	// finish the oauth cycle
	err := req.ParseForm()
	if err != nil {
		p.ErrorResponse(rw, http.StatusInternalServerError, "Internal Error", err.Error())
		return
	}
	errorString := req.Form.Get("error")
	if errorString != "" {
		p.ErrorResponse(rw, http.StatusForbidden, "Permission Denied", errorString)
		return
	}

	session, err := p.redeemCode(req.Host, req.Form.Get("code"))
	if err != nil {
		if errors.Is(err, providers.ErrPermissionDenied) {
			log.Printf("%s Permission Denied: user is unauthorized when redeeming token", remoteAddr)
			p.ErrorResponse(rw, http.StatusForbidden, "Permission Denied", "Invalid Account")
			return
		}
		log.Printf("error redeeming code (client:%s): %s", remoteAddr, err)
		p.ErrorResponse(rw, http.StatusInternalServerError, "Internal Error", "Internal Error")
		return
	}

	s := strings.SplitN(req.Form.Get("state"), ":", 2)
	if len(s) != 2 {
		p.ErrorResponse(rw, http.StatusInternalServerError, "Internal Error", "Invalid State")
		return
	}
	nonce := s[0]
	redirect := s[1]
	c, err := req.Cookie(p.CSRFCookieName)
	if err != nil {
		p.ErrorResponse(rw, http.StatusForbidden, "Permission Denied", err.Error()+", csrf token missing")
		return
	}
	p.ClearCSRFCookie(rw, req)
	if c.Value != nonce {
		log.Printf("%s csrf token mismatch, potential attack", remoteAddr)
		p.ErrorResponse(rw, http.StatusForbidden, "Permission Denied", "csrf failed")
		return
	}

	if (!strings.HasPrefix(redirect, "/") && !strings.HasPrefix(redirect, "http")) ||
		strings.HasPrefix(redirect, "//") {
		redirect = "/"
	}

	// set cookie, or deny
	if p.Validator(session.Email) && p.provider.ValidateGroup(session.Email) {
		log.Printf("%s authentication complete %s", remoteAddr, session)
		err := p.SaveSession(rw, req, session)
		if err != nil {
			log.Printf("%s %s", remoteAddr, err)
			p.ErrorResponse(rw, http.StatusInternalServerError, "Internal Error", "Internal Error")
			return
		}
		http.Redirect(rw, req, redirect, http.StatusFound)
	} else {
		log.Printf("%s Permission Denied: %q is unauthorized", remoteAddr, session.User)
		p.ErrorResponse(rw, http.StatusForbidden, "Permission Denied", "Invalid Account")
	}
}

func (p *OAuthProxy) AuthenticateOnly(rw http.ResponseWriter, req *http.Request) {
	status := p.Authenticate(rw, req)
	if status == http.StatusAccepted {
		rw.WriteHeader(http.StatusAccepted)
	} else {
		http.Error(rw, "unauthorized request", http.StatusUnauthorized)
	}
}

// Proxy first authenticate the request, if failed will redirect to OAuth server then forward to upstream
func (p *OAuthProxy) Proxy(rw http.ResponseWriter, req *http.Request) { // 允许特定源的跨域请求
	status := p.Authenticate(rw, req)
	if status == http.StatusInternalServerError {
		p.ErrorResponse(rw, http.StatusInternalServerError,
			"Internal Error", "Internal Error")
	} else if status == http.StatusForbidden {
		p.ErrorResponse(rw, http.StatusForbidden,
			"Unauthorized", fmt.Sprintf("Current user is unauthorized to request %s", req.URL.Path))
	} else {
		accessLog(rw, req)
		p.serveMux.ServeHTTP(rw, req)
	}
}

func (p *OAuthProxy) Authenticate(rw http.ResponseWriter, req *http.Request) int {
	var saveSession, clearSession, revalidated bool
	remoteAddr := getRemoteAddr(req)
	session, sessionAge, err := p.LoadCookiedSession(req)
	if err != nil && err != http.ErrNoCookie {
		log.Printf("%s %s", remoteAddr, err)
	}

	if session != nil && sessionAge > p.CookieRefresh && p.CookieRefresh != time.Duration(0) {
		log.Printf("%s refreshing %s old session cookie for %s (refresh after %s)", remoteAddr, sessionAge, session, p.CookieRefresh)
		saveSession = true
	}

	if ok, err := p.provider.RefreshSessionIfNeeded(session); err != nil {
		log.Printf("%s removing session. error refreshing access token %s %s", remoteAddr, err, session)
		clearSession = true
		session = nil
	} else if ok {
		saveSession = true
		revalidated = true
	}

	// session Expired, need to request again
	if session != nil && session.IsExpired() {
		log.Printf("%s removing session. token expired %s", remoteAddr, session)
		saveSession = false
		clearSession = true
	}

	if saveSession && !revalidated && session != nil && session.AccessToken != "" {
		if !p.provider.ValidateSessionState(session) {
			log.Printf("%s removing session. error validating %s", remoteAddr, session)
			saveSession = false
			session = nil
			clearSession = true
		}
	}

	if session != nil && session.Email != "" && !p.Validator(session.Email) {
		log.Printf("%s Permission Denied: removing session %s", remoteAddr, session)
		session = nil
		saveSession = false
		clearSession = true
	}

	if saveSession && session != nil {
		err := p.SaveSession(rw, req, session)
		if err != nil {
			log.Printf("%s %s", remoteAddr, err)
			return http.StatusInternalServerError
		}
	}

	if clearSession {
		if err = p.provider.DeleteSession(session.SessionID); err != nil {
			return http.StatusInternalServerError
		}
		p.ClearSessionCookie(rw, req)
	}

	// auth by host cluster's access token
	if session == nil {
		session, err = p.CheckMultiClusterAuth(req)
		if err != nil {
			log.Printf("multiclusterauth: %s %s", remoteAddr, err)
		}
	}

	// auth by access token input from another component
	tokenProvidedByClient := false
	if session == nil {
		session, err = p.CheckRequestAuth(req)
		if err != nil {
			log.Printf("requestauth internal error: %s %s", remoteAddr, err)
		}
		tokenProvidedByClient = true
	}

	if session == nil {
		return http.StatusForbidden
	}

	// At this point, the user is authenticated. proxy normally
	if p.PassBasicAuth {
		req.SetBasicAuth(session.User, p.BasicAuthPassword)
		req.Header["X-Forwarded-User"] = []string{session.User}
		if session.Email != "" {
			req.Header["X-Forwarded-Email"] = []string{session.Email}
		}
	}
	if p.PassUserHeaders {
		req.Header["X-Forwarded-User"] = []string{session.User}
		if session.Email != "" {
			req.Header["X-Forwarded-Email"] = []string{session.Email}
		}
	}
	if p.SetXAuthRequest {
		rw.Header().Set("X-Auth-Request-User", session.User)
		if session.Email != "" {
			rw.Header().Set("X-Auth-Request-Email", session.Email)
		}
	}
	if (!tokenProvidedByClient && p.PassAccessToken) && session.AccessToken != "" {
		req.Header["X-Forwarded-Access-Token"] = []string{session.AccessToken}
	}
	if session.Email == "" {
		rw.Header().Set("GAP-Auth", session.User)
	} else {
		rw.Header().Set("GAP-Auth", session.Email)
	}
	return http.StatusAccepted
}

// ValidateSessionAuth validate the session through auth interfaces
func (p *OAuthProxy) ValidateSessionAuth(session *providers.SessionState, req *http.Request) (bool, error) {
	return p.provider.ValidateSessionAuth(session, req)
}

func (p *OAuthProxy) CheckRequestAuth(req *http.Request) (*providers.SessionState, error) {
	// handle advanced validation
	return p.provider.ValidateRequest(req)
}

// CheckMultiClusterAuth validate user from token produced by host cluster
func (p *OAuthProxy) CheckMultiClusterAuth(req *http.Request) (*providers.SessionState, error) {
	return p.provider.ValidateMultiClusterRequest(req)
}

// Parse a valid http.SameSite value from a user supplied string for use of making cookies.
func parseSameSite(v string) http.SameSite {
	switch v {
	case "lax":
		return http.SameSiteLaxMode
	case "strict":
		return http.SameSiteStrictMode
	case "none":
		return http.SameSiteNoneMode
	case "":
		return http.SameSiteDefaultMode
	default:
		panic(fmt.Sprintf("Invalid value for SameSite: %s", v))
	}
}

func accessLog(w http.ResponseWriter, r *http.Request) {
	log.Printf(
		`%s %s successfully passed oauth proxy`,
		r.Method,
		r.RequestURI,
	)
}
