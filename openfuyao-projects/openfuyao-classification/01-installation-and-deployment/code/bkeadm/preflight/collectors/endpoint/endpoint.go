/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package endpoint collects read-only facts about installation sources:
// DNS resolvability, TCP reachability, TLS validity and authentication.
//
// It probes from the real consuming subject (the host or a node), never from
// an arbitrary source. All probes are short-lived and closed before return.
package endpoint

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/normalize"
)

// Collector is the endpoint fact collector.
type Collector struct {
	timeout     time.Duration
	credentials map[string]Credential
}

// Credential is held only by the collector instance in memory. It is never
// copied into a plan, fact, evidence, report or log.
type Credential struct{ Username, Password string }

const (
	defaultProbeTimeout = 5 * time.Second
	maxRedirects        = 5
	maxTokenPayload     = 64 << 10
	authParameterParts  = 2
)

// New returns an endpoint Collector with the given per-probe timeout.
func New(timeout time.Duration) *Collector {
	if timeout <= 0 {
		timeout = defaultProbeTimeout
	}
	return &Collector{timeout: timeout, credentials: map[string]Credential{}}
}

// NewWithCredentials returns a collector with credentials scoped by
// endpoint (host:port). The map is copied so the caller can clear its input.
func NewWithCredentials(timeout time.Duration, credentials map[string]Credential) *Collector {
	c := New(timeout)
	for endpoint, credential := range credentials {
		c.credentials[endpoint] = credential
	}
	return c
}

// ID returns the collector identifier.
func (c *Collector) ID() string { return "endpoint" }

// Collect returns facts for the requested keys. Keys encode the endpoint and
// the probe kind:
//
//	endpoint.dns.<host>
//	endpoint.tcp.<host>:<port>
//	endpoint.tls.<host>:<port>
//	endpoint.auth.<host>:<port>   (auth is reported by the host via evidence)
func (c *Collector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	out := make([]api.Fact, 0, len(req.Keys))
	for _, key := range req.Keys {
		fact := c.requestedFact(ctx, key)
		if len(req.Subjects) == 1 {
			fact.Subject = req.Subjects[0].ID
		}
		out = append(out, fact)
	}
	return out, nil
}

func (c *Collector) requestedFact(ctx context.Context, key string) api.Fact {
	select {
	case <-ctx.Done():
		ctxErr := ctx.Err()
		if ctxErr == nil {
			ctxErr = context.Canceled
		}
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrTimeout, Message: ctxErr.Error()}}
	default:
		return c.collectKey(ctx, key)
	}
}

func (c *Collector) collectKey(ctx context.Context, key string) api.Fact {
	switch {
	case strings.HasPrefix(key, "endpoint.dns."):
		host := strings.TrimPrefix(key, "endpoint.dns.")
		return c.dnsFact(ctx, key, host)
	case strings.HasPrefix(key, "endpoint.tcp."):
		addr := strings.TrimPrefix(key, "endpoint.tcp.")
		return c.tcpFact(ctx, key, addr)
	case strings.HasPrefix(key, "endpoint.tls."):
		target := strings.TrimPrefix(key, "endpoint.tls.")
		return c.tlsFact(ctx, key, target)
	case strings.HasPrefix(key, "endpoint.auth."):
		target := strings.TrimPrefix(key, "endpoint.auth.")
		return c.httpFact(ctx, key, target, "auth")
	case strings.HasPrefix(key, "endpoint.object."):
		target := strings.TrimPrefix(key, "endpoint.object.")
		return c.httpFact(ctx, key, target, "object")
	default:
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "unknown endpoint key: " + key}}
	}
}

func (c *Collector) dnsFact(ctx context.Context, key, host string) api.Fact {
	host = normalize.Address(host)
	if parsedHost, _, err := net.SplitHostPort(host); err == nil {
		host = parsedHost
	}
	resolver := net.DefaultResolver
	ips, err := resolver.LookupHost(ctx, host)
	if err != nil || len(ips) == 0 {
		message := "dns lookup returned no addresses"
		if err != nil {
			message = "dns lookup failed: " + err.Error()
		}
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: errorCode(ctx, err), Message: message}}
	}
	return api.Fact{Key: key, Available: true, Value: true,
		Evidence: []api.Evidence{{Kind: "dns", Source: "endpoint", Locator: host}}}
}

func (c *Collector) tcpFact(ctx context.Context, key, addr string) api.Fact {
	if strings.Contains(addr, "://") {
		parsed, err := parseTarget(addr)
		if err != nil {
			return api.Fact{
				Key:       key,
				Available: false,
				Error:     &api.FactError{Code: api.FactErrParseError, Message: "endpoint TCP connectivity failed"},
			}
		}
		addr = parsed.Host
	}
	d := net.Dialer{Timeout: c.timeout}
	conn, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: errorCode(ctx, err), Message: "endpoint TCP connectivity failed"}}
	}
	if err := conn.Close(); err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrUnreachable, Message: "endpoint TCP connectivity failed"}}
	}
	return api.Fact{Key: key, Available: true, Value: true,
		Evidence: []api.Evidence{{Kind: "tcp", Source: "endpoint", Locator: normalize.RedactEndpoint(addr)}}}
}

func (c *Collector) tlsFact(ctx context.Context, key, target string) api.Fact {
	u, err := parseTarget(target)
	if err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: err.Error()}}
	}
	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{
				Code:    api.FactErrParseError,
				Message: "bad addr: " + normalize.RedactEndpoint(u.Host),
			}}
	}
	dialer := &tls.Dialer{Config: &tls.Config{ServerName: host, MinVersion: tls.VersionTLS12}}
	conn, err := dialer.DialContext(ctx, "tcp", u.Host)
	if err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: errorCode(ctx, err), Message: err.Error()}}
	}
	defer conn.Close()
	return api.Fact{Key: key, Available: true, Value: true,
		Evidence: []api.Evidence{{Kind: "tls", Source: "endpoint", Locator: normalize.RedactEndpoint(u.Host)}}}
}

func errorCode(ctx context.Context, err error) string {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
		return api.FactErrTimeout
	}
	return api.FactErrUnreachable
}

func (c *Collector) httpFact(ctx context.Context, key, target, kind string) api.Fact {
	u, err := parseTarget(target)
	if err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: err.Error()}}
	}
	probeCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	resp, factErr := c.httpResponse(probeCtx, u)
	if factErr != nil {
		return api.Fact{Key: key, Available: false, Error: factErr}
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// The response has already been evaluated; a close error cannot
			// change the read-only probe result.
			_ = closeErr
		}
	}()
	accepted := resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusBadRequest
	if kind == "auth" {
		// 404 proves the server accepted the request even though this generic
		// root probe is not a release-object lookup.
		accepted = accepted || resp.StatusCode == http.StatusNotFound
	}
	return api.Fact{Key: key, Available: true, Value: accepted,
		Evidence: []api.Evidence{{
			Kind: "http", Source: "endpoint", Locator: normalize.RedactEndpoint(u.Host),
			Observed: fmt.Sprintf("status=%d", resp.StatusCode),
		}}}
}

func (c *Collector) httpResponse(ctx context.Context, u *url.URL) (*http.Response, *api.FactError) {
	client := &http.Client{Timeout: c.timeout, CheckRedirect: func(_ *http.Request, via []*http.Request) error {
		if len(via) >= maxRedirects {
			return fmt.Errorf("too many redirects")
		}
		return nil
	}}
	credential, hasCredential := c.credentials[u.Host]
	request := httpProbeRequest{target: u.String(), credential: credential, hasCredential: hasCredential}
	resp, err := doHTTPProbe(ctx, client, request)
	if err != nil {
		return nil, &api.FactError{Code: errorCode(ctx, err), Message: "endpoint HTTP probe failed"}
	}
	if resp.StatusCode != http.StatusUnauthorized {
		return resp, nil
	}
	tokenRequest := bearerTokenRequest{
		challenge: resp.Header.Get("WWW-Authenticate"), sourceHost: u.Hostname(),
		credential: credential, hasCredential: hasCredential,
	}
	token, tokenErr := acquireBearerToken(ctx, client, tokenRequest)
	if tokenErr != nil || token == "" {
		return resp, nil
	}
	if closeErr := resp.Body.Close(); closeErr != nil {
		return nil, &api.FactError{Code: api.FactErrUnreachable, Message: closeErr.Error()}
	}
	request.authorization = "Bearer " + token
	resp, err = doHTTPProbe(ctx, client, request)
	if err != nil {
		return nil, &api.FactError{
			Code: errorCode(ctx, err), Message: "authenticated endpoint HTTP probe failed",
		}
	}
	return resp, nil
}

type httpProbeRequest struct {
	target        string
	credential    Credential
	hasCredential bool
	authorization string
}

func doHTTPProbe(ctx context.Context, client *http.Client, probe httpProbeRequest) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, probe.target, nil)
	if err != nil {
		return nil, err
	}
	// Registry manifest endpoints commonly return 404 for an unqualified HEAD
	// request even when the tag exists. Advertise the OCI/Docker manifest
	// media types so the same read-only probe works for multi-arch images.
	req.Header.Set("Accept", strings.Join([]string{
		"application/vnd.oci.image.index.v1+json",
		"application/vnd.docker.distribution.manifest.list.v2+json",
		"application/vnd.oci.image.manifest.v1+json",
		"application/vnd.docker.distribution.manifest.v2+json",
	}, ", "))
	switch {
	case probe.authorization != "":
		req.Header.Set("Authorization", probe.authorization)
	case probe.hasCredential:
		req.SetBasicAuth(probe.credential.Username, probe.credential.Password)
	default:
		req.Header.Del("Authorization")
	}
	return client.Do(req)
}

type bearerTokenRequest struct {
	challenge     string
	sourceHost    string
	credential    Credential
	hasCredential bool
}

func acquireBearerToken(ctx context.Context, client *http.Client, tokenRequest bearerTokenRequest) (string, error) {
	challenge := tokenRequest.challenge
	if !strings.HasPrefix(strings.ToLower(strings.TrimSpace(challenge)), "bearer ") {
		return "", fmt.Errorf("unsupported authentication challenge")
	}
	params := map[string]string{}
	for _, item := range strings.Split(strings.TrimSpace(challenge[len("Bearer "):]), ",") {
		parts := strings.SplitN(strings.TrimSpace(item), "=", authParameterParts)
		if len(parts) == authParameterParts {
			params[strings.ToLower(parts[0])] = strings.Trim(parts[1], `"`)
		}
	}
	realm, err := url.Parse(params["realm"])
	if err != nil || realm.Host == "" || realm.Scheme != "https" {
		return "", fmt.Errorf("invalid bearer token realm")
	}
	query := realm.Query()
	for _, name := range []string{"service", "scope"} {
		if params[name] != "" {
			query.Set(name, params[name])
		}
	}
	realm.RawQuery = query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, realm.String(), nil)
	if err != nil {
		return "", err
	}
	// Never forward user credentials to a challenge-controlled cross-host
	// realm. Anonymous cross-host token flows remain supported.
	if tokenRequest.hasCredential && strings.EqualFold(realm.Hostname(), tokenRequest.sourceHost) {
		req.SetBasicAuth(tokenRequest.credential.Username, tokenRequest.credential.Password)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("token endpoint rejected credentials")
	}
	var payload struct {
		Token       string `json:"token"`
		AccessToken string `json:"access_token"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxTokenPayload)).Decode(&payload); err != nil {
		return "", err
	}
	if payload.Token != "" {
		return payload.Token, nil
	}
	return payload.AccessToken, nil
}

func parseTarget(target string) (*url.URL, error) {
	if !strings.Contains(target, "://") {
		target = "https://" + target
	}
	u, err := url.Parse(target)
	if err != nil || u.Host == "" {
		return nil, fmt.Errorf("invalid endpoint")
	}
	if _, _, err := net.SplitHostPort(u.Host); err != nil {
		defaultPort := "443"
		if u.Scheme == "http" {
			defaultPort = "80"
		}
		u.Host = net.JoinHostPort(strings.Trim(u.Host, "[]"), defaultPort)
	}
	if u.Path == "" {
		u.Path = "/"
	}
	return u, nil
}

// AuthFact builds a fact for an authentication result. The host performs the
// auth exchange (it holds the credential) and reports the boolean outcome;
// this helper keeps the evidence desensitized.
func AuthFact(key string, accepted bool, endpoint string) api.Fact {
	return api.Fact{Key: key, Available: true, Value: accepted,
		Evidence: []api.Evidence{{Kind: "tls", Source: "endpoint",
			Locator: normalize.RedactEndpoint(endpoint), Observed: fmt.Sprintf("%v", accepted)}}}
}
