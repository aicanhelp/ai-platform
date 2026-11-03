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

package endpoint

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/preflight/api"
)

func TestCollectTCPReachable(t *testing.T) {
	// Start a local TCP listener.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skip("cannot listen:", err)
	}
	defer ln.Close()
	addr := ln.Addr().String()

	c := New(time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"endpoint.tcp." + addr}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectTCPUnreachable(t *testing.T) {
	// Use a port that is very likely closed.
	c := New(200 * time.Millisecond)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"endpoint.tcp.127.0.0.1:1"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

func TestCollectDNSLocalhost(t *testing.T) {
	c := New(time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"endpoint.dns.localhost"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectDNSHostPort(t *testing.T) {
	c := New(time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"endpoint.dns.localhost:443"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectHTTPAuthAndObject(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		username, password, ok := r.BasicAuth()
		if !ok || username != "user" || password != "secret" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	parsed, _ := url.Parse(server.URL)
	c := NewWithCredentials(time.Second, map[string]Credential{parsed.Host: {Username: "user", Password: "secret"}})
	for _, kind := range []string{"auth", "object"} {
		facts, err := c.Collect(context.Background(), api.FactRequest{
			Keys: []string{"endpoint." + kind + "." + server.URL}, Subjects: []api.Subject{{ID: "bootstrap"}},
		})
		assert.NoError(t, err)
		assert.True(t, facts[0].Available)
		assert.Equal(t, true, facts[0].Value)
		assert.Equal(t, "bootstrap", facts[0].Subject)
		for _, evidence := range facts[0].Evidence {
			assert.NotContains(t, evidence.Locator, "secret")
		}
	}
}

func TestCollectHTTPTimeoutUsesTimeoutCause(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()
	c := New(20 * time.Millisecond)
	facts, err := c.Collect(context.Background(), api.FactRequest{Keys: []string{"endpoint.auth." + server.URL}})
	assert.NoError(t, err)
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}

func TestCollectDNSUnresolvable(t *testing.T) {
	c := New(time.Second)
	// A definitely-unresolvable name.
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"endpoint.dns.invalid.invalid.invalid"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnreachable, facts[0].Error.Code)
}

func TestCollectUnknownKey(t *testing.T) {
	c := New(time.Second)
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"bogus.key"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}

func TestCollectContextCanceled(t *testing.T) {
	c := New(time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	facts, _ := c.Collect(ctx, api.FactRequest{Keys: []string{"endpoint.tcp.127.0.0.1:80"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}

func TestAuthFactDesensitized(t *testing.T) {
	f := AuthFact("endpoint.auth.user:pass@host:443", true, "https://user:pass@host:443")
	assert.True(t, f.Available)
	assert.Equal(t, true, f.Value)
	// Evidence locator must not contain the credential.
	for _, e := range f.Evidence {
		assert.NotContains(t, e.Locator, "pass")
	}
}

func TestParseTargetDefaultsAndIPv6(t *testing.T) {
	tests := []struct {
		target, host, path string
	}{
		{target: "registry.example.com", host: "registry.example.com:443", path: "/"},
		{target: "http://repo.example.com/content", host: "repo.example.com:80", path: "/content"},
		{target: "https://[2001:db8::1]:5443/v2", host: "[2001:db8::1]:5443", path: "/v2"},
	}
	for _, tt := range tests {
		parsed, err := parseTarget(tt.target)
		assert.NoError(t, err)
		assert.Equal(t, tt.host, parsed.Host)
		assert.Equal(t, tt.path, parsed.Path)
	}
	_, err := parseTarget("://bad")
	assert.ErrorContains(t, err, "invalid endpoint")
}

func TestTLSFactRejectsInvalidAndUntrustedEndpoints(t *testing.T) {
	collector := New(100 * time.Millisecond)
	invalid := collector.tlsFact(context.Background(), "endpoint.tls.bad", "://bad")
	assert.Equal(t, api.FactErrParseError, invalid.Error.Code)

	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	untrusted := collector.tlsFact(context.Background(), "endpoint.tls.local", server.URL)
	assert.False(t, untrusted.Available)
	assert.Equal(t, api.FactErrUnreachable, untrusted.Error.Code)
}

func TestHTTPFactStatusSemantics(t *testing.T) {
	tests := []struct {
		kind   string
		status int
		want   bool
	}{
		{kind: "auth", status: http.StatusNotFound, want: true},
		{kind: "auth", status: http.StatusUnauthorized, want: false},
		{kind: "object", status: http.StatusOK, want: true},
		{kind: "object", status: http.StatusNotFound, want: false},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%s-%d", tt.kind, tt.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
			}))
			defer server.Close()
			fact := New(time.Second).httpFact(context.Background(), "endpoint."+tt.kind+"."+server.URL, server.URL, tt.kind)
			assert.True(t, fact.Available)
			assert.Equal(t, tt.want, fact.Value)
			assert.Contains(t, fact.Evidence[0].Observed, fmt.Sprint(tt.status))
		})
	}
}

func TestBearerTokenAcquisition(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		username, password, ok := r.BasicAuth()
		assert.True(t, ok)
		assert.Equal(t, "user", username)
		assert.Equal(t, "secret", password)
		assert.Equal(t, "registry", r.URL.Query().Get("service"))
		assert.Equal(t, "repository:openfuyao/bke:pull", r.URL.Query().Get("scope"))
		_, _ = w.Write([]byte(`{"access_token":"token-value"}`))
	}))
	defer server.Close()
	parsed, err := url.Parse(server.URL)
	assert.NoError(t, err)
	request := bearerTokenRequest{
		challenge:  `Bearer realm="` + server.URL + `",service="registry",scope="repository:openfuyao/bke:pull"`,
		sourceHost: parsed.Hostname(), credential: Credential{Username: "user", Password: "secret"}, hasCredential: true,
	}
	token, err := acquireBearerToken(context.Background(), server.Client(), request)
	assert.NoError(t, err)
	assert.Equal(t, "token-value", token)
}

func TestBearerTokenRejectsUnsafeChallenge(t *testing.T) {
	client := &http.Client{Timeout: time.Second}
	for _, challenge := range []string{"Basic realm=registry", `Bearer realm="http://token.example.com"`} {
		token, err := acquireBearerToken(context.Background(), client, bearerTokenRequest{challenge: challenge})
		assert.Error(t, err)
		assert.Empty(t, token)
	}
}

func TestCredentialsAreCopied(t *testing.T) {
	credentials := map[string]Credential{"registry.example.com:443": {Username: "user", Password: "secret"}}
	collector := NewWithCredentials(0, credentials)
	credentials["registry.example.com:443"] = Credential{}
	assert.Equal(t, "secret", collector.credentials["registry.example.com:443"].Password)
	assert.Equal(t, defaultProbeTimeout, collector.timeout)
}

func TestCollectorID(t *testing.T) {
c := New(time.Second)
assert.NotEmpty(t, c.ID())
}
