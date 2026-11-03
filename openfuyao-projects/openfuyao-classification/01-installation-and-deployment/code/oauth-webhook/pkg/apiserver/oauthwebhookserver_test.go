/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package apiserver

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/api/authentication/v1"

	"openfuyao/oauth-webhook/cmd/oauth-webhook/app/config"
	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/httpserver"
)

const (
	testSigningKey = "a-valid-string-secret-that-is-at-least-512-bits-long-which-is-very-long"
	testValidToken = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJvYXV0aC1wcm94eSIsImV4cCI6MTkyNDkyMDAwMCwic3ViIjoiYWRtaW4ifQ.49rrmIRdFXfhigt70993BThscDv6fZECMQYb6g1c9JPVpXYL7ghKJ_OCOp8eKsisZBlgsm0jAXpo0ffJzfuaYQ"
)

func TestNewOAuthWebhookAPIServer(t *testing.T) {
	server, err := NewOAuthWebhookAPIServer(&config.OAuthWebhookConfig{
		HttpServerConfig: httpserver.NewDefaultHttpServerOptions(),
		JWTPrivateKey:    []byte(testSigningKey),
	}, make(<-chan struct{}))
	require.NoError(t, err)
	require.Equal(t, ":9095", server.Server.Addr)
	require.NotNil(t, server.Router)
	require.NotNil(t, server.Webhook)
}

func TestNewOAuthWebhookAPIServerHTTPSConfigError(t *testing.T) {
	_, err := NewOAuthWebhookAPIServer(&config.OAuthWebhookConfig{
		HttpServerConfig: &httpserver.ServerOptions{
			HttpsPort:         9096,
			TlsCertFile:       "missing.crt",
			TlsPrivateKeyFile: "missing.key",
			MasterCAFile:      "missing-ca.crt",
		},
		JWTPrivateKey: []byte(testSigningKey),
	}, make(<-chan struct{}))
	require.Error(t, err)
}

func TestPrepareRunAndServeWebhookRoute(t *testing.T) {
	server, err := NewOAuthWebhookAPIServer(&config.OAuthWebhookConfig{
		HttpServerConfig: httpserver.NewDefaultHttpServerOptions(),
		JWTPrivateKey:    []byte(testSigningKey),
	}, make(<-chan struct{}))
	require.NoError(t, err)
	require.NoError(t, server.PrepareRun(make(<-chan struct{})))

	t.Run("invalid method", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/oauth/tokenauth/fuyao", nil)
		rr := httptest.NewRecorder()

		server.Router.ServeHTTP(rr, req)

		require.Equal(t, http.StatusBadRequest, rr.Code)

		var review v1.TokenReview
		require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &review))
		require.False(t, review.Status.Authenticated)
		require.Equal(t, fuyaoerrors.ErrStrRequestMethodNotAllowed, review.Status.Error)
	})

	t.Run("invalid token", func(t *testing.T) {
		body, err := json.Marshal(v1.TokenReview{Spec: v1.TokenReviewSpec{Token: "invalid-token"}})
		require.NoError(t, err)

		req := httptest.NewRequest(http.MethodPost, "/oauth/tokenauth/fuyao", bytes.NewReader(body))
		rr := httptest.NewRecorder()

		server.Router.ServeHTTP(rr, req)

		require.Equal(t, http.StatusUnauthorized, rr.Code)

		var review v1.TokenReview
		require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &review))
		require.False(t, review.Status.Authenticated)
		require.Equal(t, "token contains an invalid number of segments", review.Status.Error)
	})

	t.Run("valid token", func(t *testing.T) {
		body, err := json.Marshal(v1.TokenReview{Spec: v1.TokenReviewSpec{Token: testValidToken}})
		require.NoError(t, err)

		req := httptest.NewRequest(http.MethodPost, "/oauth/tokenauth/fuyao", bytes.NewReader(body))
		rr := httptest.NewRecorder()

		server.Router.ServeHTTP(rr, req)

		require.Equal(t, http.StatusOK, rr.Code)

		var review v1.TokenReview
		require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &review))
		require.True(t, review.Status.Authenticated)
		require.Equal(t, "admin", review.Status.User.Username)
		require.Empty(t, review.Status.Error)
	})
}

func TestRunReturnsServerClosedOnShutdown(t *testing.T) {
	server, err := NewOAuthWebhookAPIServer(&config.OAuthWebhookConfig{
		HttpServerConfig: httpserver.NewDefaultHttpServerOptions(),
		JWTPrivateKey:    []byte(testSigningKey),
	}, make(<-chan struct{}))
	require.NoError(t, err)
	require.NoError(t, server.PrepareRun(make(<-chan struct{})))
	server.Server.Addr = "127.0.0.1:0"

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.Run(ctx)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case runErr := <-errCh:
		require.ErrorIs(t, runErr, http.ErrServerClosed)
	case <-time.After(3 * time.Second):
		t.Fatal("server run did not exit after shutdown")
	}
}

func TestNewOAuthWebhookAPIServerLoadsTLSFiles(t *testing.T) {
	testDir := newTestDir(t, "tls-config-")
	certPEM, keyPEM := newTestCertificate(t)
	certFile := writeFile(t, testDir, "server.crt", certPEM)
	keyFile := writeFile(t, testDir, "server.key", keyPEM)
	caFile := writeFile(t, testDir, "ca.crt", certPEM)

	server, err := NewOAuthWebhookAPIServer(&config.OAuthWebhookConfig{
		HttpServerConfig: &httpserver.ServerOptions{
			HttpsPort:         9443,
			TlsCertFile:       certFile,
			TlsPrivateKeyFile: keyFile,
			MasterCAFile:      caFile,
		},
		JWTPrivateKey: []byte(testSigningKey),
	}, make(<-chan struct{}))
	require.NoError(t, err)
	require.Equal(t, ":9443", server.Server.Addr)
	require.NotNil(t, server.Server.TLSConfig)
	require.Len(t, server.Server.TLSConfig.Certificates, 1)
	require.NotNil(t, server.Server.TLSConfig.ClientCAs)
}

func newTestDir(t *testing.T, prefix string) string {
	t.Helper()

	dir, err := os.MkdirTemp(".", prefix)
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = os.RemoveAll(dir)
	})

	return dir
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()

	filePath := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(filePath, []byte(content), 0o600))
	return filePath
}

func newTestCertificate(t *testing.T) (string, string) {
	t.Helper()

	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "oauth-webhook-test",
		},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, template, template, &privateKey.PublicKey, privateKey)
	require.NoError(t, err)

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})

	return string(certPEM), string(keyPEM)
}
