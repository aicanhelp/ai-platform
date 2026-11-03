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

package app

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/require"

	"openfuyao/oauth-webhook/cmd/oauth-webhook/app/config"
	"openfuyao/oauth-webhook/pkg/apiserver"
	"openfuyao/oauth-webhook/pkg/httpserver"
)

func TestNewOAuthWebhookCommand(t *testing.T) {
	cmd := NewOAuthWebhookCommand()
	require.Equal(t, "openfuyao1-oauth-webhook", cmd.Use)
	require.Equal(t, "The authentication webhook to validate the access token.", cmd.Short)

	flag := cmd.Flags().Lookup("configFile")
	require.NotNil(t, flag)
	require.Equal(t, "", flag.DefValue)
	require.NoError(t, cmd.Flags().Set("configFile", "test-config.yaml"))
	require.Equal(t, "test-config.yaml", flag.Value.String())
}

func TestWrapRunOAuthWebhookServerShutdownPath(t *testing.T) {
	innerCtxDone := make(chan struct{}, 1)
	patches := gomonkey.ApplyFunc(runOAuthWebhookServer, func(_ *config.OAuthWebhookConfig, ctx context.Context) error {
		<-ctx.Done()
		innerCtxDone <- struct{}{}
		return nil
	})
	defer patches.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	err := wrapRunOAuthWebhookServer(newTestConfig(), ctx)
	require.NoError(t, err)

	select {
	case <-innerCtxDone:
	case <-time.After(time.Second):
		t.Fatal("inner context was not cancelled")
	}
}

func TestWrapRunOAuthWebhookServerErrorPath(t *testing.T) {
	expectedErr := errors.New("runner failed")
	patches := gomonkey.ApplyFunc(runOAuthWebhookServer, func(*config.OAuthWebhookConfig, context.Context) error {
		return expectedErr
	})
	defer patches.Reset()

	err := wrapRunOAuthWebhookServer(newTestConfig(), context.Background())
	require.ErrorIs(t, err, expectedErr)
}

func TestRunOAuthWebhookServer(t *testing.T) {
	testConfig := newTestConfig()
	apiServer := &apiserver.OAuthWebhookAPIServer{}
	serverType := reflect.TypeOf(apiServer)

	t.Run("new server error", func(t *testing.T) {
		expectedErr := errors.New("new server failed")
		patches := gomonkey.ApplyFunc(apiserver.NewOAuthWebhookAPIServer, func(*config.OAuthWebhookConfig, <-chan struct{}) (*apiserver.OAuthWebhookAPIServer, error) {
			return nil, expectedErr
		})
		defer patches.Reset()

		err := runOAuthWebhookServer(testConfig, context.Background())
		require.ErrorIs(t, err, expectedErr)
	})

	t.Run("prepare error", func(t *testing.T) {
		expectedErr := errors.New("prepare failed")
		patches := gomonkey.ApplyFunc(apiserver.NewOAuthWebhookAPIServer, func(*config.OAuthWebhookConfig, <-chan struct{}) (*apiserver.OAuthWebhookAPIServer, error) {
			return apiServer, nil
		})
		patches.ApplyMethod(serverType, "PrepareRun", func(*apiserver.OAuthWebhookAPIServer, <-chan struct{}) error {
			return expectedErr
		})
		defer patches.Reset()

		err := runOAuthWebhookServer(testConfig, context.Background())
		require.ErrorIs(t, err, expectedErr)
	})

	t.Run("server closed treated as success", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(apiserver.NewOAuthWebhookAPIServer, func(*config.OAuthWebhookConfig, <-chan struct{}) (*apiserver.OAuthWebhookAPIServer, error) {
			return apiServer, nil
		})
		patches.ApplyMethod(serverType, "PrepareRun", func(*apiserver.OAuthWebhookAPIServer, <-chan struct{}) error {
			return nil
		})
		patches.ApplyMethod(serverType, "Run", func(*apiserver.OAuthWebhookAPIServer, context.Context) error {
			return http.ErrServerClosed
		})
		defer patches.Reset()

		err := runOAuthWebhookServer(testConfig, context.Background())
		require.NoError(t, err)
	})

	t.Run("server error propagated", func(t *testing.T) {
		expectedErr := errors.New("run failed")
		patches := gomonkey.ApplyFunc(apiserver.NewOAuthWebhookAPIServer, func(*config.OAuthWebhookConfig, <-chan struct{}) (*apiserver.OAuthWebhookAPIServer, error) {
			return apiServer, nil
		})
		patches.ApplyMethod(serverType, "PrepareRun", func(*apiserver.OAuthWebhookAPIServer, <-chan struct{}) error {
			return nil
		})
		patches.ApplyMethod(serverType, "Run", func(*apiserver.OAuthWebhookAPIServer, context.Context) error {
			return expectedErr
		})
		defer patches.Reset()

		err := runOAuthWebhookServer(testConfig, context.Background())
		require.ErrorIs(t, err, expectedErr)
	})
}

func newTestConfig() *config.OAuthWebhookConfig {
	return &config.OAuthWebhookConfig{
		HttpServerConfig: &httpserver.ServerOptions{
			HttpPort: 9095,
		},
		JWTPrivateKey: []byte("signing-key"),
	}
}
