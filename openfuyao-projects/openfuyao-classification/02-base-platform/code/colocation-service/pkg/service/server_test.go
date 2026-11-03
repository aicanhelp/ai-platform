/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 */

package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes/fake"

	colocationclientset "openfuyao.com/colocation-service/pkg/client/clientset/versioned"
)

func TestGetPrometheusEndpoint(t *testing.T) {
	// Test unset environment variable
	cleanupPrometheusEnv(t)
	assert.Equal(t, "", getPrometheusEndpoint())

	// Test set environment variable
	setupPrometheusEnv(t)
	defer cleanupPrometheusEnv(t)
	assert.Equal(t, "http://prometheus:9090/api/v1/query_range", getPrometheusEndpoint())
}

func TestServerRun(t *testing.T) {
	ctx, cancel := createTestContext(t)
	defer cancel()

	server := createTestServerInServerTest(t, ctx)

	// Run server in goroutine and verify it can be stopped
	go server.Run(ctx)
	cancel() // Stop the server
}

func TestServerHTTPHandling(t *testing.T) {
	setupPrometheusEnv(t)
	defer cleanupPrometheusEnv(t)

	ctx, cancel := createTestContext(t)
	defer cancel()

	server := createTestServerInServerTest(t, ctx)
	verifyHTTPHandler(t, server)
}

// Helper functions for test setup
func setupPrometheusEnv(t *testing.T) {
	t.Helper()
	err := os.Setenv("PROMETHEUS_SERVICE", "http://prometheus:9090")
	assert.NoError(t, err)
}

func cleanupPrometheusEnv(t *testing.T) {
	t.Helper()
	err := os.Unsetenv("PROMETHEUS_SERVICE")
	assert.NoError(t, err)
}

func createTestContext(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithCancel(context.Background())
}

func createTestServerInServerTest(t *testing.T, ctx context.Context) *Server {
	t.Helper()
	fakeClient := fake.NewSimpleClientset()
	fakeColocationClient := &fakeColocationClient{}

	server, err := New(ctx, fakeClient, fakeColocationClient)
	assert.NoError(t, err)
	assert.NotNil(t, server)
	return server
}

// Helper functions for verification
func verifyServerInitialization(t *testing.T, server *Server) {
	t.Helper()
	assert.NotNil(t, server.Handle)
	assert.NotNil(t, server.container)
	assert.NotNil(t, server.ws)
	assert.Equal(t, "http://prometheus:9090/api/v1/query_range", server.prometheusEndpoint)
	assert.Equal(t, fake.NewSimpleClientset(), server.client)
	assert.IsType(t, &fakeColocationClient{}, server.colocationclient)
}

func verifyHTTPHandler(t *testing.T, server *Server) {
	t.Helper()
	req := httptest.NewRequest("GET", "/rest/colocation/v1/colocation-status/", nil)
	w := httptest.NewRecorder()

	server.Handle.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

// fakeColocationClient for testing
type fakeColocationClient struct {
	colocationclientset.Interface
}

func TestQueryNodeResource_EmptyEndpoint(t *testing.T) {
	cleanupPrometheusEnv(t)
	defer cleanupPrometheusEnv(t)

	ctx, cancel := createTestContext(t)
	defer cancel()

	server := createTestServerInServerTest(t, ctx)
	_, err := server.queryNodeResource([]string{"node1"})
	assert.Error(t, err)
}

func TestQueryClusterResource_EmptyEndpoint(t *testing.T) {
	cleanupPrometheusEnv(t)
	defer cleanupPrometheusEnv(t)

	ctx, cancel := createTestContext(t)
	defer cancel()

	server := createTestServerInServerTest(t, ctx)
	_, err := server.queryClusterResource()
	assert.Error(t, err)
}
