/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

const (
	numZero    = 0
	numOne     = 1
	numTwo     = 2
	numThree   = 3
	numFour    = 4
	numFive    = 5
	numTen     = 10
	numHundred = 100

	testCertPath = "/etc/kubernetes/tls-server.crt"
	testKeyPath  = "/test/key/path"
)

type mockManager struct {
	ctrl.Manager
	metricsHandlers map[string]http.Handler
}

func (m *mockManager) AddMetricsExtraHandler(path string, handler http.Handler) error {
	if m.metricsHandlers == nil {
		m.metricsHandlers = make(map[string]http.Handler)
	}
	m.metricsHandlers[path] = handler
	return nil
}

func (m *mockManager) GetEventRecorderFor(name string) record.EventRecorder {
	return nil
}

func TestValidateTLSCertificatesWithCertNotExist(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Stat,
		nil,
		os.ErrNotExist,
	)
	defer patches.Reset()

	err := validateTLSCertificates(testCertPath, testKeyPath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "TLS certificate not found")
}

func TestValidateTLSCertificatesWithKeyNotExist(t *testing.T) {
	var callCount int
	patches := gomonkey.ApplyFunc(
		os.Stat,
		func(path string) (os.FileInfo, error) {
			callCount++
			if callCount == numOne {
				return nil, nil
			}
			return nil, os.ErrNotExist
		},
	)
	defer patches.Reset()

	err := validateTLSCertificates(testCertPath, testKeyPath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "TLS key not found")
}

func TestValidateTLSCertificatesSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Stat,
		nil,
		nil,
	)
	defer patches.Reset()

	err := validateTLSCertificates(testCertPath, testKeyPath)
	assert.NoError(t, err)
}

func TestLoadTLSConfigWithInvalidCert(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		tls.LoadX509KeyPair,
		tls.Certificate{},
		errors.New("failed to load cert"),
	)
	defer patches.Reset()

	_, err := loadTLSConfig(testCertPath, testKeyPath)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load TLS certificate")
}

func TestLoadTLSConfigSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		tls.LoadX509KeyPair,
		tls.Certificate{
			Certificate: [][]byte{[]byte("cert")},
		},
		nil,
	)
	defer patches.Reset()

	tlsConfig, err := loadTLSConfig(testCertPath, testKeyPath)
	assert.NoError(t, err)
	assert.NotNil(t, tlsConfig)
	assert.Equal(t, tls.VersionTLS12, int(tlsConfig.MinVersion))
}

func TestSetupHealthEndpoints(t *testing.T) {
	mux := http.NewServeMux()
	setupHealthEndpoints(mux)

	tests := []struct {
		name       string
		path       string
		expectCode int
		expectBody string
	}{
		{
			name:       "healthz success",
			path:       "/healthz",
			expectCode: http.StatusOK,
			expectBody: "ok",
		},
		{
			name:       "readyz success",
			path:       "/readyz",
			expectCode: http.StatusOK,
			expectBody: "ok",
		},
		{
			name:       "not found",
			path:       "/invalid",
			expectCode: http.StatusNotFound,
			expectBody: "",
		},
		{
			name:       "healthz with trailing slash",
			path:       "/healthz/",
			expectCode: http.StatusNotFound,
			expectBody: "",
		},
		{
			name:       "readyz with trailing slash",
			path:       "/readyz/",
			expectCode: http.StatusNotFound,
			expectBody: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			req, _ := http.NewRequest("GET", tt.path, nil)
			mux.ServeHTTP(recorder, req)
			assert.Equal(t, tt.expectCode, recorder.Code)
			if tt.expectBody != "" {
				assert.Contains(t, recorder.Body.String(), tt.expectBody)
			}
		})
	}
}

func TestEventAggregatorByMessageFunc(t *testing.T) {
	tests := []struct {
		name      string
		event     *corev1.Event
		expectKey string
		expectMsg string
	}{
		{
			name: "normal event",
			event: &corev1.Event{
				Source: corev1.EventSource{
					Component: "test-component",
					Host:      "test-host",
				},
				InvolvedObject: corev1.ObjectReference{
					Kind:       "Pod",
					Namespace:  "default",
					Name:       "test-pod",
					UID:        "test-uid",
					APIVersion: "v1",
				},
				Type:                "Normal",
				Message:             "test message",
				ReportingController: "test-controller",
				ReportingInstance:   "test-instance",
			},
			expectKey: "test-componenttest-hostPoddefaulttest-podtest-uidv1Normaltest messagetest-controllertest-instance",
			expectMsg: "test message",
		},
		{
			name: "event with empty fields",
			event: &corev1.Event{
				Source: corev1.EventSource{},
				InvolvedObject: corev1.ObjectReference{
					Kind:       "",
					Namespace:  "",
					Name:       "",
					UID:        "",
					APIVersion: "",
				},
				Type:                "",
				Message:             "",
				ReportingController: "",
				ReportingInstance:   "",
			},
			expectKey: "",
			expectMsg: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, msg := EventAggregatorByMessageFunc(tt.event)
			assert.Equal(t, tt.expectKey, key)
			assert.Equal(t, tt.expectMsg, msg)
		})
	}
}

func TestConcurrency(t *testing.T) {
	tests := []struct {
		name        string
		concurrency int
		expectMax   int
	}{
		{
			name:        "normal concurrency",
			concurrency: numTen,
			expectMax:   numTen,
		},
		{
			name:        "single concurrency",
			concurrency: numOne,
			expectMax:   numOne,
		},
		{
			name:        "hundred concurrency",
			concurrency: numHundred,
			expectMax:   numHundred,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := concurrency(tt.concurrency)
			assert.Equal(t, tt.expectMax, opts.MaxConcurrentReconciles)
			assert.NotNil(t, opts.RateLimiter)
			assert.True(t, opts.RecoverPanic != nil && *opts.RecoverPanic)
		})
	}
}

func TestStartHTTPSHealthServerWithValidateError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		validateTLSCertificates,
		errors.New("cert not found"),
	)
	defer patches.Reset()

	mgr := &mockManager{}
	ctx := context.Background()

	config.ProbeScheme = "https"
	config.ProbePort = numZero

	err := startHTTPSHealthServer(ctx, mgr)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cert not found")
}

func TestStartHTTPSHealthServerWithLoadTLSConfigError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		validateTLSCertificates,
		nil,
	)
	patches.ApplyFuncReturn(
		loadTLSConfig,
		nil,
		errors.New("load TLS config error"),
	)
	defer patches.Reset()

	mgr := &mockManager{}
	ctx := context.Background()

	config.ProbeScheme = "https"
	config.ProbePort = 9444

	err := startHTTPSHealthServer(ctx, mgr)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "load TLS config error")
}

func TestStartHTTPSHealthServerSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		validateTLSCertificates,
		nil,
	)
	patches.ApplyFuncReturn(
		loadTLSConfig,
		&tls.Config{},
		nil,
	)
	patches.ApplyFuncReturn(
		net.Listen,
		&net.TCPListener{},
		nil,
	)
	patches.ApplyFuncReturn(
		tls.NewListener,
		nil,
	)
	patches.ApplyFunc(
		(*http.Server).Serve,
		func(_ *http.Server, _ net.Listener) error {
			return http.ErrServerClosed
		},
	)
	defer patches.Reset()

	mgr := &mockManager{}
	ctx, cancel := context.WithCancel(context.Background())

	config.ProbeScheme = "https"
	config.ProbePort = 9444

	err := startHTTPSHealthServer(ctx, mgr)
	assert.NoError(t, err)

	cancel()
}

func TestStartHTTPSListenerWithListenError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		net.Listen,
		nil,
		errors.New("listen error"),
	)
	// Add mock for tls.NewListener and server.Serve to prevent interference from other goroutines
	patches.ApplyFuncReturn(
		tls.NewListener,
		&net.TCPListener{},
	)
	patches.ApplyFunc(
		(*http.Server).Serve,
		func(_ *http.Server, _ net.Listener) error {
			return http.ErrServerClosed
		},
	)
	defer patches.Reset()

	server := &http.Server{
		Addr: ":9444",
	}
	tlsConfig := &tls.Config{}

	startHTTPSListener(server, tlsConfig, testCertPath, testKeyPath)
}

func TestStartHTTPSListenerWithServerClosed(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		net.Listen,
		&net.TCPListener{},
		nil,
	)
	patches.ApplyFuncReturn(
		tls.NewListener,
		nil,
	)
	patches.ApplyFunc(
		(*http.Server).Serve,
		func(_ *http.Server, _ net.Listener) error {
			return http.ErrServerClosed
		},
	)
	defer patches.Reset()

	server := &http.Server{
		Addr: ":9444",
	}
	tlsConfig := &tls.Config{}

	startHTTPSListener(server, tlsConfig, testCertPath, testKeyPath)
}

func TestSetupGracefulShutdown(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	server := &http.Server{}

	setupGracefulShutdown(ctx, server)

	cancel()
}

func TestPrintManifestsBuildInfoWithSuccess(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		utils.GetManifestsBuildInfo,
		[]string{"BuildInfo1", "BuildInfo2"},
		nil,
	)
	defer patches.Reset()

	printManifestsBuildInfo()
}

func TestPrintManifestsBuildInfoWithError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		utils.GetManifestsBuildInfo,
		nil,
		errors.New("failed to get manifests"),
	)
	defer patches.Reset()

	printManifestsBuildInfo()
}

func TestPrintVersionInfo(t *testing.T) {
	printVersionInfo()
}

func TestSetupHealthChecksSuccess(t *testing.T) {
	patches := gomonkey.ApplyMethod(
		&mockManager{},
		"AddHealthzCheck",
		func(_ *mockManager, _ string, _ healthz.Checker) error {
			return nil
		},
	)
	patches.ApplyMethod(
		&mockManager{},
		"AddReadyzCheck",
		func(_ *mockManager, _ string, _ healthz.Checker) error {
			return nil
		},
	)
	defer patches.Reset()

	mgr := &mockManager{}
	setupHealthChecks(mgr)
}

func TestSetupHealthChecksWithHealthzError(t *testing.T) {
	patches := gomonkey.ApplyMethod(
		&mockManager{},
		"AddHealthzCheck",
		func(_ *mockManager, _ string, _ healthz.Checker) error {
			return errors.New("healthz error")
		},
	)
	patches.ApplyMethod(
		&mockManager{},
		"AddReadyzCheck",
		func(_ *mockManager, _ string, _ healthz.Checker) error {
			return nil
		},
	)
	patches.ApplyFunc(
		os.Exit,
		func(code int) {},
	)
	defer patches.Reset()

	mgr := &mockManager{}
	setupHealthChecks(mgr)
}

func TestSetupHealthChecksWithReadyzError(t *testing.T) {
	var callCount int
	patches := gomonkey.ApplyMethod(
		&mockManager{},
		"AddHealthzCheck",
		func(_ *mockManager, _ string, _ healthz.Checker) error {
			return nil
		},
	)
	patches.ApplyMethod(
		&mockManager{},
		"AddReadyzCheck",
		func(_ *mockManager, _ string, _ healthz.Checker) error {
			callCount++
			if callCount == numOne {
				return nil
			}
			return errors.New("readyz error")
		},
	)
	patches.ApplyFunc(
		os.Exit,
		func(code int) {},
	)
	defer patches.Reset()

	mgr := &mockManager{}
	setupHealthChecks(mgr)
}

func TestRegisterMetricWithMetricsAddrZero(t *testing.T) {
	config.MetricsAddr = "0"

	mgr := &mockManager{}
	registerMetric(mgr)
}

func TestRegisterMetricWithMetricsAddrNotZero(t *testing.T) {
	config.MetricsAddr = ":8080"

	patches := gomonkey.ApplyMethod(
		&mockManager{},
		"AddMetricsExtraHandler",
		func(_ *mockManager, _ string, _ http.Handler) error {
			return nil
		},
	)
	defer patches.Reset()

	mgr := &mockManager{}
	registerMetric(mgr)
}

func TestRegisterMetricWithMetricsAddrNotZeroError(t *testing.T) {
	config.MetricsAddr = ":8080"

	patches := gomonkey.ApplyMethod(
		&mockManager{},
		"AddMetricsExtraHandler",
		func(_ *mockManager, _ string, _ http.Handler) error {
			return errors.New("add metrics error")
		},
	)
	patches.ApplyFunc(
		os.Exit,
		func(code int) {},
	)
	defer patches.Reset()

	mgr := &mockManager{}
	registerMetric(mgr)
}

func TestRegisterProfilerWithDebugTrue(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"true",
	)
	defer patches.Reset()

	config.MetricsAddr = "0"

	mgr := &mockManager{}
	registerProfiler(mgr)
}

func TestRegisterProfilerWithMetricsAddrNotZero(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"",
	)
	patches.ApplyMethod(
		&mockManager{},
		"AddMetricsExtraHandler",
		func(_ *mockManager, _ string, _ http.Handler) error {
			return nil
		},
	)
	defer patches.Reset()

	config.MetricsAddr = ":8080"

	mgr := &mockManager{}
	registerProfiler(mgr)
}

func TestRegisterProfilerWithDebugFalseAndMetricsAddrZero(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"false",
	)
	defer patches.Reset()

	config.MetricsAddr = "0"

	mgr := &mockManager{}
	registerProfiler(mgr)
}

func TestRegisterProfilerWithError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(
		os.Getenv,
		"",
	)
	patches.ApplyMethod(
		&mockManager{},
		"AddMetricsExtraHandler",
		func(_ *mockManager, _ string, _ http.Handler) error {
			return errors.New("add metrics error")
		},
	)
	patches.ApplyFunc(
		os.Exit,
		func(code int) {},
	)
	defer patches.Reset()

	config.MetricsAddr = ":8080"

	mgr := &mockManager{}
	registerProfiler(mgr)
}
