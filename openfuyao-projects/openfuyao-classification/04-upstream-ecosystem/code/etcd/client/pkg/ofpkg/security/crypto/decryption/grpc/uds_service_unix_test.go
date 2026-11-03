/*
Copyright 2017 The Kubernetes Authors.
Copyright (c) 2025 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package grpc

import (
	"fmt"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/api"
	mock "go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/testing/v1"
)

type testSocket struct {
	path     string
	endpoint string
}

// newEndpoint constructs a unique name for a Linux Abstract Socket to be used in a test.
// This package uses Linux Domain Sockets to remove the need for clean-up of socket files.
func newEndpoint() *testSocket {
	p := fmt.Sprintf("@%s.sock", uuid.New().String())

	return &testSocket{
		path:     p,
		endpoint: fmt.Sprintf("unix:///%s", p),
	}
}

// TestDecryptionPluginLateStart tests the scenario where decryption-plugin pod/container starts after k8s pod/container.
// Since the Dial to decryption-plugin is non-blocking we expect the construction of gRPC service to succeed even when
// decryption-plugin is not yet up - dialing happens in the background.
func TestDecryptionPluginLateStart(t *testing.T) {
	t.Parallel()
	callTimeout := 3 * time.Second
	s := newEndpoint()

	service, err := CreateGRPCService(s.endpoint, callTimeout)
	if err != nil {
		t.Fatalf("failed to create decryption service, error: %v", err)
	}
	defer service.Release()

	time.Sleep(callTimeout / 2) //模拟gRPC服务正在启动过程
	tempDir, cleanFunc, err := createTmpDirWithClean("TestDecryptionPluginLateStart")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
		return
	}
	defer cleanFunc()
	f, err := mock.NewFileMapPlugin(s.path, tempDir)
	if err != nil {
		t.Fatalf("failed to start test Decryption provider server, error: %v", err)
	}
	if err := f.Start(true); err != nil {
		t.Fatalf("Failed to start decryption-plugin, err: %v", err)
	}
	defer f.CleanUp()

	data := []byte(mock.DefaultKey)
	_, err = service.DecryptSecret(data)
	if err != nil {
		t.Fatalf("failed when execute decrypt, error: %v", err)
	}
}

// TestTimeouts TestTimeout tests behaviour of the k8s component based on the supplied timeout and delayed start of decryption-plugin.
func TestTimeouts(t *testing.T) {
	t.Parallel()
	var testCases = []struct {
		desc              string
		callTimeout       time.Duration
		pluginDelay       time.Duration
		k8sComponentDelay time.Duration
		wantErr           string
	}{
		{
			desc:        "timeout zero - expect failure when call from k8s arrives before plugin starts",
			callTimeout: 0 * time.Second,
			pluginDelay: 3 * time.Second,
			wantErr:     "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
		},
		{
			desc:              "timeout zero but decrypt-plugin already up - still failure - zero timeout is an invalid value",
			callTimeout:       0 * time.Second,
			pluginDelay:       0 * time.Second,
			k8sComponentDelay: 2 * time.Second,
			wantErr:           "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
		},
		{
			desc:        "timeout greater than decrypt-plugin delay - expect success",
			callTimeout: 6 * time.Second,
			pluginDelay: 3 * time.Second,
		},
		{
			desc:        "timeout less than decrypt-plugin delay - expect failure",
			callTimeout: 3 * time.Second,
			pluginDelay: 6 * time.Second,
			wantErr:     "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.desc, func(t *testing.T) {
			t.Parallel()
			var (
				service         api.Service
				err             error
				data            = []byte(mock.DefaultKey)
				k8sComponentWG  sync.WaitGroup
				decryptPluginWG sync.WaitGroup
				testCompletedWG sync.WaitGroup
				socketName      = newEndpoint()
			)

			testCompletedWG.Add(1)
			defer testCompletedWG.Done()

			k8sComponentWG.Add(1)
			go func() {
				// Simulating late start of k8s component - plugin is up before k8s, if requested by the testcase.
				time.Sleep(tt.k8sComponentDelay)

				service, err = CreateGRPCService(socketName.endpoint, tt.callTimeout)
				if err != nil {
					t.Fatalf("failed to create decryption service, error: %v", err)
					return
				}
				defer service.Release()
				k8sComponentWG.Done()
				// Keeping k8s component up to process requests.
				testCompletedWG.Wait()
			}()

			decryptPluginWG.Add(1)
			go func() {
				// Simulating delayed start of decrypt-plugin, k8s is up before the plugin, if requested by the testcase.
				time.Sleep(tt.pluginDelay)
				tempDir, cleanFunc, err := createTmpDirWithClean("TestTimeouts")
				if err != nil {
					t.Fatalf("failed to create temp dir, error: %v", err)
				}
				defer cleanFunc()

				f, err := mock.NewFileMapPlugin(socketName.path, tempDir)
				if err != nil {
					t.Fatalf("failed to construct test Decryption provider server, error: %v", err)
				}
				if err := f.Start(true); err != nil {
					t.Fatalf("Failed to start test Decryption provider server, error: %v", err)
				}
				defer f.CleanUp()
				decryptPluginWG.Done()
				// Keeping plugin up to process requests.
				testCompletedWG.Wait()
			}()

			k8sComponentWG.Wait()
			_, err = service.DecryptCert(data)

			if err == nil && tt.wantErr != "" {
				t.Fatalf("got nil, want %s", tt.wantErr)
			}

			if err != nil && tt.wantErr == "" {
				t.Fatalf("got %q, want nil", err.Error())
			}

			// Collecting decrypt-plugin - allowing plugin to clean-up.
			decryptPluginWG.Wait()
		})
	}
}

// TestIntermittentConnectionLoss tests the scenario where the connection with decrypt-plugin is intermittently lost.
func TestIntermittentConnectionLoss(t *testing.T) {
	t.Parallel()
	var (
		wg1        sync.WaitGroup
		wg2        sync.WaitGroup
		timeout    = 30 * time.Second
		blackOut   = 1 * time.Second
		data       = []byte(mock.DefaultKey)
		endpoint   = newEndpoint()
		decryptErr error
	)
	// Start Decryption Plugin
	tempDir, cleanFunc, err := createTmpDirWithClean("TestIntermittentConnectionLoss-1")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
	}
	defer cleanFunc()
	f, err := mock.NewFileMapPlugin(endpoint.path, tempDir)
	if err != nil {
		t.Fatalf("failed to start test Decryption provider server, error: %v", err)
	}
	if err := f.Start(true); err != nil {
		t.Fatalf("Failed to start decrypt-plugin, err: %v", err)
	}

	//  connect to decrypt plugin
	service, err := CreateGRPCService(endpoint.endpoint, timeout)
	if err != nil {
		t.Fatalf("failed to create decryption service, error: %v", err)
	}
	defer service.Release()

	_, err = service.DecryptSecret(data)
	if err != nil {
		t.Fatalf("failed when execute decrypt, error: %v", err)
	}
	t.Log("Connected to DecryptionPlugin")

	// Stop Decryption Plugin - simulating connection loss
	t.Log("Decryption Plugin is stopping")
	f.CleanUp()
	time.Sleep(2 * time.Second)

	wg1.Add(1)
	wg2.Add(1)
	go func() {
		defer wg2.Done()
		// Call service to decrypt data.
		t.Log("Sending decrypt request")
		wg1.Done()
		_, err := service.DecryptCert(data)
		if err != nil {
			decryptErr = fmt.Errorf("failed when executing decrypt, error: %v", err)
		}
	}()

	wg1.Wait()
	time.Sleep(blackOut)
	// Start Decryption Plugin
	tempDir, cleanFunc, err = createTmpDirWithClean("TestIntermittentConnectionLoss-2")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
	}
	defer cleanFunc()
	f, err = mock.NewFileMapPlugin(endpoint.path, tempDir)
	if err != nil {
		t.Fatalf("failed to start test Decryption provider server, error: %v", err)
	}
	if err := f.Start(true); err != nil {
		t.Fatalf("Failed to start decrypt-plugin, err: %v", err)
	}
	defer f.CleanUp()
	t.Log("Restarted Decryption Plugin")

	wg2.Wait()

	if decryptErr != nil {
		t.Error(decryptErr)
	}
}

// TestGRPCService Normal decryption operation.
func TestGRPCService(t *testing.T) {
	t.Parallel()
	// Start a test gRPC server.
	endpoint := newEndpoint()
	tempDir, cleanFunc, err := createTmpDirWithClean("TestGRPCService")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
	}
	defer cleanFunc()
	f, err := mock.NewFileMapPlugin(endpoint.path, tempDir)
	if err != nil {
		t.Fatalf("failed to construct test Decryption provider server, error: %v", err)
	}
	if err := f.Start(false); err != nil {
		t.Fatalf("Failed to start decrypt-plugin, err: %v", err)
	}
	defer f.CleanUp()

	// Create the gRPC client service.
	service, err := CreateGRPCService(endpoint.endpoint, 1*time.Second)
	if err != nil {
		t.Fatalf("failed to create decryption service, error: %v", err)
	}
	defer service.Release()

	mockKey := "mock-key"
	mockValue := "mock-value"
	err = f.PushMockData(map[string][]byte{mockKey: []byte(mockValue)})
	if err != nil {
		t.Fatalf("failed to push mock data, error: %v", err)
	}
	// Call service to decrypt data.
	result, err := service.DecryptCert([]byte(mockKey))
	if err != nil {
		t.Fatalf("failed when execute decrypt, error: %v", err)
	}

	if !reflect.DeepEqual(mockValue, string(result)) {
		t.Errorf("expect: %v, but: %v", mockValue, result)
	}
}

// TestGRPCServiceConcurrentAccess Normal decryption operation by multiple go-routines.
func TestGRPCServiceConcurrentAccess(t *testing.T) {
	t.Parallel()
	// Start a test gRPC server.
	endpoint := newEndpoint()
	tempDir, cleanFunc, err := createTmpDirWithClean("TestGRPCServiceConcurrentAccess")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
	}
	defer cleanFunc()
	f, err := mock.NewFileMapPlugin(endpoint.path, tempDir)
	if err != nil {
		t.Fatalf("failed to start test Decryption provider server, error: %v", err)
	}
	if err := f.Start(true); err != nil {
		t.Fatalf("Failed to start decrypt-plugin, err: %v", err)
	}
	defer f.CleanUp()

	// Create the gRPC client service.
	service, err := CreateGRPCService(endpoint.endpoint, 15*time.Second)
	if err != nil {
		t.Fatalf("failed to create decryption service, error: %v", err)
	}
	defer service.Release()

	var wg sync.WaitGroup
	n := 100
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			// Call service to decrypt data.
			data := []byte(mock.DefaultKey)
			result, err := service.DecryptSecret(data)
			if err != nil {
				t.Errorf("failed when execute decrypt, error: %v", err)
			}

			if !reflect.DeepEqual([]byte(mock.DefaultValue), result) {
				t.Errorf("expect: %v, but: %v", mock.DefaultValue, result)
			}
		}()
	}

	wg.Wait()
}

// TestInvalidConfiguration Test all those invalid configuration for Decryption provider.
func TestInvalidConfiguration(t *testing.T) {
	t.Parallel()
	// Start a test gRPC server.
	tempDir, cleanFunc, err := createTmpDirWithClean("TestInvalidConfiguration")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
	}
	defer cleanFunc()
	f, err := mock.NewFileMapPlugin(newEndpoint().path, tempDir)
	if err != nil {
		t.Fatalf("failed to start test Decryption provider server, error: %v", err)
	}
	if err := f.Start(false); err != nil {
		t.Fatalf("Failed to start decrypt-plugin, err: %v", err)
	}
	defer f.CleanUp()

	invalidConfigs := []struct {
		name     string
		endpoint string
	}{
		{"emptyConfiguration", ""},
		{"invalidScheme", "tcp://localhost:6060"},
	}

	for _, testCase := range invalidConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := CreateGRPCService(testCase.endpoint, 1*time.Second)
			if err == nil {
				t.Fatalf("should fail to create decryption service for %s.", testCase.name)
			}
		})
	}
}

func createTmpDirWithClean(key string) (string, func(), error) {
	tempDir, err := os.MkdirTemp("", key+"-*-decrypt-plugin-test")
	if err != nil {
		return "", nil, err
	}
	return tempDir, func() {
		_ = os.RemoveAll(tempDir)
	}, nil
}
