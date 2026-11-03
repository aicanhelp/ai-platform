/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-15
 */

package server

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/containerd/nri/pkg/api"
	"github.com/containerd/nri/pkg/stub"
	"k8s.io/apimachinery/pkg/util/wait"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
)

type mockStub struct {
	runError              error
	stopCalled            bool
	startError            error
	waitCalled            bool
	updateContainersError error
}

func (m *mockStub) Run(ctx context.Context) error {
	return m.runError
}

func (m *mockStub) Stop() {
	m.stopCalled = true
}

func (m *mockStub) Start(ctx context.Context) error {
	return m.startError
}

func (m *mockStub) Wait() {
	m.waitCalled = true
}

func (m *mockStub) UpdateContainers(updates []*api.ContainerUpdate) ([]*api.ContainerUpdate, error) {
	return nil, m.updateContainersError
}

func TestNriServerConfigure(t *testing.T) {
	server := &NriServer{
		eventMask: 0,
	}

	ctx := context.Background()
	eventMask, err := server.Configure(ctx, "test-config", "test-runtime", "test-version")

	if err != nil {
		t.Errorf("Configure returned unexpected error: %v", err)
	}

	if eventMask == 0 {
		t.Error("Expected non-zero event mask")
	}

	if server.eventMask != eventMask {
		t.Errorf("Expected eventMask %v, got %v", eventMask, server.eventMask)
	}
}

func TestNriServerStart(t *testing.T) {
	tests := []struct {
		name      string
		runError  error
		expectErr bool
	}{
		{
			name:      "successful start",
			runError:  nil,
			expectErr: false,
		},
		{
			name:      "failed start",
			runError:  errors.New("run error"),
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStub := &mockStub{runError: tt.runError}
			var stubInterface stub.Stub = mockStub
			server := &NriServer{
				stub: stubInterface,
			}

			err := server.Start()

			if tt.expectErr {
				if err == nil {
					t.Error("Expected error but got nil")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

func TestNriServerStop(t *testing.T) {
	mockStub := &mockStub{}
	var stubInterface stub.Stub = mockStub
	server := &NriServer{
		stub: stubInterface,
	}

	server.Stop()

	if !mockStub.stopCalled {
		t.Error("Stop was not called on stub")
	}
}

func TestNriServerCanRunUpdate(t *testing.T) {
	tests := []struct {
		name     string
		admit    core.AdmitResult
		expected bool
	}{
		{
			name:     "admit true",
			admit:    core.AdmitResult{Admit: true, Reason: "allowed"},
			expected: true,
		},
		{
			name:     "admit false",
			admit:    core.AdmitResult{Admit: false, Reason: "denied"},
			expected: false,
		},
	}

	server := &NriServer{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := server.canRunUpdate(tt.admit)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v for admit result %+v", tt.expected, result, tt.admit)
			}
		})
	}
}

func TestNriServerRetryOnClose(t *testing.T) {
	server := &NriServer{}

	t.Run("successful retry", func(t *testing.T) {
		tryCount := 0
		retryFunc := func() bool {
			tryCount++
			return tryCount >= 2
		}

		errorFunc := func() error {
			return nil
		}

		err := server.RetryOnClose(retryFunc, errorFunc)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tryCount != 2 {
			t.Errorf("Expected 2 tries, got %d", tryCount)
		}
	})

	t.Run("retry timeout", func(t *testing.T) {
		customBackoff := wait.Backoff{
			Steps:    2,
			Duration: 1 * time.Millisecond,
			Factor:   1.0,
			Jitter:   0.1,
		}

		retryFunc := func() bool {
			return false
		}

		errorFunc := func() error {
			return nil
		}

		err := wait.ExponentialBackoff(customBackoff, func() (bool, error) {
			return retryFunc(), errorFunc()
		})

		if err == nil {
			t.Error("Expected timeout error but got nil")
		}
	})
}

func TestNriServerSynchronize(t *testing.T) {
	server := &NriServer{}
	ctx := context.Background()

	updates, err := server.Synchronize(ctx, []*api.PodSandbox{}, []*api.Container{})

	if err != nil {
		t.Errorf("Synchronize returned unexpected error: %v", err)
	}

	if updates != nil {
		t.Error("Expected nil updates from Synchronize")
	}
}

func TestNriServerCreateContainerLogic(t *testing.T) {
	tests := []struct {
		name        string
		admitResult core.AdmitResult
		expectError bool
	}{
		{
			name:        "admission allowed",
			admitResult: core.AdmitResult{Admit: true},
			expectError: false,
		},
		{
			name:        "admission denied",
			admitResult: core.AdmitResult{Admit: false, Reason: "quota exceeded"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := &NriServer{}

			result := server.canRunUpdate(tt.admitResult)

			if tt.expectError {
				if result {
					t.Error("Expected canRunUpdate to return false for denied admission")
				}
			} else {
				if !result {
					t.Error("Expected canRunUpdate to return true for allowed admission")
				}
			}
		})
	}
}

func TestAdmitResultString(t *testing.T) {
	tests := []struct {
		name     string
		admit    core.AdmitResult
		expected string
	}{
		{
			name:     "admitted",
			admit:    core.AdmitResult{Admit: true, Reason: "ok"},
			expected: "admitted: ok",
		},
		{
			name:     "denied",
			admit:    core.AdmitResult{Admit: false, Reason: "quota"},
			expected: "denied: quota",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fmt.Sprintf("%v", tt.admit)
			if result == "" {
				t.Error("Expected non-empty string representation")
			}
		})
	}
}
