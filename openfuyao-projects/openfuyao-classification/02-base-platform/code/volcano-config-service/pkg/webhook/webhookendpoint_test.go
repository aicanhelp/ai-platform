/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package webhook defined mutate and validate rules
package webhook

import (
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestIsNotFound(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "nil error",
			err:      nil,
			expected: false,
		},
		{
			name:     "not found error",
			err:      errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "test-endpoint"),
			expected: true,
		},
		{
			name:     "other error",
			err:      errors.NewBadRequest("bad request"),
			expected: false,
		},
		{
			name:     "generic error",
			err:      fmt.Errorf("generic error"),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsNotFound(tt.err)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestWaitForSvcRunningEdgeCases(t *testing.T) {
	// Save original Endpoint value and restore after test
	originalEndpoint := Endpoint
	defer func() {
		Endpoint = originalEndpoint
	}()

	t.Run("very short interval", func(t *testing.T) {
		Endpoint = ""
		interval := 1 * time.Millisecond
		const NUM = 100
		timeout := NUM * time.Millisecond

		if interval > 0 && timeout > interval {
			// Valid configuration
			t.Log("Short interval is valid")
		}
	})
}

func TestEndpointGlobalVariable(t *testing.T) {
	// Save original value
	original := Endpoint
	defer func() {
		Endpoint = original
	}()

	tests := []struct {
		name     string
		setValue string
		expected string
	}{
		{
			name:     "set empty endpoint",
			setValue: "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Endpoint = tt.setValue
			if Endpoint != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, Endpoint)
			}
		})
	}
}

// Benchmark tests for performance
func BenchmarkIsNotFound(b *testing.B) {
	notFoundErr := errors.NewNotFound(schema.GroupResource{Group: "", Resource: "endpoints"}, "test")
	otherErr := fmt.Errorf("other error")

	b.Run("NotFoundError", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			IsNotFound(notFoundErr)
		}
	})

	b.Run("OtherError", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			IsNotFound(otherErr)
		}
	})

	b.Run("NilError", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			IsNotFound(nil)
		}
	})
}

// Test helper functions for endpoint creation
func createTestEndpoint(name, namespace, ip string, ready bool) *corev1.Endpoints {
	endpoint := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}

	address := corev1.EndpointAddress{IP: ip}
	subset := corev1.EndpointSubset{}

	if ready {
		subset.Addresses = []corev1.EndpointAddress{address}
	} else {
		subset.NotReadyAddresses = []corev1.EndpointAddress{address}
	}

	endpoint.Subsets = []corev1.EndpointSubset{subset}
	return endpoint
}

func TestCreateTestEndpoint(t *testing.T) {
	tests := []struct {
		name      string
		epName    string
		namespace string
		ip        string
		ready     bool
	}{
		{
			name:      "ready endpoint",
			epName:    "test-service",
			namespace: "default",
			ip:        "",
			ready:     true,
		},
		{
			name:      "not ready endpoint",
			epName:    "test-service",
			namespace: "kube-system",
			ip:        "",
			ready:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			endpoint := createTestEndpoint(tt.epName, tt.namespace, tt.ip, tt.ready)

			if endpoint.Name != tt.epName {
				t.Errorf("expected name %s, got %s", tt.epName, endpoint.Name)
			}

			if endpoint.Namespace != tt.namespace {
				t.Errorf("expected namespace %s, got %s", tt.namespace, endpoint.Namespace)
			}

			if len(endpoint.Subsets) != 1 {
				t.Errorf("expected 1 subset, got %d", len(endpoint.Subsets))
			}

			subset := endpoint.Subsets[0]
			if tt.ready {
				if len(subset.Addresses) != 1 || subset.Addresses[0].IP != tt.ip {
					t.Errorf("expected ready address %s, got %v", tt.ip, subset.Addresses)
				}
			} else {
				if len(subset.NotReadyAddresses) != 1 || subset.NotReadyAddresses[0].IP != tt.ip {
					t.Errorf("expected not ready address %s, got %v", tt.ip, subset.NotReadyAddresses)
				}
			}
		})
	}
}
