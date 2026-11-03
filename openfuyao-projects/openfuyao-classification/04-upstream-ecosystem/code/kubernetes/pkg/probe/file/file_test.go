// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

package file

import (
	"os"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/probe"
)

func TestFileProbe(t *testing.T) {
	// Setup test cases
	tests := []struct {
		name        string
		setup       func() string // returns file path
		validity    int32         // in seconds
		expected    probe.Result
		expectError bool
	}{
		{
			name: "fresh file within validity period",
			setup: func() string {
				f, _ := os.CreateTemp("", "probe-*")
				f.Close()
				return f.Name()
			},
			validity:    30,
			expected:    probe.Success,
			expectError: false,
		},
		{
			name: "expired file beyond validity period",
			setup: func() string {
				f, _ := os.CreateTemp("", "probe-*")
				f.Close()
				// Set modification time to 1 hour ago
				oldTime := time.Now().Add(-1 * time.Hour)
				os.Chtimes(f.Name(), oldTime, oldTime)
				return f.Name()
			},
			validity:    30, // 30 seconds
			expected:    probe.Failure,
			expectError: false,
		},
		{
			name: "non-existent file",
			setup: func() string {
				return "nonexistent-file"
			},
			validity:    30,
			expected:    probe.Failure,
			expectError: true,
		},
	}

	prober := New()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup test file
			filePath := tt.setup()
			defer func() {
				if filePath != "nonexistent-file" {
					os.Remove(filePath)
				}
			}()

			// Execute probe
			result, msg, err := prober.Probe(filePath, tt.validity)

			// Verify error expectation
			if tt.expectError && err == nil {
				t.Errorf("Expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			// Verify result
			if result != tt.expected {
				t.Errorf("Expected result %v, got %v", tt.expected, result)
			}

			// Verify message for error cases
			if tt.expectError && msg == "" {
				t.Error("Expected error message but got empty string")
			}
		})
	}
}

func TestFileProbeEdgeCases(t *testing.T) {
	prober := New()

	t.Run("file not exists", func(t *testing.T) {
		result, _, err := prober.Probe("/nonexistent/file", 30)
		if err == nil || result != probe.Failure {
			t.Errorf("Expected failure for non-existent file, got result: %v, err: %v", result, err)
		}
	})

	t.Run("valid file within validity period", func(t *testing.T) {
		f, _ := os.CreateTemp("", "probe-*")
		defer os.Remove(f.Name())
		f.Close()

		result, _, err := prober.Probe(f.Name(), 3600) // 1小时有效期
		if err != nil || result != probe.Success {
			t.Errorf("Expected success for valid file, got result: %v, err: %v", result, err)
		}
	})

	t.Run("expired file", func(t *testing.T) {
		f, _ := os.CreateTemp("", "probe-*")
		defer os.Remove(f.Name())
		f.Close()

		// 修改文件时间为过去
		oldTime := time.Now().Add(-2 * time.Hour)
		os.Chtimes(f.Name(), oldTime, oldTime)

		result, _, err := prober.Probe(f.Name(), 1800) // 30分钟有效期
		if err != nil || result != probe.Failure {
			t.Errorf("Expected failure for expired file, got result: %v, err: %v", result, err)
		}
	})
}
