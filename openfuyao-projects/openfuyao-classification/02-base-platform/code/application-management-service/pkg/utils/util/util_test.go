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

// Package util include application-management-service level util function
package util

import (
	"errors"
	"io"
	"testing"

	"application-management-service/pkg/constant"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

// TestClearByte checks if all bytes in the slice are set to zero.
func TestClearByte(t *testing.T) {
	// Create a byte slice with non-zero values.
	data := []byte{1, 2, 3, 4, 5}

	// Call ClearByte to zero out the slice.
	ClearByte(data)

	// Check each byte to ensure it's been set to zero.
	for i, b := range data {
		if b != 0 {
			t.Errorf("byte at index %d is not zero, got %d", i, b)
		}
	}

	var emptyData []byte
	ClearByte(emptyData) // This should not cause any issue or panic.
}

// MockFile implement multipart.File interface for testing
type MockFile struct {
	Data   []byte
	Offset int
	Err    error // 可以被设置为模拟读取错误
}

func (m *MockFile) Read(p []byte) (int, error) {
	if m.Err != nil {
		return 0, m.Err
	}
	if m.Offset >= len(m.Data) {
		return 0, io.EOF
	}
	n := copy(p, m.Data[m.Offset:])
	m.Offset += n
	return n, nil
}

func (m *MockFile) Close() error {
	return nil
}

func (m *MockFile) Seek(offset int64, whence int) (int64, error) {
	return 0, nil
}

func (m *MockFile) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("negative offset")
	}
	if int(off) >= len(m.Data) {
		return 0, io.EOF
	}
	n := copy(p, m.Data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func TestContains(t *testing.T) {
	// 定义一组测试用例
	tests := []struct {
		name     string
		slice    []string
		str      string
		expected bool
	}{
		{"Found", []string{"apple", "banana", "cherry"}, "banana", true},
		{"Not Found", []string{"apple", "banana", "cherry"}, "mango", false},
		{"Empty Slice", []string{}, "banana", false},
		{"Empty String", []string{"apple", "banana", "cherry"}, "", false},
		{"Nil Slice", nil, "banana", false},
		{"Looking for Nil", []string{"apple", "banana", "cherry"}, "", false},
	}

	// 循环执行每个测试用例
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Contains(tt.slice, tt.str)
			if result != tt.expected {
				t.Errorf("Contains(%v, %q) = %v, expected %v", tt.slice, tt.str, result, tt.expected)
			}
		})
	}
}

func TestContainsOne(t *testing.T) {
	tests := []struct {
		name     string
		leftArr  []string
		rightArr []string
		expected bool
	}{
		{"Contains One", []string{"apple", "banana", "cherry"}, []string{"banana", "mango"}, true},
		{"Contains Multiple", []string{"apple", "banana", "cherry"}, []string{"banana", "cherry"}, true},
		{"Contains None", []string{"apple", "banana", "cherry"}, []string{"mango", "orange"}, false},
		{"Left Empty", []string{}, []string{"mango", "orange"}, false},
		{"Right Empty", []string{"apple", "banana", "cherry"}, []string{}, false},
		{"Both Empty", []string{}, []string{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsOne(tt.leftArr, tt.rightArr)
			if result != tt.expected {
				t.Errorf("%s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

func TestContainsAll(t *testing.T) {
	tests := []struct {
		name     string
		outer    []string
		inner    []string
		expected bool
	}{
		{"All Contained", []string{"apple", "banana", "cherry"}, []string{"banana", "apple"}, true},
		{"Missing Some", []string{"apple", "banana", "cherry"}, []string{"banana", "mango"}, false},
		{"Inner Empty", []string{"apple", "banana", "cherry"}, []string{}, true},
		{"Outer Empty", []string{}, []string{"banana", "apple"}, false},
		{"Both Empty", []string{}, []string{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsAll(tt.outer, tt.inner)
			if result != tt.expected {
				t.Errorf("%s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

func TestParseStringSlice_Success(t *testing.T) {
	got, err := parseStringSlice("- a\n- b\n")
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Unexpected result: %v", got)
	}
}

func TestParseStringSlice_Error(t *testing.T) {
	_, err := parseStringSlice("[")
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
}

func TestParseConfig_Success(t *testing.T) {
	m := map[string]string{
		"marketplace-service-host": "http://marketplace-service",
		"helm-timeout":             "5m",
		"helm-wait-charts":         "- chart1\n- chart2\n",
	}

	cfg := parseConfig(m)
	if cfg == nil {
		t.Fatalf("Expected non-nil config")
	}
	if cfg.MarketPlaceServiceHost != "http://marketplace-service" {
		t.Fatalf("Unexpected MarketPlaceServiceHost: %s", cfg.MarketPlaceServiceHost)
	}
	if cfg.HelmTimeout != "5m" {
		t.Fatalf("Unexpected HelmTimeout: %s", cfg.HelmTimeout)
	}
	if len(cfg.HelmWaitCharts) != 2 || cfg.HelmWaitCharts[0] != "chart1" || cfg.HelmWaitCharts[1] != "chart2" {
		t.Fatalf("Unexpected HelmWaitCharts: %v", cfg.HelmWaitCharts)
	}
}

func TestParseConfig_WaitChartsInvalid(t *testing.T) {
	m := map[string]string{
		"marketplace-service-host": "http://marketplace-service",
		"helm-timeout":             "5m",
		"helm-wait-charts":         "[",
	}

	cfg := parseConfig(m)
	if cfg == nil {
		t.Fatalf("Expected non-nil config")
	}
	if cfg.MarketPlaceServiceHost != "http://marketplace-service" {
		t.Fatalf("Unexpected MarketPlaceServiceHost: %s", cfg.MarketPlaceServiceHost)
	}
	if cfg.HelmTimeout != "5m" {
		t.Fatalf("Unexpected HelmTimeout: %s", cfg.HelmTimeout)
	}
	if len(cfg.HelmWaitCharts) != 0 {
		t.Fatalf("Expected empty HelmWaitCharts, got %v", cfg.HelmWaitCharts)
	}
}

func TestTrimOpenFuyaoRolePrefix(t *testing.T) {
	got := TrimOpenFuyaoRolePrefix("openfuyao-admin")
	if got != "admin" {
		t.Fatalf("Expected admin, got %s", got)
	}

	got = TrimOpenFuyaoRolePrefix("admin")
	if got != "admin" {
		t.Fatalf("Expected admin, got %s", got)
	}
}

func TestGetApplicationManagementServiceConfig_Success(t *testing.T) {
	const ns = constant.ApplicationManagementServiceDefaultNamespace
	clientset := fake.NewSimpleClientset(
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.ApplicationManagementServiceConfigmap,
				Namespace: ns,
			},
			Data: map[string]string{
				// keys are defined in pkg/utils/util/util.go
				"marketplace-service-host": "http://marketplace-service",
				"helm-timeout":             "5s",
				"helm-wait-charts":         "- chart1\n- chart2\n",
			},
		},
	)

	cfg, err := GetApplicationManagementServiceConfig(clientset)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if cfg == nil {
		t.Fatalf("expected non-nil config")
	}
	if cfg.MarketPlaceServiceHost != "http://marketplace-service" {
		t.Fatalf("unexpected MarketPlaceServiceHost: %s", cfg.MarketPlaceServiceHost)
	}
	if cfg.HelmTimeout != "5s" {
		t.Fatalf("unexpected HelmTimeout: %s", cfg.HelmTimeout)
	}
	if len(cfg.HelmWaitCharts) != 2 || cfg.HelmWaitCharts[0] != "chart1" || cfg.HelmWaitCharts[1] != "chart2" {
		t.Fatalf("unexpected HelmWaitCharts: %#v", cfg.HelmWaitCharts)
	}
}

func TestGetApplicationManagementServiceConfig_InvalidWaitChartsFromConfigMap(t *testing.T) {
	const ns = constant.ApplicationManagementServiceDefaultNamespace
	clientset := fake.NewSimpleClientset(
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.ApplicationManagementServiceConfigmap,
				Namespace: ns,
			},
			Data: map[string]string{
				"marketplace-service-host": "http://marketplace-service",
				"helm-timeout":             "30s",
				"helm-wait-charts":         "[",
			},
		},
	)

	cfg, err := GetApplicationManagementServiceConfig(clientset)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if cfg == nil {
		t.Fatalf("expected non-nil config")
	}
	if cfg.MarketPlaceServiceHost != "http://marketplace-service" {
		t.Fatalf("unexpected MarketPlaceServiceHost: %s", cfg.MarketPlaceServiceHost)
	}
	if cfg.HelmTimeout != "30s" {
		t.Fatalf("unexpected HelmTimeout: %s", cfg.HelmTimeout)
	}
	if len(cfg.HelmWaitCharts) != 0 {
		t.Fatalf("expected empty HelmWaitCharts, got %#v", cfg.HelmWaitCharts)
	}
}

func TestGetApplicationManagementServiceConfig_ConfigMapMissingKeys(t *testing.T) {
	const ns = constant.ApplicationManagementServiceDefaultNamespace
	clientset := fake.NewSimpleClientset(
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.ApplicationManagementServiceConfigmap,
				Namespace: ns,
			},
			Data: map[string]string{},
		},
	)

	cfg, err := GetApplicationManagementServiceConfig(clientset)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if cfg == nil {
		t.Fatalf("expected non-nil config")
	}
	if cfg.MarketPlaceServiceHost != "" || cfg.HelmTimeout != "" {
		t.Fatalf("expected empty host and timeout, got host=%q timeout=%q", cfg.MarketPlaceServiceHost, cfg.HelmTimeout)
	}
	if len(cfg.HelmWaitCharts) != 0 {
		t.Fatalf("expected empty HelmWaitCharts, got %#v", cfg.HelmWaitCharts)
	}
}

func TestGetApplicationManagementServiceConfig_FallbackPath(t *testing.T) {
	clientset := fake.NewSimpleClientset()

	cfg, err := GetApplicationManagementServiceConfig(clientset)
	if err != nil {
		if cfg != nil {
			t.Fatalf("expected nil config when error occurs, got %#v", cfg)
		}
		return
	}

	if cfg == nil {
		t.Fatalf("expected non-nil config when no error")
	}
}
