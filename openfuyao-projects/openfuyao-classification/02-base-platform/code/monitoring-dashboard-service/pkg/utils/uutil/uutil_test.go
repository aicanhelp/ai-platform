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

package uutil

import (
	"errors"
	"testing"
	"time"
)

func TestGenerateID(t *testing.T) {
	// Test case 1: Generate a new ID
	id1 := GenerateID()
	if id1 == "" {
		t.Errorf("Test case 1 failed: GenerateID() returned an empty string")
	}

	// Test case 2: Generate multiple IDs and check if they are unique
	id2 := GenerateID()
	if id2 == "" {
		t.Errorf("Test case 2 failed: GenerateID() returned an empty string")
	}
	if id1 == id2 {
		t.Errorf("Test case 2 failed: Generated IDs are not unique")
	}

	// Test case 3: Generate IDs at different times and check if they are unique
	time.Sleep(1 * time.Second)
	id3 := GenerateID()
	if id3 == "" {
		t.Errorf("Test case 3 failed: GenerateID() returned an empty string")
	}
	if id1 == id3 || id2 == id3 {
		t.Errorf("Test case 3 failed: Generated IDs are not unique")
	}
}

func TestValidateID(t *testing.T) {
	// 测试ID为空的情况
	id1 := ""
	err1 := ValidateID(id1)
	if err1 == nil {
		t.Errorf("Expected error for empty id, but got nil")
	}

	// 测试ID长度不为标准长度的情况
	id2 := "1234567890"
	err2 := ValidateID(id2)
	if err2 == nil {
		t.Errorf("Expected error for id length not equal to standard length, but got nil")
	}

	// 测试ID长度为标准长度的情况
	id3 := "01234567890123456789012345678901234567890123456789012345"
	err3 := ValidateID(id3)
	if err3 != nil {
		t.Errorf("Expected no error for valid id, but got %v", err3)
	}
}

func TestValidateUTC(t *testing.T) {
	// 测试时间格式正确的情况
	timeStr1 := "2022-01-01T00:00:00Z"
	err1 := ValidateUTC(timeStr1)
	if err1 != nil {
		t.Errorf("Expected no error for valid UTC time, but got %v", err1)
	}

	// 测试时间格式错误的情况
	timeStr2 := "2022-01-01 00:00:00"
	err2 := ValidateUTC(timeStr2)
	if err2 == nil {
		t.Errorf("Expected error for invalid UTC time, but got nil")
	}
}

func TestValidateStep(t *testing.T) {
	tests := []struct {
		startUTC string
		endUTC   string
		stepStr  string
		expected error
	}{
		{
			startUTC: "2022-01-01T00:00:00Z",
			endUTC:   "2022-01-01T00:00:00Z",
			stepStr:  "1",
			expected: errors.New("end time is less than start time plus step"),
		},
		{
			startUTC: "2022-01-01T00:00:00Z",
			endUTC:   "2022-01-01T00:00:01Z",
			stepStr:  "1",
			expected: nil,
		},
		{
			startUTC: "2022-01-01T00:00:00Z",
			endUTC:   "2022-01-01T00:00:01Z",
			stepStr:  "2",
			expected: errors.New("end time is less than start time plus step"),
		},
		{
			startUTC: "2022-01-01T00:00:00Z",
			endUTC:   "2022-01-01T00:00:00Z",
			stepStr:  "0",
			expected: errors.New("invalid step duration"),
		},
		{
			startUTC: "2022-01-01T00:00:00Z",
			endUTC:   "2022-01-01T00:00:00Z",
			stepStr:  "-1",
			expected: errors.New("invalid step duration"),
		},
	}
	for _, test := range tests {
		err := ValidateStep(test.startUTC, test.endUTC, test.stepStr)
		assertError(t, err, test.expected)
	}
}

const (
	hostValid             = "http://prometheus-k8s.monitoring.svc.cluster.local:9090"
	hostInvalidScheme     = "https://example.com:8080"
	hostMissingPort       = "http://example.com"
	hostTrailingSlash     = "http://example.com:8080/"
	hostWithPath          = "http://example.com:8080/path"
	hostLoopbackV4        = "http://127.0.0.1:8080"
	hostLoopbackV6        = "http://[::1]:8080"
	hostLoopbackLocal     = "http://localhost:8080"
	errInvalidHost        = "invalid host format"
	errLoopbackNotAllow   = "loopback address is not allowed"
	caseNameValid         = "valid"
	caseNameInvalidScheme = "invalid-scheme"
	caseNameMissingPort   = "missing-port"
	caseNameTrailingSlash = "trailing-slash"
	caseNameWithPath      = "with-path"
	caseNameLoopbackV4    = "loopback-v4"
	caseNameLoopbackV6    = "loopback-v6"
	caseNameLoopbackLocal = "loopback-localhost"
)

func TestValidateHost(t *testing.T) {
	cases := []struct {
		name    string
		uri     string
		wantErr string
	}{
		{caseNameValid, hostValid, ""},
		{caseNameInvalidScheme, hostInvalidScheme, errInvalidHost},
		{caseNameMissingPort, hostMissingPort, errInvalidHost},
		{caseNameTrailingSlash, hostTrailingSlash, errInvalidHost},
		{caseNameWithPath, hostWithPath, errInvalidHost},
		{caseNameLoopbackV4, hostLoopbackV4, errLoopbackNotAllow},
		{caseNameLoopbackV6, hostLoopbackV6, errLoopbackNotAllow},
		{caseNameLoopbackLocal, hostLoopbackLocal, errLoopbackNotAllow},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateHost(tc.uri)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				return
			}
			if err == nil || err.Error() != tc.wantErr {
				t.Fatalf("expected error %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidateDisplayName(t *testing.T) {
	tests := []struct {
		displayName string
		maxLength   int
		expected    error
	}{
		{
			displayName: "ValidName",
			maxLength:   10,
			expected:    nil,
		},
		{
			displayName: "",
			maxLength:   10,
			expected:    errors.New("displayName is empty"),
		},
		{
			displayName: "ThisNameIsTooLong",
			maxLength:   10,
			expected:    errors.New("displayName is too long"),
		},
	}

	for _, test := range tests {
		err := ValidateDisplayName(test.displayName, test.maxLength)
		assertError(t, err, test.expected)
	}
}

func assertError(t *testing.T, err, expected error) {
	t.Helper()
	if expected == nil && err != nil {
		t.Errorf("Expected no error, but got %v", err)
	}
	if errors.Is(err, nil) && expected != nil {
		t.Errorf("Expected error %v, but got %v", expected, err)
	}
	if err != nil && err.Error() != expected.Error() {
		t.Errorf("Expected error %v, but got %v", expected, err)
	}
}
