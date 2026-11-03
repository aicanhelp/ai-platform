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
	"testing"
)

// TestValidateStepInvalidTime 补充ValidateStep非法时间分支
func TestValidateStepInvalidTime(t *testing.T) {
	cases := []struct {
		name      string
		startUTC  string
		endUTC    string
		stepStr   string
		wantError string
	}{
		{"非法开始时间", "2022-01-01 00:00:00", "2022-01-01T00:00:01Z", "1", "invalid start time"},
		{"非法结束时间", "2022-01-01T00:00:00Z", "not-a-time", "1", "invalid end time"},
		{"结束时间早于开始时间", "2022-01-01T00:00:02Z", "2022-01-01T00:00:01Z", "1",
			"end time is less than start time"},
		{"非法步长格式", "2022-01-01T00:00:00Z", "2022-01-01T00:00:10Z", "abc",
			"invalid step duration"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateStep(tc.startUTC, tc.endUTC, tc.stepStr)
			if err == nil || err.Error() != tc.wantError {
				t.Errorf("expected error %q, got %v", tc.wantError, err)
			}
		})
	}
}

// TestValidateHostPortRange 补充ValidateHost端口范围分支
func TestValidateHostPortRange(t *testing.T) {
	cases := []struct {
		name    string
		uri     string
		wantErr bool
	}{
		{"端口为0", "http://example.com:0", true},
		{"端口超过65535", "http://example.com:99999", true},
		{"端口非法字符", "http://example.com:8o80", true},
		{"无host", "http://:8080", true},
		{"合法端口", "http://example.com:8080", false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateHost(tc.uri)
			if tc.wantErr && err == nil {
				t.Errorf("expected error for %s, got nil", tc.uri)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("expected no error for %s, got %v", tc.uri, err)
			}
		})
	}
}
