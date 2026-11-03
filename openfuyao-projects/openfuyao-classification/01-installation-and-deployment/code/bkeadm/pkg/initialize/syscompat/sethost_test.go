/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package syscompat

import (
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

const (
	testNumericZero  = 0
	testNumericOne   = 1
	testNumericTwo   = 2
	testNumericThree = 3
)

const (
	testIPv4SegmentA = 192
	testIPv4SegmentB = 168
	testIPv4SegmentC = 1
	testIPv4SegmentD = 100
)

func TestSetHostsFunction(t *testing.T) {
	tests := []struct {
		name        string
		mockOpenErr error
		expectError bool
	}{
		{
			name:        "open file fails with permission denied",
			mockOpenErr: &os.PathError{Op: "open", Path: "/etc/hosts", Err: errors.New("permission denied")},
			expectError: true,
		},
		{
			name:        "open file fails with not found",
			mockOpenErr: &os.PathError{Op: "open", Path: "/etc/hosts", Err: errors.New("no such file or directory")},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(os.Open, func(filename string) (*os.File, error) {
				if filename == "/etc/hosts" {
					return nil, tt.mockOpenErr
				}
				return os.Open(filename)
			})

			ip := formatIPForTest()
			err := SetHosts(ip, "testhost")

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRemoveManagedBlock(t *testing.T) {
	lines := []string{
		"127.0.0.1 localhost",
		BKEManagedHostsBegin,
		"10.0.0.1 old-host",
		BKEManagedHostsEnd,
		"192.168.1.1 keep-host",
	}

	result := removeManagedBlock(lines, BKEManagedHostsBegin, BKEManagedHostsEnd)

	assert.Equal(t, []string{
		"127.0.0.1 localhost",
		"192.168.1.1 keep-host",
	}, result)
}

func TestRemoveManagedBlockFromContent(t *testing.T) {
	content := strings.Join([]string{
		"fs.file-max = 9000000",
		BKEManagedSysctlBegin,
		"net.ipv4.ip_forward = 1",
		BKEManagedSysctlEnd,
		"kernel.pid_max = 4194304",
	}, "\r\n")

	result := removeManagedBlockFromContent(content, BKEManagedSysctlBegin, BKEManagedSysctlEnd)

	assert.NotContains(t, result, BKEManagedSysctlBegin)
	assert.NotContains(t, result, "net.ipv4.ip_forward = 1")
	assert.Contains(t, result, "fs.file-max = 9000000")
	assert.Contains(t, result, "kernel.pid_max = 4194304")
}

func formatIPForTest() string {
	return formatIPAddress(testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)
}

func formatIPAddress(a, b, c, d int) string {
	return string(rune('0'+a/100%10)) + string(rune('0'+a/10%10)) + string(rune('0'+a%10)) + "." +
		string(rune('0'+b/100%10)) + string(rune('0'+b/10%10)) + string(rune('0'+b%10)) + "." +
		string(rune('0'+c/100%10)) + string(rune('0'+c/10%10)) + string(rune('0'+c%10)) + "." +
		string(rune('0'+d/100%10)) + string(rune('0'+d/10%10)) + string(rune('0'+d%10))
}
