/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package net

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestValidIP(t *testing.T) {
	ip127 := "127.0.0.1"
	assert.True(t, ValidIP(ip127))
	assert.False(t, ValidIP(""))
	assert.False(t, ValidIP("not-an-ip"))
}

func TestGetExternalIP(t *testing.T) {
	ip, err := GetExternalIP()
	assert.NotEmpty(t, ip)
	assert.Nil(t, err)
	assert.True(t, ValidIP(ip))
}

func TestSameNetworkSegment(t *testing.T) {
	assert.True(t, SameNetworkSegment("127.0.0.1", "127.0.0.2"))
}

func TestIsDNS1123Subdomain(t *testing.T) {
	repStr := "a"
	repNum := 253
	validDomains := []string{
		"example",
		"example.com",
		"my-service.k8s.local",
		"a.b.c.d",
		strings.Repeat(repStr, repNum),
	}
	for _, domain := range validDomains {
		err := IsDNS1123Subdomain(domain)
		assert.NoError(t, err, "should be valid: %s", domain)
	}

	invalidDomains := []string{
		"Example.com",            // 大写
		"-start.com",             // 以 - 开头
		"end-.com",               // 以 - 结尾
		"invalid_domain",         // 包含非法字符
		strings.Repeat("a", 254), // 长度超限
	}
	for _, domain := range invalidDomains {
		err := IsDNS1123Subdomain(domain)
		assert.Error(t, err, "should be invalid: %s", domain)
	}
}
