/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package sntp

import (
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/btfak/sntp/sntp"
	"github.com/stretchr/testify/assert"
)

const (
	testTimeout = 1
)

func TestClient_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	expectedTime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	patches.ApplyFunc(sntp.Client, func(host string) (time.Time, error) {
		return expectedTime, nil
	})

	result, err := Client("pool.ntp.org")
	assert.NoError(t, err)
	assert.Equal(t, expectedTime, result)
}

func TestClient_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(sntp.Client, func(host string) (time.Time, error) {
		return time.Time{}, assert.AnError
	})

	result, err := Client("pool.ntp.org")
	assert.Error(t, err)
	assert.True(t, result.IsZero())
}

func TestClient_WithDifferentHosts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testCases := []string{
		"pool.ntp.org",
		"time.google.com",
		"0.pool.ntp.org",
		"127.0.0.1",
	}

	for _, host := range testCases {
		patches.ApplyFunc(sntp.Client, func(h string) (time.Time, error) {
			return time.Now(), nil
		})

		result, err := Client(host)
		assert.NoError(t, err)
		assert.False(t, result.IsZero())
	}
}

func TestClient_EmptyHost(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(sntp.Client, func(host string) (time.Time, error) {
		return time.Time{}, assert.AnError
	})

	result, err := Client("")
	assert.Error(t, err)
	assert.True(t, result.IsZero())
}
