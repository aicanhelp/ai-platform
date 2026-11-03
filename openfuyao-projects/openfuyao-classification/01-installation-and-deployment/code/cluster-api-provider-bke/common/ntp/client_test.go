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

package ntp

import (
	"os/exec"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/beevik/ntp"
	"github.com/stretchr/testify/assert"
)

const (
	testTimeout = 1
)

func TestDate_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(ntp.Time, func(host string) (time.Time, error) {
		return time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC), nil
	})

	var mockCmd *exec.Cmd
	patches.ApplyMethod(reflect.TypeOf(mockCmd), "CombinedOutput", func(_ *exec.Cmd) ([]byte, error) {
		return []byte("time set"), nil
	})

	err := Date("pool.ntp.org")
	assert.NoError(t, err)
}

func TestDate_NtpTimeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(ntp.Time, func(host string) (time.Time, error) {
		return time.Time{}, assert.AnError
	})

	err := Date("pool.ntp.org")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to query ntp server")
}

func TestDate_ExecError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(ntp.Time, func(host string) (time.Time, error) {
		return time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC), nil
	})

	var mockCmd *exec.Cmd
	patches.ApplyMethod(reflect.TypeOf(mockCmd), "CombinedOutput", func(_ *exec.Cmd) ([]byte, error) {
		return nil, assert.AnError
	})

	err := Date("pool.ntp.org")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to set system time")
}

func TestDate_DifferentServers(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	servers := []string{
		"pool.ntp.org",
		"time.google.com",
		"cn.pool.ntp.org",
		"0.pool.ntp.org",
	}

	for _, server := range servers {
		patches.ApplyFunc(ntp.Time, func(host string) (time.Time, error) {
			return time.Now(), nil
		})

		var mockCmd *exec.Cmd
		patches.ApplyMethod(reflect.TypeOf(mockCmd), "CombinedOutput", func(_ *exec.Cmd) ([]byte, error) {
			return []byte("ok"), nil
		})

		err := Date(server)
		assert.NoError(t, err)
	}
}
