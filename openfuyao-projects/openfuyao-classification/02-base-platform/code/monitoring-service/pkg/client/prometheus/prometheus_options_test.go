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

package prometheus

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

func TestNewPrometheusOptionsDefaultEnv(t *testing.T) {
	// No env vars set
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})
	defer patches.Reset()

	opts := NewPrometheusOptions()
	assert.NotNil(t, opts)
	assert.Empty(t, opts.Endpoint)
	assert.False(t, opts.InsecureSkipVerify)
}

func TestNewPrometheusOptionsWithEndpoint(t *testing.T) {
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "PROMETHEUS_HOST" {
			return "http://prometheus:9090"
		}
		if key == "PROMETHEUS_INSECURE_SKIP_VERIFY" {
			return ""
		}
		return ""
	})
	defer patches.Reset()

	opts := NewPrometheusOptions()
	assert.NotNil(t, opts)
	assert.Equal(t, "http://prometheus:9090", opts.Endpoint)
	assert.False(t, opts.InsecureSkipVerify)
}

func TestNewPrometheusOptionsInsecureSkipVerifyTrue(t *testing.T) {
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "PROMETHEUS_HOST" {
			return "https://prometheus:9090"
		}
		if key == "PROMETHEUS_INSECURE_SKIP_VERIFY" {
			return "true"
		}
		return ""
	})
	defer patches.Reset()

	opts := NewPrometheusOptions()
	assert.NotNil(t, opts)
	assert.Equal(t, "https://prometheus:9090", opts.Endpoint)
	assert.True(t, opts.InsecureSkipVerify)
}

func TestNewPrometheusOptionsInsecureSkipVerifyFalse(t *testing.T) {
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "PROMETHEUS_HOST" {
			return "https://prometheus:9090"
		}
		if key == "PROMETHEUS_INSECURE_SKIP_VERIFY" {
			return "false"
		}
		return ""
	})
	defer patches.Reset()

	opts := NewPrometheusOptions()
	assert.NotNil(t, opts)
	assert.Equal(t, "https://prometheus:9090", opts.Endpoint)
	assert.False(t, opts.InsecureSkipVerify)
}
