/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package options

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewOptions(t *testing.T) {
	opts := NewOptions()
	assert.NotNil(t, opts)
	assert.False(t, opts.DryRun)
	assert.False(t, opts.Healthz)
	assert.Equal(t, 443, opts.Port)
}

func TestOpts_Struct(t *testing.T) {
	opts := &Opts{
		DryRun:         true,
		Healthz:        true,
		Port:           8443,
		CertFile:       "/cert/cert.pem",
		KeyFile:        "/cert/key.pem",
		SidecarCfgFile: "/cfg/sidecar.yaml",
		SvcNamespace:   "test-ns",
		SvcName:        "test-svc",
	}
	assert.True(t, opts.DryRun)
	assert.True(t, opts.Healthz)
	assert.Equal(t, 8443, opts.Port)
	assert.Equal(t, "/cert/cert.pem", opts.CertFile)
	assert.Equal(t, "/cert/key.pem", opts.KeyFile)
	assert.Equal(t, "/cfg/sidecar.yaml", opts.SidecarCfgFile)
	assert.Equal(t, "test-ns", opts.SvcNamespace)
	assert.Equal(t, "test-svc", opts.SvcName)
}

func TestFlags(t *testing.T) {
	opts := NewOptions()
	Flags(opts)
	assert.Equal(t, 443, opts.Port)
	assert.Equal(t, "/etc/webhook/certs/server.pem", opts.CertFile)
	assert.Equal(t, "/etc/webhook/certs/key.pem", opts.KeyFile)
	assert.Equal(t, "volcano-system", opts.SvcNamespace)
	assert.Equal(t, "numa-affinity-webhook-svc", opts.SvcName)
	assert.False(t, opts.DryRun)
	assert.False(t, opts.Healthz)
}
