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

package operator

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/reconciler"
)

func TestDefaultStartOpts(t *testing.T) {
	opts := defaultStartOpts()
	assert.Equal(t, "/etc/ub-network-bandwidth/config.yaml", opts.ConfigPath)
	assert.Equal(t, "8089", opts.HealthPort)
}

func TestWithManager(t *testing.T) {
	manager := &mockManagerForOpts{}
	opts := defaultStartOpts()
	WithManager(manager)(opts)
	assert.Equal(t, manager, opts.Manager)
}

func TestWithConfigPath(t *testing.T) {
	opts := defaultStartOpts()
	WithConfigPath("/custom/path")(opts)
	assert.Equal(t, "/custom/path", opts.ConfigPath)
}

func TestWithHealthPort(t *testing.T) {
	opts := defaultStartOpts()
	WithHealthPort("9090")(opts)
	assert.Equal(t, "9090", opts.HealthPort)
}

type mockManagerForOpts struct{}

func (m *mockManagerForOpts) BandwidthSet(_ []config.BandwidthItem) error { return nil }
func (m *mockManagerForOpts) BandwidthReset() error                       { return nil }
func (m *mockManagerForOpts) BandwidthGet() ([]config.BandwidthItem, error) {
	return nil, nil
}

var _ reconciler.BandwidthManager = &mockManagerForOpts{}
