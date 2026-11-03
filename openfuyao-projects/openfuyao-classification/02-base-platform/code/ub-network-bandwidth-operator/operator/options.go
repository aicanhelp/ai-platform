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
	"github.com/urfave/cli/v2"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/reconciler"
)

// StartOpts contains options used to start the bandwidth operator.
type StartOpts struct {
	Flags      []cli.Flag
	ConfigPath string
	HealthPort string
	Manager    reconciler.BandwidthManager
}

// Option customizes StartOpts.
type Option func(*StartOpts)

// WithManager sets the bandwidth manager implementation.
func WithManager(manager reconciler.BandwidthManager) Option {
	return func(o *StartOpts) {
		o.Manager = manager
	}
}

// WithConfigPath sets the mounted config file path.
func WithConfigPath(path string) Option {
	return func(o *StartOpts) {
		o.ConfigPath = path
	}
}

// WithHealthPort sets the health server port.
func WithHealthPort(port string) Option {
	return func(o *StartOpts) {
		o.HealthPort = port
	}
}

func defaultStartOpts() *StartOpts {
	return &StartOpts{
		ConfigPath: "/etc/ub-network-bandwidth/config.yaml",
		HealthPort: "8089",
	}
}
