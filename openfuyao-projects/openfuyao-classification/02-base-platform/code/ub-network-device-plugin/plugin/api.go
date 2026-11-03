/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"time"

	"github.com/urfave/cli/v2"

	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
)

type StartOpts struct {
	Flags         []cli.Flag
	KubeletSocket string
	SyncInterval  time.Duration
	HealthPort    string
}

type Interface interface {
	Devices() resource.Devices
	Start(*StartOpts) error
	Stop() error
}
