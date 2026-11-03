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

// Package types 定义K3s重启所需的配置参数
package types

// K3sRestartConfig 包含重启 K3s 所需的所有配置参数
type K3sRestartConfig struct {
	OnlineImage    string
	OtherRepo      string
	OtherRepoIp    string
	OtherSource    string
	HostIP         string
	ImageRepo      string
	ImageRepoPort  string
	KubernetesPort string
	ConsolePort    string
}
