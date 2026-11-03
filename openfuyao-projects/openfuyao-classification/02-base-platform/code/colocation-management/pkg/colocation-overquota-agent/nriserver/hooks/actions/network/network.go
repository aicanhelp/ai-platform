/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

// Package network
package network

import (
	"k8s.io/klog/v2"
)

const (
	hookName = "network"
)

var configPath string

// NetworkRunner is the runner of network hook
type NetworkRunner struct {
}

var runner *NetworkRunner

// NewNetworkRunner is the constructor of NetworkRunner
func NewNetworkRunner() *NetworkRunner {
	if runner != nil {
		return runner
	}
	runner = &NetworkRunner{}
	return runner
}

// RegisterFunc cgroup中io的配置
func (r *NetworkRunner) RegisterFunc() {
	klog.Infof("register hook %v", hookName)
}
