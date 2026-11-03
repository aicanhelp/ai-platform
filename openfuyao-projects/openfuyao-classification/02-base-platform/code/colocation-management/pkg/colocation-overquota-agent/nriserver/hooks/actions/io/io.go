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

// Package io
package io

import (
	"k8s.io/klog/v2"
)

const (
	hookName = "io"
)

var configPath string

// IoRunner is the runner of io hook
type IoRunner struct {
}

var runner *IoRunner

// NewIoRunner returns a new IoRunner
func NewIoRunner() *IoRunner {
	if runner != nil {
		return runner
	}
	runner = &IoRunner{}
	return runner
}

// RegisterFunc cgroup中io的配置
func (r *IoRunner) RegisterFunc() {
	klog.Infof("register hook %v", hookName)
}
