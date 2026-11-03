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

// Package util defines data structure and provide util function for xpu scheduler plugin implementation
package util

import (
	"sync"

	"k8s.io/api/core/v1"
)

// XPUTask for xpu task need.
type XPUTask struct {
	Name      string
	NameSpace string
	*TaskResource
	Selector   map[string]string
	Annotation map[string]string
	Label      map[string]string
	NodeName   string
	PodStatus  v1.PodPhase
	ScoreMap   map[string]float64
	sync.Mutex
}

// TaskResource for xpu task pod.
type TaskResource struct {
	ReqXPUName          string
	ReqXPUNum           int
	ReqXPUType          string
	ReqXPUCores         int
	ReqXPUMem           int
	ReqXPUMemPercentage int
	ReqCpuLevel         string
	ReqPolicy           string
	IsPrivileged        bool
	ReqXPUDieID         string
	ReqUseNPUType       string
}

// ContainerResource for xpu container.
type ContainerResource TaskResource
