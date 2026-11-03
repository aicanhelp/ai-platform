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

// Package model 定义了monitor模块涉及的的 DO VO BO
package model

import (
	"errors"

	"monitoring-dashboard-service/pkg/utils/uutil"
)

// Indicator 指标结构体，用于在预览阶段返回指标及包含的实例信息
type Indicator struct {
	DisplayName string     `json:"displayName"`
	Instances   []Instance `json:"instances"`
}

// Instance 指标结构体
type Instance struct {
	Metrics map[string]interface{} `json:"metric"`
	Values  []interface{}          `json:"value"` // [{float64,string},...]
}

// InstanceQuery 用于接收Instance查询请求的参数
type InstanceQuery struct {
	DataSource string
	Start      string
	End        string
	Step       string
	Instances  []string // JSON字符串array
}

// NewInstanceQuery InstanceQuery 的构造函数
func NewInstanceQuery(dataSource, start, end, step string, instances []string) *InstanceQuery {
	return &InstanceQuery{
		DataSource: dataSource,
		Start:      start,
		End:        end,
		Step:       step,
		Instances:  instances,
	}
}

// Validate InstanceQuery 的校验函数
func (iq *InstanceQuery) Validate() error {
	errHost := uutil.ValidateHost(iq.DataSource)
	if errHost != nil {
		return errHost
	}
	// validate time
	errUTC := uutil.ValidateUTC(iq.Start)
	if errUTC != nil {
		return errUTC
	}
	errUTC = uutil.ValidateUTC(iq.End)
	if errUTC != nil {
		return errUTC
	}
	errStep := uutil.ValidateStep(iq.Start, iq.End, iq.Step)
	if errStep != nil {
		return errStep
	}
	if len(iq.Instances) == 0 {
		return errors.New("instances List is empty")
	}
	for _, instance := range iq.Instances {
		if instance == "" {
			return errors.New("instance is empty string")
		}
	}
	return nil
}
