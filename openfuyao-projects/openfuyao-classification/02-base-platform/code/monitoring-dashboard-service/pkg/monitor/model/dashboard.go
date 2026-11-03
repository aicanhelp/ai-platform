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
	"unicode/utf8"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/utils/uutil"
)

// DashboardCRD 用于和 k8s CR 对接的 DO
type DashboardCRD struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	Spec DashboardSpec `json:"spec"`
}

// DashboardSpec  DashboardCRD 的 spec 自定义部分
type DashboardSpec struct {
	DashboardID    string `json:"dashboardID"`
	DisplayName    string `json:"displayName"`
	Creator        string `json:"creator"`
	TotalWidgets   int    `json:"totalWidgets"`
	LastUpdateTime string `json:"lastUpdateTime"` // UTC 格式
}

// Dashboard 面向 Service 的完整的 BO
type Dashboard struct {
	DashboardID    string   `json:"dashboardID"`
	DisplayName    string   `json:"displayName"`
	Creator        string   `json:"creator"`
	TotalWidgets   int      `json:"totalWidgets"`
	LastUpdateTime string   `json:"lastUpdateTime"` // UTC 格式
	Widgets        []Widget `json:"widgets"`
}

// DashboardVO 用于接收前端请求的 VO
type DashboardVO struct {
	DashboardID string `json:"dashboardID"`
	DisplayName string `json:"displayName"`
	Creator     string `json:"creator"`
	Start       string `json:"start"`
	End         string `json:"end"`
	Step        string `json:"step"`
}

// ValidateCreate 对Create请求体进行校验，校验内容包括DisplayName和Creator
func (d *DashboardVO) ValidateCreate() error {
	errName := uutil.ValidateDisplayName(d.DisplayName, global.MaxNameLength)
	if errName != nil {
		return errName
	}
	if d.Creator == "" {
		return errors.New("creator is empty")
	}
	if utf8.RuneCountInString(d.Creator) > global.MaxNameLength {
		return errors.New("creator is too long")
	}
	return nil
}

// ValidatePatchUpdate 对Patch请求体进行校验，校验内容包括DashboardID和DisplayName
func (d *DashboardVO) ValidatePatchUpdate() error {
	errID := uutil.ValidateID(d.DashboardID)
	if errID != nil {
		return errID
	}
	errName := uutil.ValidateDisplayName(d.DisplayName, global.MaxNameLength)
	if errName != nil {
		return errName
	}
	return nil
}

// ValidatePostQuery 对Post请求体进行校验，校验内容包括DashboardID、Start、End和Step
func (d *DashboardVO) ValidatePostQuery() error {
	errID := uutil.ValidateID(d.DashboardID)
	if errID != nil {
		return errID
	}
	// validate time
	errUTC := uutil.ValidateUTC(d.Start)
	if errUTC != nil {
		return errUTC
	}
	errUTC = uutil.ValidateUTC(d.End)
	if errUTC != nil {
		return errUTC
	}
	errStep := uutil.ValidateStep(d.Start, d.End, d.Step)
	if errStep != nil {
		return errStep
	}
	return nil
}

// IsDisplayNameExists 检查DisplayName是否已存在
func (d *DashboardVO) IsDisplayNameExists(dashboards []*DashboardSpec) bool {
	for _, spec := range dashboards {
		if spec.DisplayName == d.DisplayName {
			return true
		}
	}
	return false
}
