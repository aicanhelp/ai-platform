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

// Package uutil 用于提供通用工具
package uutil

import (
	"errors"
	"net"
	"net/url"
	"strconv"
	"time"
	"unicode/utf8"

	"k8s.io/apimachinery/pkg/util/uuid"

	"monitoring-dashboard-service/pkg/config/global"
)

// GenerateID 生成唯一ID
func GenerateID() string {
	timeNow := time.Now().UTC().Format(global.UTCTimePartOfID)
	id := timeNow + string(uuid.NewUUID())
	return id
}

// ValidateID 用于校验ID
func ValidateID(id string) error {
	if len(id) == 0 {
		return errors.New("id is empty")
	}
	if len(id) != global.StandardIDLength {
		return errors.New("id length is not valid")
	}
	return nil
}

// ValidateUTC 校验传入的字符串timeStr是否为UTC时间格式
func ValidateUTC(timeStr string) error {
	_, err := time.Parse(global.UTCTimeLayout, timeStr)
	if err != nil {
		return errors.New("time format is not valid")
	}
	return nil
}

// ValidateStep 校验步长是否合法，包含与start、end的逻辑检查
func ValidateStep(startUTC, endUTC, stepStr string) error {
	startTime, err := time.Parse(global.UTCTimeLayout, startUTC)
	if err != nil {
		return errors.New("invalid start time")
	}
	endTime, err := time.Parse(global.UTCTimeLayout, endUTC)
	if err != nil {
		return errors.New("invalid end time")
	}
	// 将步长字符串转换为int64类型的秒数
	step, err := time.ParseDuration(stepStr + "s")
	if err != nil || step.Seconds() <= 0 {
		return errors.New("invalid step duration")
	}
	// 计算时间差（以秒为单位）
	duration := endTime.Sub(startTime).Seconds()
	if duration < 0 {
		return errors.New("end time is less than start time")
	}

	// 检查时间差是否大于或等于步长
	if duration < step.Seconds() {
		return errors.New("end time is less than start time plus step")
	}
	// 如果通过所有检查，则返回nil表示没有错误
	return nil
}

// ValidateHost 校验 http1.1 host地址是否合法:http://ip:port
func ValidateHost(uri string) error {
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "http" {
		return errors.New("invalid host format")
	}
	host := u.Hostname()
	port := u.Port()
	// 不允许path/query/fragment非空
	if u.Path != "" || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("invalid host format")
	}
	if host == "" || port == "" {
		return errors.New("invalid host format")
	}
	if p, err := strconv.Atoi(port); err != nil ||
		p < global.MinPortNumber || p > global.MaxPortNumber {
		return errors.New("invalid host format")
	}
	// 排除本地环回地址（127.0.0.0/8、::1、localhost）
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return errors.New("loopback address is not allowed")
	}
	if host == "localhost" {
		return errors.New("loopback address is not allowed")
	}
	return nil
}

// ValidateDisplayName 校验displayName是否合法，不包括重名检查
func ValidateDisplayName(displayName string, maxLength int) error {
	if displayName == "" {
		return errors.New("displayName is empty")
	}
	utf8lens := utf8.RuneCountInString(displayName)
	if utf8lens > maxLength {
		return errors.New("displayName is too long")
	}
	return nil
}
