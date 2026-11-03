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

// Package config parses and validates bandwidth configuration loaded from files.
package config

// BandwidthItem describes one priority and bandwidth pair from config.yaml.
type BandwidthItem struct {
	Priority  uint32 `yaml:"priority"`
	Bandwidth uint32 `yaml:"bandwidth"`
}

// BandwidthConfig is the root structure of config.yaml.
type BandwidthConfig struct {
	Bandwidths []BandwidthItem `yaml:"bandwidths"`
}
