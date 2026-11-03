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

// Package config
package config

// Config 用于配置NRI插件
type Config struct {
	Network      string
	Addr         string
	HostEndpoint string
	SocketPath   string
	CgroupDriver string
	CgroupRoot   string
	NumaAffinity NumaAffinityConfig `json:"numaAffinity"`
}

// NumaAffinityConfig 用于配置NUMA亲和性功能
type NumaAffinityConfig struct {
	// 是否启用NUMA亲和性功能
	Enabled bool `json:"enabled"`
}
