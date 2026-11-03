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

package runtime

// ServerConfig 定义一个 http.server 结构
type DbConfig struct {
	// server bind address
	DbFilePath    string
	RemoveExistDb bool
}

// NewServerConfig create new server config
func NewDbConfig() *DbConfig {
	// create default server run options
	s := DbConfig{
		DbFilePath:    "/var/log/openfuyao-rayservice/openfuyao-ray.db",
		RemoveExistDb: false,
	}

	return &s
}
