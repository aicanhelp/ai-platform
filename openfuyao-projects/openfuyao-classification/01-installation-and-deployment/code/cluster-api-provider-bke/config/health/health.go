/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/
package health

import _ "embed"

const (
	Namespace = "bke-config"
	Name      = "health-check-config"
	DataKey   = "config.yaml"
)

// DefaultConfig contains the default health check config.
//
//go:embed config.yaml
var DefaultConfig string

// DefaultManifest contains the default health check ConfigMap manifest.
//
//go:embed health-check-config.yaml
var DefaultManifest string
