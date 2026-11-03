/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package metadata

import "installer/pkg/fuyao/v1beta1/base"

// RegistryConfig is registry config
type RegistryConfig struct {
	APIVersion string        `yaml:"apiVersion"`
	Kind       string        `yaml:"kind"`
	Metadata   base.Metadata `yaml:"metadata"`
	Spec       Spec          `yaml:"spec"`
}

// Spec is 具体数据
type Spec struct {
	IP           string `yaml:"ip"`
	Domain       string `yaml:"domain"`
	CaPath       string `yaml:"caPath"`
	OriginDomain string `yaml:"originDomain"`
}
