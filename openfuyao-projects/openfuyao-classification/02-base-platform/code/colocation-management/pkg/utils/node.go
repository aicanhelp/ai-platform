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

// Package utils
package utils

import (
	"k8s.io/api/core/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

// IsOversubscriptionNode 判断节点是否为超卖节点
func IsOversubscriptionNode(node *v1.Node) bool {
	if node == nil {
		return false
	}
	oversubNode := false
	if value, ok := node.Labels[common.ColocationNodeLabel]; ok && value == "true" {
		if value, ok := node.Labels[common.OversubNodeLabel]; ok && value == "true" {
			oversubNode = true
		}
	}
	return oversubNode
}

// IsColocationNode 判断节点是否为混部节点
func IsColocationNode(node *v1.Node) bool {
	if node == nil {
		return false
	}
	colocationNode := false
	if value, ok := node.Labels[common.ColocationNodeLabel]; ok && value == "true" {
		colocationNode = true
	}
	return colocationNode
}
