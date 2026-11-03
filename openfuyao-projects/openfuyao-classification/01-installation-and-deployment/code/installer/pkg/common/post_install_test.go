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

package common

import (
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"

	"installer/pkg/fuyao/v1beta1/cluster"
)

// TestPostInstallCheckFastRun 确保 PostInstallCheck 在秒级内跑完
func TestPostInstallCheckFastRun(t *testing.T) {
	// 1. 构造测试节点
	node := &cluster.Node{
		Hostname: "hostname",
		User:     "fuyao",
		Password: []byte("password"),
	}

	// 2. 新建补丁容器，测试结束时自动还原
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 3. 用极短超时时间创建一个 statusCheck 实例，仅用于获取真实类型
	tmp := NewClusterStatusCheck(node, 1*time.Millisecond)
	typ := reflect.TypeOf(tmp)

	// 4. 打补丁：任何时候调用 PodStatusCheck 都立刻返回 true
	patches.ApplyMethod(typ, "PodStatusCheck",
		func(_ *ClusterStatusCheck) bool {
			return true
		},
	)

	// 5. 调用被测函数，因 PodStatusCheck 已被 stub，整个流程瞬时返回
	PostInstallCheck(node)

	// 6. 如果代码卡死或 panic，测试会超时或失败；否则视为通过
}
