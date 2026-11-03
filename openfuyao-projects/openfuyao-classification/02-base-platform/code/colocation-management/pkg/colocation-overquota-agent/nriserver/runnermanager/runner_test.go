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

package runnermanager

import (
	"testing"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/runnermanager/cgroup"
	"openfuyao.com/colocation-management/pkg/common"
)

func TestCgroupAbsPath(t *testing.T) {
	path := cgroup.CgroupAbsPath("kubepods.slice/kubepods-podxxxxxx.slice/cri-containerd-container1.scope",
		"", common.CPUSharesPath)
	expect := "/sys/fs/cgroup/cpu/kubepods.slice/kubepods-podxxxxxx.slice/cri-containerd-container1.scope/cpu.shares"
	if path != expect {
		t.Errorf("Expected  to be %v, got %v", expect, path)
	}
}
