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

package operate

import (
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/clients"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/test/mock"
)

func TestGetMemorySize(t *testing.T) {
	// 构建节点
	nodes := mock.GetAllNodes()

	patchExecCmd := gomonkey.NewPatches()
	patchExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "ExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "16000000", nil
		},
	)
	defer patchExecCmd.Reset()

	nodesOperate := NewNodesOperate(nodes)
	// 测试所有节点获取内存大小
	nodesOperate.AllNodeExecOperate(nodesOperate.GetMemorySize)
	memorySize, ok := nodesOperate.Cache.Get(nodes.Masters[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, memorySize, float64(16000000))
	nodesOperate.AllNodeExecOperate(nodesOperate.GetMemorySize)
	memorySize, ok = nodesOperate.Cache.Get(nodes.Registry[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, memorySize, float64(16000000))
	nodesOperate.AllNodeExecOperate(nodesOperate.GetMemorySize)
	memorySize, ok = nodesOperate.Cache.Get(nodes.Workers[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, memorySize, float64(16000000))
	nodesOperate.Cache.Clear()

	// 测试master节点获取内存大小
	nodesOperate.MasterNodeExecOperate(nodesOperate.GetMemorySize)
	memorySize, ok = nodesOperate.Cache.Get(nodes.Masters[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, memorySize, float64(16000000))
	nodesOperate.Cache.Clear()

	// 测试master节点获取内存大小
	nodesOperate.WorkerNodeExecOperate(nodesOperate.GetMemorySize)
	memorySize, ok = nodesOperate.Cache.Get(nodes.Workers[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, memorySize, float64(16000000))
	nodesOperate.Cache.Clear()

	// 测试master节点获取内存大小
	nodesOperate.RegistryNodeExecOperate(nodesOperate.GetMemorySize)
	memorySize, ok = nodesOperate.Cache.Get(nodes.Registry[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, memorySize, float64(16000000))
	nodesOperate.Cache.Clear()
}

func TestGetCPUCount(t *testing.T) {
	// 构建节点
	nodes := mock.GetOneMasterNode()

	// 返回的cpu数量符合预期
	patchSudoExecCmd := gomonkey.NewPatches()
	patchSudoExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "4", nil
		},
	)

	nodesOperate := NewNodesOperate(nodes)
	// 测试master节点获取CPU数量
	nodesOperate.MasterNodeExecOperate(nodesOperate.GetCPUCount)
	cpuCount, ok := nodesOperate.Cache.Get(nodes.Masters[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, cpuCount, 4)
	nodesOperate.Cache.Clear()
}

func TestInstallTools(t *testing.T) {
	// 构建节点
	nodes := mock.GetOneMasterNode()

	// 返回的数据符合预期
	patchSudoExecCmd := gomonkey.NewPatches()
	patchSudoExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "", nil
		},
	)

	// 返回的数据符合预期
	patchExecCmd := gomonkey.NewPatches()
	patchExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "ExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "", nil
		},
	)

	nodesOperate := NewNodesOperate(nodes)
	nodesOperate.Tools = []string{"tar"}

	// 测试master节点
	nodesOperate.MasterNodeExecOperate(nodesOperate.InstallTools)
	cpuCount, ok := nodesOperate.Cache.Get(nodes.Masters[0].IP)
	assert.Equal(t, ok, true)
	installStatus := make(map[string]bool)
	installStatus["tar"] = true
	assert.Equal(t, cpuCount, installStatus)
	nodesOperate.Cache.Clear()
}

func TestGetOSName(t *testing.T) {
	// 构建节点
	nodes := mock.GetOneMasterNode()

	patchOsInfo := gomonkey.NewPatches()
	defer patchOsInfo.Reset()
	patchOsInfo.ApplyPrivateMethod(
		reflect.TypeOf(&fuyaov1beta1.Node{}), "OsInfo", func(_ *fuyaov1beta1.Node) (string, string, string) {
			return "openEuler", "", ""
		},
	)

	nodesOperate := NewNodesOperate(nodes)
	// 测试master节点获取OS名称
	nodesOperate.MasterNodeExecOperate(nodesOperate.GetOSName)
	osName, ok := nodesOperate.Cache.Get(nodes.Masters[0].IP)
	assert.Equal(t, ok, true)
	assert.Equal(t, osName, "openEuler")
}
