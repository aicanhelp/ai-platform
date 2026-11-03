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

package precheck

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/clients"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/operate"
	"installer/test/mock"
)

func TestNodeCupCheck(t *testing.T) {
	// 构建节点
	nodes := mock.GetAllNodes()
	// 设定CPU标准
	cpuStandard := 4

	// 返回的cpu数量符合预期
	patchSudoExecCmd := gomonkey.NewPatches()
	patchSudoExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "4", nil
		},
	)

	flightCheck := NewFlightCheck(nodes)

	// 所有节点符合预期
	err := flightCheck.allNodeCupCheck(cpuStandard)
	assert.NoError(t, err)

	// master节点符合预期
	err = flightCheck.masterNodesCupCheck(cpuStandard)
	assert.NoError(t, err)

	// worker节点符合预期
	err = flightCheck.workerNodesCupCheck(cpuStandard)
	assert.NoError(t, err)

	// registry节点符合预期
	err = flightCheck.registryNodesCupCheck(cpuStandard)
	assert.NoError(t, err)

	// 返回的cpu数量不符合预期
	patchSudoExecCmd.Reset()
	defer patchSudoExecCmd.Reset()
	patchSudoExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "2", nil
		},
	)

	// 不符合预期
	err = flightCheck.allNodeCupCheck(cpuStandard)
	assert.Error(t, err)

	// master节点不符合预期
	err = flightCheck.masterNodesCupCheck(cpuStandard)
	assert.Error(t, err)

	// worker节点不符合预期
	err = flightCheck.workerNodesCupCheck(cpuStandard)
	assert.Error(t, err)

	// registry节点不符合预期
	err = flightCheck.registryNodesCupCheck(cpuStandard)
	assert.Error(t, err)
}

func TestNodeMemCheck(t *testing.T) {
	// 构建节点
	nodes := mock.GetAllNodes()
	// 设定内存标准
	memStandard := 16

	// 返回的内存数量符合预期
	patchExecCmd := gomonkey.NewPatches()
	patchExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "ExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "16000000", nil
		},
	)

	flightCheck := NewFlightCheck(nodes)

	// 所有节点符合预期
	err := flightCheck.allNodeMemoryCheck(memStandard)
	assert.NoError(t, err)

	// master节点符合预期
	err = flightCheck.masterNodesMemoryCheck(memStandard)
	assert.NoError(t, err)

	// worker节点符合预期
	err = flightCheck.workerNodesMemoryCheck(memStandard)
	assert.NoError(t, err)

	// registry节点符合预期
	err = flightCheck.registryNodesMemoryCheck(memStandard)
	assert.NoError(t, err)

	// 返回的内存数量不符合预期
	patchExecCmd.Reset()
	defer patchExecCmd.Reset()
	patchExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "ExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "13000000", nil
		},
	)

	// 所有节点不符合预期
	err = flightCheck.allNodeMemoryCheck(memStandard)
	assert.Error(t, err)

	// master节点不符合预期
	err = flightCheck.masterNodesMemoryCheck(memStandard)
	fmt.Println(err)
	assert.Error(t, err)

	// worker节点不符合预期
	err = flightCheck.workerNodesMemoryCheck(memStandard)
	assert.Error(t, err)

	// registry节点不符合预期
	err = flightCheck.registryNodesMemoryCheck(memStandard)
	assert.Error(t, err)
}

func TestNodeToolCheck(t *testing.T) {
	// 构建节点
	nodes := mock.GetAllNodes()

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

	flightCheck := NewFlightCheck(nodes)
	flightCheck.NodesOperate.Tools = []string{"tar", "openssl"}

	// 所有节点符合预期
	err := flightCheck.allNodeToolCheck()
	assert.NoError(t, err)

	// master节点符合预期
	err = flightCheck.masterNodesToolCheck()
	assert.NoError(t, err)

	// worker节点符合预期
	err = flightCheck.workerNodesToolCheck()
	assert.NoError(t, err)

	// registry节点符合预期
	err = flightCheck.registryNodesToolCheck()
	assert.NoError(t, err)

	// 返回的数据不符合预期
	patchSudoExecCmd.Reset()
	defer patchSudoExecCmd.Reset()
	patchSudoExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "", fmt.Errorf("error")
		},
	)

	patchExecCmd.Reset()
	defer patchExecCmd.Reset()
	patchExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "ExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "", fmt.Errorf("error")
		},
	)

	// 所有节点不符合预期
	err = flightCheck.allNodeToolCheck()
	assert.Error(t, err)

	// master节点不符合预期
	err = flightCheck.masterNodesToolCheck()
	assert.Error(t, err)

	// worker节点不符合预期
	err = flightCheck.workerNodesToolCheck()
	assert.Error(t, err)

	// registry节点不符合预期
	err = flightCheck.registryNodesToolCheck()
	assert.Error(t, err)
}

func TestNodePingCheck(t *testing.T) {
	// 构建节点
	nodes := mock.GetAllNodes()

	patchPing := gomonkey.NewPatches()
	patchPing.ApplyPrivateMethod(
		reflect.TypeOf(&operate.NodesOperate{}), "Ping", func(n *operate.NodesOperate, node *fuyaov1beta1.Node, wg *sync.WaitGroup) {
			defer wg.Done()
			defer n.RoutineMaximum.Done()

			n.Cache.Set(node.IP, true)
		},
	)
	// 所有节点ping通
	flightCheck := NewFlightCheck(nodes)
	err := flightCheck.allNodePingCheck()
	assert.NoError(t, err)

	patchPing.Reset()
	defer patchPing.Reset()
	patchPing.ApplyPrivateMethod(
		reflect.TypeOf(&operate.NodesOperate{}), "Ping", func(n *operate.NodesOperate, node *fuyaov1beta1.Node, wg *sync.WaitGroup) {
			defer wg.Done()
			defer n.RoutineMaximum.Done()

			n.Cache.Set(node.IP, false)
		},
	)
	// 所有节点ping不通
	err = flightCheck.allNodePingCheck()
	assert.Error(t, err)
}

func TestNodesOsNameCheck(t *testing.T) {
	// 构建节点
	nodes := mock.GetAllNodes()

	patchOsInfo := gomonkey.NewPatches()
	defer patchOsInfo.Reset()
	patchOsInfo.ApplyPrivateMethod(
		reflect.TypeOf(&fuyaov1beta1.Node{}), "OsInfo", func(_ *fuyaov1beta1.Node) (string, string, string) {
			return "openEuler", "", ""
		},
	)

	flightCheck := NewFlightCheck(nodes)

	err := flightCheck.checkAllNodesOsName()
	assert.NoError(t, err)
}
