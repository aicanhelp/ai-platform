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
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/test/mock"
)

func TestCreateCLuster(t *testing.T) {
	nodes := mock.GetAllNodes()

	patchOsName := gomonkey.NewPatches()
	defer patchOsName.Reset()
	patchOsName.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "checkAllNodesOsName", func(_ *FlightCheck) error {
			return nil
		},
	)

	patchNodeToolCheck := gomonkey.NewPatches()
	defer patchNodeToolCheck.Reset()
	patchNodeToolCheck.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "allNodeToolCheck", func(_ *FlightCheck) error {
			return nil
		},
	)

	patchWorkerMemory := gomonkey.NewPatches()
	defer patchWorkerMemory.Reset()
	patchWorkerMemory.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "workerNodesMemoryCheck", func(_ *FlightCheck, _ int) error {
			return nil
		},
	)

	patchWorkerNodesCup := gomonkey.NewPatches()
	defer patchWorkerNodesCup.Reset()
	patchWorkerNodesCup.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "workerNodesCupCheck", func(_ *FlightCheck, _ int) error {
			return nil
		},
	)

	patchMasterMemory := gomonkey.NewPatches()
	defer patchMasterMemory.Reset()
	patchMasterMemory.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "masterNodesMemoryCheck", func(_ *FlightCheck, _ int) error {
			return nil
		},
	)

	patchMasterNodesCup := gomonkey.NewPatches()
	defer patchMasterNodesCup.Reset()
	patchMasterNodesCup.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "masterNodesCupCheck", func(_ *FlightCheck, _ int) error {
			return nil
		},
	)

	patchPing := gomonkey.NewPatches()
	defer patchPing.Reset()
	patchPing.ApplyPrivateMethod(
		reflect.TypeOf(&FlightCheck{}), "allNodePingCheck", func(_ *FlightCheck, _ int) error {
			return nil
		},
	)

	// 多节点测试
	flightCheck := NewCreateClusterFlightCheck(nodes, true)
	err := flightCheck.StartCheck()
	assert.NoError(t, err)
}
