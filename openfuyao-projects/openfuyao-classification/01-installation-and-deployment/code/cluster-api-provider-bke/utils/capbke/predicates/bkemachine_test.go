/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package predicates

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/event"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestBKEMachineConditionUpdate_Update(t *testing.T) {
	pred := BKEMachineConditionUpdate()
	machine := &bkev1beta1.BKEMachine{}
	e := event.UpdateEvent{ObjectNew: machine, ObjectOld: machine}
	result := pred.Update(e)
	assert.False(t, result)
}

func TestBKEMachineConditionUpdate_Create(t *testing.T) {
	pred := BKEMachineConditionUpdate()
	e := event.CreateEvent{}
	result := pred.Create(e)
	assert.False(t, result)
}
