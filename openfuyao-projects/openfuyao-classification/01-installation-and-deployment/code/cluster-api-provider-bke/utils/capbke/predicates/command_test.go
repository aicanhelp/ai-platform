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

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
)

func TestCommandUpdateCompleted_Update(t *testing.T) {
	pred := CommandUpdateCompleted()
	oldCmd := &agentv1beta1.Command{Status: map[string]*agentv1beta1.CommandStatus{"node1": {}}}
	newCmd := &agentv1beta1.Command{Status: map[string]*agentv1beta1.CommandStatus{"node1": {}, "node2": {}}}
	e := event.UpdateEvent{ObjectOld: oldCmd, ObjectNew: newCmd}
	result := pred.Update(e)
	assert.True(t, result)
}

func TestCommandUpdateCompleted_Create(t *testing.T) {
	pred := CommandUpdateCompleted()
	e := event.CreateEvent{}
	result := pred.Create(e)
	assert.False(t, result)
}
