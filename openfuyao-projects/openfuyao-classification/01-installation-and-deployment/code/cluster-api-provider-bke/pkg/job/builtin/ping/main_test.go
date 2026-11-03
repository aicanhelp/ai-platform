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

package ping

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPingName(t *testing.T) {
	plugin := &Ping{}
	assert.Equal(t, name, plugin.Name())
}

func TestPingParam(t *testing.T) {
	plugin := &Ping{}
	params := plugin.Param()
	assert.Nil(t, params)
}

func TestNewPing(t *testing.T) {
	plugin := New()
	assert.NotNil(t, plugin)
	assert.Equal(t, name, plugin.Name())
}

func TestPingExecute(t *testing.T) {
	plugin := &Ping{}
	commands := []string{name}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Len(t, result, 2)
	assert.Equal(t, "pong", result[0])
}

func TestPingExecuteWithArgs(t *testing.T) {
	plugin := &Ping{}
	commands := []string{name, "extraArg"}
	result, err := plugin.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.Len(t, result, 2)
}
