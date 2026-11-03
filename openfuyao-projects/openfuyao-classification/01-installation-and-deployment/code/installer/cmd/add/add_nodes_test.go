/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package add

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewAddNodesCmd(t *testing.T) {
	cmd := NewAddNodesCmd()
	assert.Equal(t, "nodes", cmd.Use, "expected command name to be 'nodes'")
	assert.Equal(t, "Add nodes to the cluster based on the configuration file",
		cmd.Short, "expected command short description to be 'Add nodes to the cluster based on the configuration file'")
}
