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

func TestNewAddCmd(t *testing.T) {
	cmd := NewAddCmd()
	assert.Equal(t, "add", cmd.Use, "expected command name to be 'add'")
	assert.Equal(t, "Add command", cmd.Short, "expected command short description to be 'Add command'")
}

func TestNewAddOptions(t *testing.T) {
	o := NewAddOptions()
	assert.NotNil(t, o, "expected NewAddOptions to return a non-nil object")
}
