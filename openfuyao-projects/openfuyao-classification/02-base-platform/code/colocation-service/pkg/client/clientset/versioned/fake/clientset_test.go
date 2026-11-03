/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package fake

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewSimpleClientset(t *testing.T) {
	cs := NewSimpleClientset()
	assert.NotNil(t, cs)
	assert.NotNil(t, cs.Discovery())
	assert.NotNil(t, cs.Tracker())
}

func TestClientset_ColocationV1(t *testing.T) {
	cs := NewSimpleClientset()
	v1 := cs.ColocationV1()
	assert.NotNil(t, v1)
}
