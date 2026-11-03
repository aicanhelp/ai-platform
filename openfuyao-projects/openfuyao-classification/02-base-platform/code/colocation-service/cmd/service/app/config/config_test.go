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

package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOverallConfig_Complete(t *testing.T) {
	oc := &OverallConfig{}
	cc := oc.Complete()
	assert.NotNil(t, cc)
}

func TestDefaultServerPort(t *testing.T) {
	assert.Equal(t, 8080, DefaultServerPort)
}
