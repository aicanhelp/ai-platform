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

package applyconfiguration

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

func TestForKind_ColocationWorkloadMatchRule(t *testing.T) {
	gvk := v1.SchemeGroupVersion.WithKind("ColocationWorkloadMatchRule")
	result := ForKind(gvk)
	assert.NotNil(t, result)
}

func TestForKind_ColocationWorkloadMatchRuleSpec(t *testing.T) {
	gvk := v1.SchemeGroupVersion.WithKind("ColocationWorkloadMatchRuleSpec")
	result := ForKind(gvk)
	assert.NotNil(t, result)
}

func TestForKind_NameSelector(t *testing.T) {
	gvk := v1.SchemeGroupVersion.WithKind("NameSelector")
	result := ForKind(gvk)
	assert.NotNil(t, result)
}

func TestForKind_Unknown(t *testing.T) {
	gvk := v1.SchemeGroupVersion.WithKind("Unknown")
	result := ForKind(gvk)
	assert.Nil(t, result)
}
