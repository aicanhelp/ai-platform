/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package releasefacts

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/preflight/api"
)

func TestStaticRulesCount(t *testing.T) {
	rules := StaticRules()
	// The release catalog contains the reviewed, minimal init and
	// cluster-create rule packs.
	assert.Equal(t, 73, len(rules))
	for _, r := range rules {
		assert.Nil(t, r.Validate())
	}
}

func TestStaticRulesNoSensitive(t *testing.T) {
	// The static rules must not carry credentials. Source/Fix must not contain
	// password-like values.
	for _, r := range StaticRules() {
		assert.NotContains(t, r.Fix, "password")
		assert.NotContains(t, r.Fix, "token")
	}
}

func TestRulesAreData(t *testing.T) {
	// Rules are data: they carry no execution logic, only eq/expected/fix.
	for _, r := range StaticRules() {
		assert.NotEmpty(t, r.ID)
		assert.NotEmpty(t, r.Fix)
		assert.True(t, r.Op.Valid())
		assert.NotEmpty(t, r.Operation)
	}
	// Smoke: ensure init rules use OperationInit, cluster rules use cluster-create.
	var hasInit, hasCluster bool
	for _, r := range StaticRules() {
		if r.Operation == api.OperationInit {
			hasInit = true
		}
		if r.Operation == api.OperationClusterCreate {
			hasCluster = true
		}
	}
	assert.True(t, hasInit)
	assert.True(t, hasCluster)
}
