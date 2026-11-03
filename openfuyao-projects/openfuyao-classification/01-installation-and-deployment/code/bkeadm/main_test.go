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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightbridge"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
)

// TestEmbeddedPreflightPolicyCompiles freezes that the single embedded policy
// is the runtime source of truth: it loads, validates, and compiles into a
// Bundle the SDK checker accepts.
func TestEmbeddedPreflightPolicyCompiles(t *testing.T) {
	config, err := preflightconfig.Load(embeddedPreflightPolicy)
	require.NoError(t, err)
	require.NoError(t, preflightconfig.Validate(config))
	raw, err := preflightbridge.BuildBundle(config)
	require.NoError(t, err)
	assert.NotEmpty(t, raw)
}
