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

package preflightlogenv

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsPreflightCommand(t *testing.T) {
	assert.True(t, isPreflightCommand([]string{"bke", "preflight", "init"}))
	assert.True(t, isPreflightCommand([]string{"bke", "--kubeconfig", "/tmp/k", "preflight", "cluster-create"}))
	assert.True(t, isPreflightCommand([]string{"bke", "--kubeconfig=/tmp/k", "preflight", "cluster-create"}))
	assert.False(t, isPreflightCommand([]string{"bke", "init"}))
	assert.False(t, isPreflightCommand([]string{"bke", "init", "--file", "preflight-cluster.yaml"}))
	assert.False(t, isPreflightCommand([]string{"bke", "init", "--domain", "preflight-prod"}))
}

func TestIsPreflightCommandFlagVariants(t *testing.T) {
// Value-taking root flags in both forms are skipped before the subcommand.
assert.True(t, isPreflightCommand([]string{"bke", "--log-level", "debug", "preflight", "init"}))
assert.True(t, isPreflightCommand([]string{"bke", "--log-level=debug", "preflight", "init"}))
// Help-style flags never claim the command slot.
assert.False(t, isPreflightCommand([]string{"bke", "--doc"}))
assert.False(t, isPreflightCommand([]string{"bke", "--help"}))
assert.False(t, isPreflightCommand([]string{"bke", "-h"}))
// Only the first non-flag token decides; later tokens are irrelevant.
assert.False(t, isPreflightCommand([]string{"bke", "cluster", "preflight"}))
assert.True(t, isPreflightCommand([]string{"bke", "preflight", "whatever"}))
// Empty tails and bare binaries are not preflight.
assert.False(t, isPreflightCommand([]string{"bke"}))
assert.False(t, isPreflightCommand([]string{"bke", "--kubeconfig", "/tmp/k"}))
}
