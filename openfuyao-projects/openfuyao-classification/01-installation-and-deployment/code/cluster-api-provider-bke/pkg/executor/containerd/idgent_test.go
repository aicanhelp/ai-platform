/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package containerd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenerateIDReturnsCorrectLength(t *testing.T) {
	id := GenerateID()

	assert.Equal(t, IDLength, len(id))
}

func TestGenerateIDReturnsHexadecimal(t *testing.T) {
	id := GenerateID()

	for _, char := range id {
		assert.True(t, (char >= '0' && char <= '9') || (char >= 'a' && char <= 'f'),
			"ID should contain only hexadecimal characters")
	}
}

func TestGenerateIDReturnsUniqueIDs(t *testing.T) {
	id1 := GenerateID()
	id2 := GenerateID()

	assert.NotEqual(t, id1, id2)
}

func TestGenerateIDConformsToPattern(t *testing.T) {
	id := GenerateID()

	assert.Regexp(t, "^[0-9a-f]{64}$", id)
}
