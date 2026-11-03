/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phaseutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestStructToUnstructured(t *testing.T) {
	user := &User{
		TypeMeta:   v1.TypeMeta{Kind: "User", APIVersion: "v1"},
		ObjectMeta: v1.ObjectMeta{Name: "test"},
	}
	result, err := StructToUnstructured(user)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestPrepareUserInstance(t *testing.T) {
	user := prepareUserInstance("admin", "encrypted")
	assert.Equal(t, "admin", user.Name)
	assert.Equal(t, "admin", user.Spec.Username)
}

func TestGeneratePassword(t *testing.T) {
	password := generatePassword(16)
	assert.Equal(t, 16, len(password))
}

func TestGenerateRandomIndex(t *testing.T) {
	index := generateRandomIndex(100)
	assert.GreaterOrEqual(t, index, 0)
	assert.Less(t, index, 100)
}
