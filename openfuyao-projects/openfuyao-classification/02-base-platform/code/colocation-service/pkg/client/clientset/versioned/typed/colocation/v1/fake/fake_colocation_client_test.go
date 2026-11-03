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
	k8stesting "k8s.io/client-go/testing"
)

func TestFakeColocationV1_ColocationWorkloadMatchRules(t *testing.T) {
	c := &FakeColocationV1{Fake: &k8stesting.Fake{}}
	i := c.ColocationWorkloadMatchRules("default")
	assert.NotNil(t, i)
}

func TestFakeColocationV1_RESTClient(t *testing.T) {
	c := &FakeColocationV1{Fake: &k8stesting.Fake{}}
	rc := c.RESTClient()
	assert.Nil(t, rc)
}
