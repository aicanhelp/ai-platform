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

package colocation_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
	"openfuyao.com/colocation-service/pkg/client/informers/externalversions"
	"openfuyao.com/colocation-service/pkg/client/informers/externalversions/colocation"
)

func TestNew(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := externalversions.NewSharedInformerFactory(cs, time.Minute)
	g := colocation.New(factory, "default", nil)
	assert.NotNil(t, g)
}

func TestGroup_V1(t *testing.T) {
	cs := fake.NewSimpleClientset()
	factory := externalversions.NewSharedInformerFactory(cs, time.Minute)
	g := colocation.New(factory, "default", nil)
	v := g.V1()
	assert.NotNil(t, v)
}
