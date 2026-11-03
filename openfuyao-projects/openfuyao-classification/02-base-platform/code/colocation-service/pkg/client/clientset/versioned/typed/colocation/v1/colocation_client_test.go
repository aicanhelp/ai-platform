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

package v1

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/rest"
	fakerest "k8s.io/client-go/rest/fake"
)

func TestNew(t *testing.T) {
	c := New(&fakerest.RESTClient{})
	assert.NotNil(t, c)
}

func TestColocationV1Client_RESTClient(t *testing.T) {
	rc := &fakerest.RESTClient{}
	c := &ColocationV1Client{restClient: rc}
	assert.NotNil(t, c.RESTClient())
}

func TestColocationV1Client_RESTClient_Nil(t *testing.T) {
	var c *ColocationV1Client
	assert.Nil(t, c.RESTClient())
}

func TestColocationV1Client_ColocationWorkloadMatchRules(t *testing.T) {
	c := New(&fakerest.RESTClient{})
	i := c.ColocationWorkloadMatchRules("default")
	assert.NotNil(t, i)
}

func TestNewForConfig(t *testing.T) {
	cfg := &rest.Config{
		Host: "http://localhost:8080",
	}
	c, err := NewForConfig(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, c)
}

func TestNewForConfigOrDie(t *testing.T) {
	cfg := &rest.Config{
		Host: "http://localhost:8080",
	}
	c := NewForConfigOrDie(cfg)
	assert.NotNil(t, c)
}
