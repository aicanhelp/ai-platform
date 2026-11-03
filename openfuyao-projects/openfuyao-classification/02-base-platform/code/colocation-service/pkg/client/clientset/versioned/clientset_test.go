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

package versioned

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/discovery"
	fakerest "k8s.io/client-go/rest/fake"
	"k8s.io/client-go/rest"

	colocationv1 "openfuyao.com/colocation-service/pkg/client/clientset/versioned/typed/colocation/v1"
)

func TestNew(t *testing.T) {
	c := New(&fakerest.RESTClient{})
	assert.NotNil(t, c)
	assert.NotNil(t, c.ColocationV1())
	assert.NotNil(t, c.Discovery())
}

func TestClientset_Discovery_Nil(t *testing.T) {
	var c *Clientset
	assert.Nil(t, c.Discovery())
}

func TestClientset_Discovery_NonNil(t *testing.T) {
	c := &Clientset{
		DiscoveryClient: discovery.NewDiscoveryClient(&fakerest.RESTClient{}),
	}
	assert.NotNil(t, c.Discovery())
}

func TestClientset_ColocationV1(t *testing.T) {
	restClient := &fakerest.RESTClient{}
	c := &Clientset{
		colocationV1: colocationv1.New(restClient),
	}
	assert.NotNil(t, c.ColocationV1())
}

func TestNewForConfig(t *testing.T) {
	cfg := &rest.Config{
		Host: "http://localhost:8080",
	}
	c, err := NewForConfig(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, c)
}

func TestNewForConfigAndClient(t *testing.T) {
	cfg := &rest.Config{
		Host: "http://localhost:8080",
	}
	httpClient := &http.Client{}
	c, err := NewForConfigAndClient(cfg, httpClient)
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
