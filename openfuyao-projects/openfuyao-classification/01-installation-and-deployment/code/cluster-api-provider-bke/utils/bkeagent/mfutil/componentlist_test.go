/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package mfutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHAComponentsSetMfPath(t *testing.T) {
	components := HAComponents{
		{Name: "test1"},
		{Name: "test2"},
	}
	components.SetMfPath("/test/path")
	assert.Equal(t, "/test/path", components[0].MfPath)
	assert.Equal(t, "/test/path", components[1].MfPath)
}

func TestComponentsSetMfPath(t *testing.T) {
	components := Components{
		{Name: "test1"},
		{Name: "test2"},
	}
	components.SetMfPath("/test/path")
	assert.Equal(t, "/test/path", components[0].MfPath)
	assert.Equal(t, "/test/path", components[1].MfPath)
}

func TestGetHAComponentList(t *testing.T) {
	list := GetHAComponentList()
	assert.Len(t, list, 2)
	assert.Equal(t, HAProxy, list[0].Name)
	assert.Equal(t, Keepalived, list[1].Name)
}

func TestGetIngressHaComponentList(t *testing.T) {
	list := GetIngressHaComponentList()
	assert.Len(t, list, 1)
	assert.Equal(t, Keepalived, list[0].Name)
}

func TestGetDefaultComponentList(t *testing.T) {
	list := GetDefaultComponentList()
	assert.Len(t, list, 4)
	assert.Equal(t, KubeAPIServer, list[0].Name)
	assert.Equal(t, KubeScheduler, list[1].Name)
	assert.Equal(t, KubeControllerManager, list[2].Name)
	assert.Equal(t, Etcd, list[3].Name)
}

func TestGetComponentListWithOutEtcd(t *testing.T) {
	list := GetComponentListWithOutEtcd()
	assert.Len(t, list, 3)
	assert.Equal(t, KubeAPIServer, list[0].Name)
	assert.Equal(t, KubeScheduler, list[1].Name)
	assert.Equal(t, KubeControllerManager, list[2].Name)
}

func TestAPIServerComponent(t *testing.T) {
	comp := APIServerComponent()
	assert.NotNil(t, comp)
	assert.Equal(t, KubeAPIServer, comp.Name)
	assert.NotNil(t, comp.RenderFunc)
}

func TestSchedulerComponent(t *testing.T) {
	comp := SchedulerComponent()
	assert.NotNil(t, comp)
	assert.Equal(t, KubeScheduler, comp.Name)
	assert.NotNil(t, comp.RenderFunc)
}

func TestControllerComponent(t *testing.T) {
	comp := ControllerComponent()
	assert.NotNil(t, comp)
	assert.Equal(t, KubeControllerManager, comp.Name)
	assert.NotNil(t, comp.RenderFunc)
}

func TestEtcdComponent(t *testing.T) {
	comp := EtcdComponent()
	assert.NotNil(t, comp)
	assert.Equal(t, Etcd, comp.Name)
	assert.NotNil(t, comp.RenderFunc)
}

func TestHAProxyComponent(t *testing.T) {
	comp := HAProxyComponent()
	assert.NotNil(t, comp)
	assert.Equal(t, HAProxy, comp.Name)
	assert.NotNil(t, comp.RenderFunc)
}

func TestKeepalivedComponent(t *testing.T) {
	comp := KeepalivedComponent()
	assert.NotNil(t, comp)
	assert.Equal(t, Keepalived, comp.Name)
	assert.NotNil(t, comp.RenderFunc)
}
