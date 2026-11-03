/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package runtime

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"

	"monitoring-dashboard-service/pkg/config/global"
)

// TestNewServerConfig 校验默认ServerConfig
func TestNewServerConfig(t *testing.T) {
	cfg := NewServerConfig()
	if cfg.BindAddress != "0.0.0.0" {
		t.Errorf("unexpected bind address: %s", cfg.BindAddress)
	}
	if cfg.InsecurePort != 9297 {
		t.Errorf("unexpected insecure port: %d", cfg.InsecurePort)
	}
	if cfg.SecurePort != 0 {
		t.Errorf("unexpected secure port: %d", cfg.SecurePort)
	}
}

// TestNewRunConfig 校验RunConfig组装
func TestNewRunConfig(t *testing.T) {
	cfg := NewRunConfig()
	if cfg.Server == nil {
		t.Error("expected non-nil Server config")
	}
	if cfg.KubernetesCfg == nil {
		t.Error("expected non-nil Kubernetes config")
	}
}

// TestNewRestfulWebService 校验按GroupVersion创建webservice
func TestNewRestfulWebService(t *testing.T) {
	gv := schema.GroupVersion{
		Group:   global.MonitorServiceDefaultHost,
		Version: global.MonitorServiceDefaultAPIVersion,
	}
	ws := NewRestfulWebService(gv)
	if ws == nil {
		t.Fatal("NewRestfulWebService returned nil")
	}
	expectedPath := global.ApiRootPath + "/" + gv.String()
	if ws.RootPath() != expectedPath {
		t.Errorf("expected root path %s, got %s", expectedPath, ws.RootPath())
	}
}

// TestNewWebServiceFromStrEmptyPath 校验空subPath时路径裁剪
func TestNewWebServiceFromStrEmptyPath(t *testing.T) {
	ws := NewWebServiceFromStr("")
	if ws.RootPath() != global.ApiRootPath {
		t.Errorf("expected root path %s, got %s", global.ApiRootPath, ws.RootPath())
	}
}

// TestGetIndicatorWebService 校验全局webservice获取
func TestGetIndicatorWebService(t *testing.T) {
	ws := GetIndicatorWebService()
	if ws == nil {
		t.Fatal("GetIndicatorWebService returned nil")
	}
	expectedPath := global.ApiRootPath + "/" +
		schema.GroupVersion{Group: global.MonitorServiceDefaultHost,
			Version: global.MonitorServiceDefaultAPIVersion}.String()
	if ws.RootPath() != expectedPath {
		t.Errorf("expected root path %s, got %s", expectedPath, ws.RootPath())
	}
}
