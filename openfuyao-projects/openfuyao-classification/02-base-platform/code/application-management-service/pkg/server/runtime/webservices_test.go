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
	"reflect"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestNewWebService(t *testing.T) {
	tests := []struct {
		name string
		gv   schema.GroupVersion
		want *restful.WebService
	}{
		{
			name: "TestNewWebService",
			gv:   groupVersion,
			want: &restful.WebService{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewRestfulWebService(tt.gv); reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewWebService() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewWebServiceFromStr(t *testing.T) {
	tests := []struct {
		name    string
		subPath string
		want    *restful.WebService
	}{
		{
			name:    "TestNewWebServiceFromStr",
			subPath: "auth",
			want:    &restful.WebService{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewWebServiceFromStr(tt.subPath); reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewWebServiceFromStr() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewWebServiceFromStr_PathTrimRightSlash(t *testing.T) {
	ws := NewWebServiceFromStr("auth/")
	if ws.RootPath() != ApiRootPath+"/auth" {
		t.Fatalf("expected %s/auth, got %s", ApiRootPath, ws.RootPath())
	}
}

func TestNewRestfulWebService_UsesGroupVersionString(t *testing.T) {
	gv := schema.GroupVersion{Group: "example.group", Version: "v1"}
	ws := NewRestfulWebService(gv)
	if ws == nil {
		t.Fatalf("expected non-nil webservice")
	}
	if ws.RootPath() != ApiRootPath+"/"+gv.String() {
		t.Fatalf("expected path %s/%s, got %s", ApiRootPath, gv.String(), ws.RootPath())
	}
}

func TestGetApplicationManagementWebService_NotNil(t *testing.T) {
	ws := GetApplicationManagementWebService()
	if ws == nil {
		t.Fatalf("Expected non-nil web service")
	}
}

func TestGetApplicationManagementWebService_SameInstance(t *testing.T) {
	ws1 := GetApplicationManagementWebService()
	ws2 := GetApplicationManagementWebService()
	if ws1 != ws2 {
		t.Fatalf("Expected same web service instance")
	}
}

func TestGetApplicationManagementWebService_PathInitialized(t *testing.T) {
	ws := GetApplicationManagementWebService()
	if ws.RootPath() == "" {
		t.Fatalf("Expected RootPath to be set")
	}
	// Should be under ApiRootPath
	if len(ws.RootPath()) < len(ApiRootPath) || ws.RootPath()[:len(ApiRootPath)] != ApiRootPath {
		t.Fatalf("Expected RootPath to start with %s, got %s", ApiRootPath, ws.RootPath())
	}
}

func TestInitRestfulRegister_DoesNotPanic(t *testing.T) {
	// initRestfulRegister registers entity accessors; calling again should not panic.
	initRestfulRegister()
	_ = restful.NewEntityAccessorJSON(restful.MIME_JSON)
}