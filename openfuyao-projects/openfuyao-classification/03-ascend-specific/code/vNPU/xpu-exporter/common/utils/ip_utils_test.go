/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package utils

import (
	"net/http/httptest"
	"testing"
)

func TestClientIP(t *testing.T) {
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("X-Forwarded-For", "192.168.1.1, 192.168.1.2")
	if ip := ClientIP(req); ip != "192.168.1.1" {
		t.Errorf("Want 192.168.1.1, got %s", ip)
	}

	req = httptest.NewRequest("GET", "/", nil)
	req.Header.Set("X-Real-Ip", "192.168.1.3")
	if ip := ClientIP(req); ip != "192.168.1.3" {
		t.Errorf("Want 192.168.1.3, got %s", ip)
	}

	req = httptest.NewRequest("GET", "/", nil)
	req.RemoteAddr = "192.168.1.4:8080"
	if ip := ClientIP(req); ip != "192.168.1.4" {
		t.Errorf("Want 192.168.1.4, got %s", ip)
	}

	req = httptest.NewRequest("GET", "/", nil)
	req.RemoteAddr = ""
	if ip := ClientIP(req); ip != "" {
		t.Errorf("Want '', got %s", ip)
	}
}
