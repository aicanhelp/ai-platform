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

package server

import "testing"

func TestNewServer(t *testing.T) {
	server := NewServer()
	if server.BindAddress != "0.0.0.0" {
		t.Errorf("Expected BindAddress to be '0.0.0.0', got '%s'", server.BindAddress)
	}
	if server.InsecurePort != 9197 {
		t.Errorf("Expected InsecurePort to be 9197, got %d", server.InsecurePort)
	}
	if server.SecurePort != 0 {
		t.Errorf("Expected SecurePort to be 0, got %d", server.SecurePort)
	}
}
func TestNewRunConfig(t *testing.T) {
	runConfig := NewRunConfig()
	if runConfig.Config == nil {
		t.Fatal("Expected Config to be non-nil")
	}
	if runConfig.Config.BindAddress != "0.0.0.0" {
		t.Errorf("Expected Config.BindAddress to be '0.0.0.0', got '%s'", runConfig.Config.BindAddress)
	}
	if runConfig.Config.InsecurePort != 9197 {
		t.Errorf("Expected Config.InsecurePort to be 9197, got %d", runConfig.Config.InsecurePort)
	}
	if runConfig.Config.SecurePort != 0 {
		t.Errorf("Expected Config.SecurePort to be 0, got %d", runConfig.Config.SecurePort)
	}
}
