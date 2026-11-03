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

package logging

import (
	"fmt"
	"testing"

	"openfuyao/logging-operator/pkg/server"
)

const TestInsecurePort = 8080

func TestNewAPIServer(t *testing.T) {
	test := struct {
		name      string
		cfg       *server.RunConfig
		expectErr bool
	}{

		name:      "Valid Configuration",
		cfg:       &server.RunConfig{Config: &server.Config{InsecurePort: TestInsecurePort}},
		expectErr: false,
	}

	apiServer, err := NewAPIServer(test.cfg)
	if test.expectErr {
		if err == nil {
			t.Errorf("Expected error but got nil")
		} else {
			t.Logf("Received expected error: %v", err)
		}
	} else {
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if apiServer == nil {
			t.Errorf("Expected APIServer, but got nil")
		}
		if apiServer.Server.Addr != fmt.Sprintf(":%d", test.cfg.Config.InsecurePort) {
			t.Errorf("Expected server address :%d, but got %s", test.cfg.Config.InsecurePort, apiServer.Server.Addr)
		}
	}

}

func TestInitServer(t *testing.T) {

	test := struct {
		name      string
		cfg       *server.RunConfig
		expectErr bool
	}{

		name:      "Valid Configuration",
		cfg:       &server.RunConfig{Config: &server.Config{InsecurePort: TestInsecurePort}},
		expectErr: false,
	}

	server, err := initServer(test.cfg)
	if test.expectErr {
		if err == nil {
			t.Errorf("Expected error but got nil")
		}
	} else {
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if server == nil {
			t.Errorf("Expected http.Server, but got nil")
		}
		expectedAddr := fmt.Sprintf(":%d", test.cfg.Config.InsecurePort)
		if server.Addr != expectedAddr {
			t.Errorf("Expected server address %s, but got %s", expectedAddr, server.Addr)
		}
	}
}
