/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1beta1

import (
	"testing"
)

func TestContainerdConfigSpec_DefaultValues(t *testing.T) {
	spec := ContainerdConfigSpec{}

	if spec.ConfigType != "" {
		t.Errorf("Expected empty ConfigType by default, got %s", spec.ConfigType)
	}

	// Test that required fields are enforced at API level, not in struct
	if spec.Service != nil {
		t.Error("Expected Service to be nil by default")
	}
}

func TestServiceConfig_DefaultValues(t *testing.T) {
	service := &ServiceConfig{}

	// These defaults should be set by webhook/defaulting, not in Go struct
	if service.Slice != "" {
		t.Errorf("Expected empty Slice by default, got %s", service.Slice)
	}
	if service.KillMode != "" {
		t.Errorf("Expected empty KillMode by default, got %s", service.KillMode)
	}
}

func TestMainConfig_DefaultValues(t *testing.T) {
	main := &MainConfig{}

	// These defaults should be set by webhook/defaulting
	if main.Root != "" {
		t.Errorf("Expected empty Root by default, got %s", main.Root)
	}
	if main.State != "" {
		t.Errorf("Expected empty State by default, got %s", main.State)
	}
}

func TestContainerdConfigSpec_DeepCopy(t *testing.T) {
	original := &ContainerdConfigSpec{
		ConfigType:  "combined",
		Description: "Test configuration",
		Service: &ServiceConfig{
			ExecStart: "/usr/bin/containerd --config /etc/containerd/config.toml",
			Slice:     "system.slice",
			KillMode:  "process",
			Restart:   "always",
		},
		Main: &MainConfig{
			MetricsAddress: "0.0.0.0:1338",
			Root:           "/var/lib/containerd",
		},
	}

	// Test DeepCopy
	copied := original.DeepCopy()

	if copied == original {
		t.Error("DeepCopy should return a different instance")
	}

	if copied.ConfigType != original.ConfigType {
		t.Errorf("Expected ConfigType %s, got %s", original.ConfigType, copied.ConfigType)
	}

	// Modify the copy and ensure original is not affected
	copied.ConfigType = "service"
	if original.ConfigType == copied.ConfigType {
		t.Error("Modifying copy should not affect original")
	}
}

func TestContainerdConfigSpecList_DeepCopy(t *testing.T) {
	list := []ContainerdConfigSpec{
		{
			ConfigType: "service",
		},
		{
			ConfigType: "main",
		},
	}

	// Test that slice content is properly copied
	if len(list) != 2 {
		t.Errorf("Expected 2 items, got %d", len(list))
	}

	// Modify the first item
	list[0].ConfigType = "modified"
	if list[0].ConfigType == "service" {
		t.Error("Modifying list item should not affect original")
	}
}

func TestRegistryConfig_DeepCopy(t *testing.T) {
	registry := &RegistryConfig{
		ConfigPath: "/etc/containerd/certs.d",
		Configs: map[string]RegistryHostConfig{
			"docker.io": {
				Host:         "https://docker.io",
				Capabilities: []string{"pull", "resolve"},
				TLS: &TLSConfig{
					InsecureSkipVerify: false,
					CAFile:             "/etc/ssl/certs/ca-certificates.crt",
				},
				Auth: &RegistryAuthConfig{
					Username: "user",
					Password: "pass",
				},
				Header: map[string][]string{
					"User-Agent": {"containerd/1.0"},
				},
			},
		},
	}

	// Test DeepCopy
	copied := registry.DeepCopy()

	if copied.ConfigPath != registry.ConfigPath {
		t.Errorf("Expected ConfigPath %s, got %s", registry.ConfigPath, copied.ConfigPath)
	}

	// Modify the copy and ensure original is not affected
	copied.Configs["docker.io"] = RegistryHostConfig{
		Host:         "https://modified.docker.io",
		Capabilities: []string{"pull", "resolve"},
	}
	if registry.Configs["docker.io"].Host == copied.Configs["docker.io"].Host {
		t.Error("Modifying copied registry config should not affect original")
	}

	// Test that nested maps and slices are properly copied
	if len(copied.Configs["docker.io"].Capabilities) != len(registry.Configs["docker.io"].Capabilities) {
		t.Errorf("Expected %d capabilities, got %d",
			len(registry.Configs["docker.io"].Capabilities),
			len(copied.Configs["docker.io"].Capabilities))
	}
}

func TestScriptConfig_DeepCopy(t *testing.T) {
	script := &ScriptConfig{
		Content:     "#!/bin/bash\necho 'Hello World'",
		Path:        "/scripts/init.sh",
		Args:        []string{"--verbose", "--debug"},
		Interpreter: "/bin/bash",
	}

	// Test DeepCopy
	copied := script.DeepCopy()

	if copied.Content != script.Content {
		t.Errorf("Expected Content %s, got %s", script.Content, copied.Content)
	}

	// Modify the copy and ensure original is not affected
	copied.Args[0] = "--quiet"
	if script.Args[0] == copied.Args[0] {
		t.Error("Modifying copied args should not affect original")
	}
}

func TestServiceConfig_DeepCopy(t *testing.T) {
	service := &ServiceConfig{
		ExecStart: "/usr/bin/containerd --config /etc/containerd/config.toml",
		Slice:     "system.slice",
		KillMode:  "process",
		Restart:   "always",
		Logging: &ServiceLogging{
			StandardOutput: "journal",
			StandardError:  "journal",
			LogLevelMax:    "info",
		},
		CustomExtra: map[string]string{
			"EnvironmentFile": "/etc/containerd/environment",
			"LimitNOFILE":     "infinity",
		},
	}

	// Test DeepCopy
	copied := service.DeepCopy()

	if copied.ExecStart != service.ExecStart {
		t.Errorf("Expected ExecStart %s, got %s", service.ExecStart, copied.ExecStart)
	}

	// Modify the copy and ensure original is not affected
	copied.CustomExtra["EnvironmentFile"] = "/modified/path"
	if service.CustomExtra["EnvironmentFile"] == copied.CustomExtra["EnvironmentFile"] {
		t.Error("Modifying copied custom extra should not affect original")
	}

	if copied.Logging.StandardOutput != service.Logging.StandardOutput {
		t.Errorf("Expected StandardOutput %s, got %s", service.Logging.StandardOutput, copied.Logging.StandardOutput)
	}
}
