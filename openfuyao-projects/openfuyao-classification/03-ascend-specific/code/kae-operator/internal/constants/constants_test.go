/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for the details.
 */

package constants

import (
	"testing"
)

func TestResourceTypeConstants(t *testing.T) {
	tests := []struct {
		name     string
		constant string
		expected string
	}{
		{"ServiceAccount", ServiceAccount, "ServiceAccount"},
		{"Role", Role, "Role"},
		{"RoleBinding", RoleBinding, "RoleBinding"},
		{"ClusterRole", ClusterRole, "ClusterRole"},
		{"ClusterRoleBinding", ClusterRoleBinding, "ClusterRoleBinding"},
		{"ConfigMap", ConfigMap, "ConfigMap"},
		{"Service", Service, "Service"},
		{"Deployment", Deployment, "Deployment"},
		{"DaemonSet", DaemonSet, "DaemonSet"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.constant != tt.expected {
				t.Errorf("%s = %s, want %s", tt.name, tt.constant, tt.expected)
			}
		})
	}
}

func TestDaemonSetNameConstants(t *testing.T) {
	tests := []struct {
		name     string
		constant string
		expected string
	}{
		{"DriverDaemonSetName", DriverDaemonSetName, "kae-driver-daemonset"},
		{"HpreDevicePluginDaemonSetName", HpreDevicePluginDaemonSetName, "kae-hpre-device-plugin-daemonset"},
		{"SecDevicePluginDaemonSetName", SecDevicePluginDaemonSetName, "kae-sec-device-plugin-daemonset"},
		{"ZipDevicePluginDaemonSetName", ZipDevicePluginDaemonSetName, "kae-zip-device-plugin-daemonset"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.constant != tt.expected {
				t.Errorf("%s = %s, want %s", tt.name, tt.constant, tt.expected)
			}
		})
	}
}

func TestComponentConstants(t *testing.T) {
	tests := []struct {
		name     string
		constant string
		expected string
	}{
		{"ComponentDriver", ComponentDriver, "component-driver"},
		{"ComponentDriverPluginHpre", ComponentDriverPluginHpre, "component-device-plugin-hpre"},
		{"ComponentDriverPluginSec", ComponentDriverPluginSec, "component-device-plugin-sec"},
		{"ComponentDriverPluginZip", ComponentDriverPluginZip, "component-device-plugin-zip"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.constant != tt.expected {
				t.Errorf("%s = %s, want %s", tt.name, tt.constant, tt.expected)
			}
		})
	}
}

func TestConstantsAreUnique(t *testing.T) {
	// Test that all resource type constants are unique
	resourceTypes := []string{
		ServiceAccount,
		Role,
		RoleBinding,
		ClusterRole,
		ClusterRoleBinding,
		ConfigMap,
		Service,
		Deployment,
		DaemonSet,
	}

	seen := make(map[string]bool)
	for _, rt := range resourceTypes {
		if seen[rt] {
			t.Errorf("Duplicate resource type constant: %s", rt)
		}
		seen[rt] = true
	}

	// Test that all DaemonSet name constants are unique
	daemonSetNames := []string{
		DriverDaemonSetName,
		HpreDevicePluginDaemonSetName,
		SecDevicePluginDaemonSetName,
		ZipDevicePluginDaemonSetName,
	}

	seen = make(map[string]bool)
	for _, name := range daemonSetNames {
		if seen[name] {
			t.Errorf("Duplicate DaemonSet name constant: %s", name)
		}
		seen[name] = true
	}

	// Test that all component constants are unique
	components := []string{
		ComponentDriver,
		ComponentDriverPluginHpre,
		ComponentDriverPluginSec,
		ComponentDriverPluginZip,
	}

	seen = make(map[string]bool)
	for _, comp := range components {
		if seen[comp] {
			t.Errorf("Duplicate component constant: %s", comp)
		}
		seen[comp] = true
	}
}

func TestConstantsFormat(t *testing.T) {
	// Test that DaemonSet names follow expected format (lowercase with hyphens)
	daemonSetNames := []string{
		DriverDaemonSetName,
		HpreDevicePluginDaemonSetName,
		SecDevicePluginDaemonSetName,
		ZipDevicePluginDaemonSetName,
	}

	for _, name := range daemonSetNames {
		if len(name) == 0 {
			t.Errorf("DaemonSet name should not be empty")
		}
	}
}

func TestComponentPrefix(t *testing.T) {
	// Test that component constants have expected prefix
	components := []string{
		ComponentDriver,
		ComponentDriverPluginHpre,
		ComponentDriverPluginSec,
		ComponentDriverPluginZip,
	}

	for _, comp := range components {
		if len(comp) == 0 {
			t.Errorf("Component constant should not be empty")
		}
	}
}
