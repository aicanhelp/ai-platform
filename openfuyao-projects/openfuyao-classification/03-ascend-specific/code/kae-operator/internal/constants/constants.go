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

// package constants store constants
package constants

const (
	// ServiceAccount stands for resource type ServiceAccount
	ServiceAccount = "ServiceAccount"
	// Role stands for resource type Role
	Role = "Role"
	// RoleBinding stands for resource type RoleBinding
	RoleBinding = "RoleBinding"
	// ClusterRole stands for resource type ClusterRole
	ClusterRole = "ClusterRole"
	// ClusterRoleBinding stands for resource type ClusterRoleBinding
	ClusterRoleBinding = "ClusterRoleBinding"
	// ConfigMap stands for resource type ConfigMap
	ConfigMap = "ConfigMap"
	// Service stands for resource type Service
	Service = "Service"
	// Deployment stands for resource type Deployment
	Deployment = "Deployment"
	// DaemonSet stands for resource type DaemonSet
	DaemonSet = "DaemonSet"
)

const (
	// DriverDaemonSetName stands for KAE driver DaemonSet
	DriverDaemonSetName = "kae-driver-daemonset"
	// HpreDevicePluginDaemonSetName stands for HPRE device plugin DaemonSet
	HpreDevicePluginDaemonSetName = "kae-hpre-device-plugin-daemonset"
	// SecDevicePluginDaemonSetName stands for SEC device plugin DaemonSet
	SecDevicePluginDaemonSetName = "kae-sec-device-plugin-daemonset"
	// ZipDevicePluginDaemonSetName stands for ZIP device plugin DaemonSet
	ZipDevicePluginDaemonSetName = "kae-zip-device-plugin-daemonset"
)

const (
	// ComponentDriver stands for KAE component driver
	ComponentDriver = "component-driver"
	// ComponentDriverPluginHpre stands for KAE component driver HPRE plugin
	ComponentDriverPluginHpre = "component-device-plugin-hpre"
	// ComponentDriverPluginSec stands for KAE component driver SEC plugin
	ComponentDriverPluginSec = "component-device-plugin-sec"
	// ComponentDriverPluginZip stands for KAE component driver Zip plugin
	ComponentDriverPluginZip = "component-device-plugin-zip"
)
