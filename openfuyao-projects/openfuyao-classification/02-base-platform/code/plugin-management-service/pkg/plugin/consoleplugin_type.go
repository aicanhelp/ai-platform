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

// package plugin defines the model for consoleplugin and a console plugin manager
package plugin

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// ConsolePluginSpec specifies the expected status of a console consoleplugin resource
type ConsolePluginSpec struct {
	// PluginName is the unique name of the consoleplugin. The name should only include alphabets, digits and '-'
	PluginName string `json:"pluginName"`

	// Order is the index of the consoleplugin. The value should only be non-negative integers
	Order *int64 `json:"order,omitempty"`

	// DisplayName is the display name of the consoleplugin on the UI entrypoint, should be between 1 and 128 characters.
	DisplayName string `json:"displayName"`

	// SubPages stands for the pages under the main console consoleplugin. Only applicable for "Side" Entrypoint
	SubPages []ConsolePluginName `json:"subPages,omitempty"`

	// Entrypoint is the location where the entrypoint of the consoleplugin will be rendered on the console webpage.
	// Current support values are [Nav, Side]
	Entrypoint ConsolePluginEntrypoint `json:"entrypoint"`

	// Backend holds the configuration of backend which is serving console's consoleplugin.
	Backend *ConsolePluginBackend `json:"backend"`

	// Enabled specifies whether the consoleplugin would be loaded on console webpage.
	// Default tto be true (would be loaded)
	Enabled bool `json:"enabled"`
}

// ConsolePluginName is the name of the consoleplugin
type ConsolePluginName struct {
	// PageName is the unique name of the page. The name should only include alphabets, digits and '-'
	PageName string `json:"pageName"`

	// DisplayName is the display name of the consoleplugin on the UI entrypoint, should be between 1 and 128 characters.
	DisplayName string `json:"displayName"`
}

// ConsolePluginEntrypoint is an enumeration of entrypoint location
type ConsolePluginEntrypoint string

const (
	// NavEntrypoint renders the entrypoint in the top navigation bar.
	NavEntrypoint ConsolePluginEntrypoint = "Nav"

	// SideEntrypoint renders the entrypoint in the side menu.
	SideEntrypoint ConsolePluginEntrypoint = "Side"
)

// ConsolePluginBackend holds information about the endpoint which serves the consoleplugin.
type ConsolePluginBackend struct {
	// Type is the type of the backend that supplies the consoleplugin UI resources.
	// Currently only service is supported.
	// Current supported types are [Service]
	Type ConsolePluginBackendType `json:"type"`

	// Service is the kubernetes service that exposes the consoleplugin UI resources using a
	// deployment with an HTTP server.
	Service *ConsolePluginService `json:"service"`
}

// ConsolePluginBackendType is an enumeration of types of the backend that serves the consoleplugin UI resource.
type ConsolePluginBackendType string

const (
	// ServiceBackendType means the UI resource of the consoleplugin is supplied by a kubernetes service resource.
	ServiceBackendType ConsolePluginBackendType = "Service"
)

// ConsolePluginService holds information of the service that is serving consoleplugin UI resources.
type ConsolePluginService struct {
	// Name of the service serving the consoleplugin UI resources.
	Name string `json:"name"`

	// Namespace of the service serving the consoleplugin UI resources.
	Namespace string `json:"namespace"`

	// Port on which the service serving the consoleplugin is listening to.
	// This field is optional, default to be 80
	Port int32 `json:"port"`

	// BasePath is the base path to the consoleplugin UI resource in the HTTP server.
	// This field is optional, default to be /
	BasePath string `json:"basePath"`
}

// ConsolePluginStatus defines the observed state of ConsolePlugin
type ConsolePluginStatus struct {
	// Link is the URL with which the front-end load the consoleplugin UI resource
	Link string `json:"link"`
}

// ConsolePlugin is the Schema for the consoleplugins API
type ConsolePlugin struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ConsolePluginSpec   `json:"spec,omitempty"`
	Status ConsolePluginStatus `json:"status,omitempty"`
}

// ConsolePluginList contains a list of ConsolePlugin
type ConsolePluginList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ConsolePlugin `json:"items"`
}
