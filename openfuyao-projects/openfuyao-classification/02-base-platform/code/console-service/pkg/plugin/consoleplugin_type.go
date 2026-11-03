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

package plugin

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// ConsolePluginSpec defines the desired state of ConsolePlugin
type ConsolePluginSpec struct {
	// PluginName is the unique name of the plugin. The name should only include alphabets, digits and '-'
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^[a-zA-Z0-9-]+$`
	PluginName string `json:"pluginName"`

	// Order is the display index of the plugin, only work if the plugin is rendered on the left
	// navigation menu. Negative and out of bounds number will be treated as last index in the menu
	Order *int64 `json:"order,omitempty"`

	// DisplayName is the display name of the plugin on the UI entrypoint, should be between
	// 1 and 256 characters.
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	DisplayName string `json:"displayName"`

	// DisplayNameI18n is the i18n display name map, key is locale (e.g. "en-US", "zh-CN"),
	// value is the display name.
	// +optional
	DisplayNameI18n map[string]string `json:"displayNameI18n,omitempty"`

	// SubPages stands for the pages under the main console plugin. Only applicable for "Side" Entrypoint
	// +optional
	SubPages []ConsolePluginName `json:"subPages,omitempty"`

	// Entrypoint is the path where the entrypoint of the plugin will be rendered on the console webpage.
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^/[a-zA-Z0-9-_/]*$`
	Entrypoint ConsolePluginEntrypoint `json:"entrypoint"`

	// Backend holds the configuration of backend which is serving console's plugin.
	Backend *ConsolePluginBackend `json:"backend"`

	// Enabled specifies whether the plugin would be loaded on console webpage.
	// Default to be true (would be loaded)
	// +optional
	// +kubebuilder:default=true
	Enabled bool `json:"enabled,omitempty"`
}

// ConsolePluginName is the name of the plugin
type ConsolePluginName struct {
	// PageName is the unique name of the page. The name should only include alphabets, digits and '-'
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^[a-zA-Z0-9-]+$`
	PageName string `json:"pageName"`

	// DisplayName is the display name of the plugin on the UI entrypoint,
	// should be between 1 and 256 characters.
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	DisplayName string `json:"displayName"`

	// DisplayNameI18n is the i18n display name map, key is locale (e.g. "en-US", "zh-CN"),
	// value is the display name.
	// +optional
	DisplayNameI18n map[string]string `json:"displayNameI18n,omitempty"`
}

// ConsolePluginEntrypoint is the root path of the mounting point
// e.g.
//   - entrypoint / is the top level mounting point, the console plugin would occupy
//     the whole page except for the top navigation bar.
//   - entrypoint /container_platform would mount the consoleplugin into the "Container Platform"
//     page, occupying page except for the top navigation bar and side menu.
type ConsolePluginEntrypoint string

// ConsolePluginBackend holds the configuration of backend which is serving console's plugin.
type ConsolePluginBackend struct {
	// Type is the type of the backend that supplies the plugin UI resources.
	// Currently only service is supported. Current supported types are [Service]
	Type ConsolePluginBackendType `json:"type"`

	// Service is the kubernetes service that exposes the plugin UI resources
	// using a deployment with an HTTP server.
	// +optional
	Service *ConsolePluginService `json:"service,omitempty"`
}

// +kubebuilder:validation:Enum=Service
type ConsolePluginBackendType string

const (
	ServiceBackendType ConsolePluginBackendType = "Service"
)

// ConsolePluginService is the kubernetes service that exposes the plugin UI resources.
type ConsolePluginService struct {
	// Name of the service serving the plugin UI resources.
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^[a-zA-Z0-9-]+$`
	Name string `json:"name"`

	// Namespace of the service serving the plugin UI resources.
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^[a-zA-Z0-9-]+$`
	Namespace string `json:"namespace"`

	// Port on which the service serving the plugin is listening to.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:default=80
	Port int32 `json:"port"`

	// BasePath is the base path to the plugin UI resource in the HTTP server.
	// +optional
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^/[a-zA-Z0-9-]*$`
	BasePath string `json:"basePath,omitempty"`

	// Scheme specifies the protocol used to communicate with the backend service.
	// Valid values are "http" or "https". Defaults to "http" if not specified.
	// +optional
	// +kubebuilder:validation:Enum=http;https
	// +kubebuilder:default=http
	Scheme string `json:"scheme,omitempty"`

	// CABundle is a PEM-encoded CA certificate bundle used to verify
	// the backend service's TLS certificate.
	// The value is base64-encoded when stored in YAML/JSON.
	// +optional
	CABundle []byte `json:"caBundle,omitempty"`

	// InsecureSkipVerify controls whether the console client skips TLS
	// certificate verification when connecting to the backend service.
	// If true, the CA bundle will be ignored and any certificate is accepted.
	// Use with caution.
	// +optional
	InsecureSkipVerify bool `json:"insecureSkipVerify,omitempty"`
}

// ConsolePluginStatus defines the observed state of ConsolePlugin
type ConsolePluginStatus struct {
	// Link is the URL with which the front-end load the plugin UI resource
	Link string `json:"link"`
}

// ConsolePlugin is the Schema for the consoleplugins API
//
// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=cp,scope=Cluster
// +kubebuilder:subresource:status
type ConsolePlugin struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ConsolePluginSpec   `json:"spec,omitempty"`
	Status ConsolePluginStatus `json:"status,omitempty"`
}

// ConsolePluginList contains a list of ConsolePlugin
//
// +kubebuilder:object:root=true
type ConsolePluginList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ConsolePlugin `json:"items"`
}

// ConsolePluginTrimmed contains only the essential info of a plugin for front-end
type ConsolePluginTrimmed struct {
	Release         string              `json:"release"`
	DisplayName     string              `json:"displayName"`
	DisplayNameI18n map[string]string   `json:"displayNameI18n,omitempty"`
	PluginName      string              `json:"pluginName"`
	Order           *string             `json:"order,omitempty"`
	SubPages        []ConsolePluginName `json:"subPages"`
	Entrypoint      string              `json:"entrypoint"`
	URL             string              `json:"url"`
	Enabled         bool                `json:"enabled"`
}
