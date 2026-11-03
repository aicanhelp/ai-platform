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

/*
Package constant
contains constant for plugin-management-service
*/
package constant

// plugin-management-service host constant
const (
	ResourcesPluralCluster                   = "clusters"
	PluginManagementServiceDefaultNamespace  = "openfuyao-system"
	PluginManagementServiceDefaultHost       = "plugin-management"
	PluginManagementServiceDefaultAPIVersion = "v1beta1"
	PluginManagementServiceDefaultOrgName    = "openfuyao.com"
)

// regular expression constant
const (
	MetadataNameRegExPattern = "[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
)

// restful response code
const (
	Success                = 200
	FileCreated            = 201
	NoContent              = 204
	ClientError            = 400
	ExceedChartUploadLimit = 4001
	ResourceNotFound       = 404
	ServerError            = 500
)

// consoleplugin-management-service k8s component
const (
	PluginManagementServiceConfigmap = "plugin-management-service-config"
	PluginManagementServiceTokenKey  = "plugin-management-service-token-key"
	PluginManagementServiceSecretKey = "plugin-management-service-secret-key"
	SymmetricKey                     = "plugin-management-service-symmetric-key"
)

// helm chart keyword constant
const (
	FuyaoExtensionKeyword  = "openfuyao-extension"
	OpenFuyaoAuthHeaderKey = "X-OpenFuyao-Authorization"
	UserKey                = "user"
)

// numeric constant
const (
	BaseTen                   = 10
	DefaultHttpRequestSeconds = 30
)

// CRD version and group constant
const (
	CRDRepoGroup   = "console.openfuyao.com"
	CRDRepoVersion = "v1beta1"
)

// cert path constant
const (
	CAPath      = "/ssl/ca.pem"
	TLSCertPath = "/ssl/server.crt"
	TLSKeyPath  = "/ssl/server.key"
)

// param const
const (
	PluginName = "pluginName"
)
