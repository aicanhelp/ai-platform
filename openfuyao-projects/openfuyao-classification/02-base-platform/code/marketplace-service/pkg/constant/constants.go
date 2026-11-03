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
contains constant for marketplace-service
*/
package constant

// marketplace-service host constant
const (
	ResourcesPluralCluster              = "clusters"
	MarketplaceServiceDefaultNamespace  = "openfuyao-system"
	MarketplaceServiceDefaultHost       = "marketplace"
	MarketplaceServiceDefaultAPIVersion = "v1beta1"
	MarketplaceServiceDefaultOrgName    = "openfuyao.com"
)

// regular expression constant
const (
	MetadataNameRegExPattern = "^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
)

// restful response code
const (
	ResponseBodyLimit = 100
	ToMegabytes       = 20

	Success                = 200
	FileCreated            = 201
	NoContent              = 204
	ClientError            = 400
	ExceedChartUploadLimit = 4001
	ResourceNotFound       = 404
	ServerError            = 500
)

// marketplace-service k8s component
const (
	MarketplaceServiceConfigmap = "marketplace-service-config"
	SecretFieldSymmetricKey     = "marketplace-service-symmetric-key"
)

// helm chart keyword constant
const (
	FuyaoExtensionKeyword = "openfuyao-extension"
)

// numeric constant
const (
	SearchParamLengthLimit    = 53
	BaseTen                   = 10
	SixtyFourBits             = 64
	DefaultHttpRequestSeconds = 30
)

// CRD version and gropu constant
const (
	CRDRepoGroup    = "console.openfuyao.com"
	CRDRepoVersion  = "v1beta1"
	CRDRepoResource = "helmchartrepositories"
	CRDRepoKind     = "HelmChartRepository"
	CRDRepoListKind = "HelmChartRepositoryList"
)

// cert path constant
const (
	CAPath                 = "/ssl/ca.pem"
	TLSCertPath            = "/ssl/server.crt"
	TLSKeyPath             = "/ssl/server.key"
	LocalHarborCAPath      = "/ssl/local-harbor/ca.pem"
	LocalHarborTLSCertPath = "/ssl/local-harbor/server.crt"
	LocalHarborTLSKeyPath  = "/ssl/local-harbor/server.key"
)
