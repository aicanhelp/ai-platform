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
contains constant for console-service
*/
package constant

// console-service host constant
const (
	ResourcesPluralCluster          = "clusters"
	ConsoleServiceDefaultNamespace  = "openfuyao-system"
	ConsoleServiceDefaultHost       = "console"
	ConsolePluginDefaultHost        = "console/plugin"
	ConsoleServiceDefaultAPIVersion = "v1beta1"
	ConsoleServiceDefaultOrgName    = "openfuyao.com"
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

// console-service k8s component
const (
	ConsoleServiceConfigmap = "console-service-config"
	ConsoleServiceTokenKey  = "console-service-token-key"
	ConsoleServiceSecretKey = "console-service-secret-key"
	SymmetricKey            = "console-service-symmetric-key"
)

// numeric constant
const (
	SearchParamLengthLimit    = 53
	BaseTen                   = 10
	SixtyFourBits             = 64
	DefaultHttpRequestSeconds = 30
)

// cert path constant
const (
	CAPath                      = "/ssl/ca.pem"
	TLSCertPath                 = "/ssl/server.crt"
	TLSKeyPath                  = "/ssl/server.key"
	AlertCAPath                 = "/ssl/alert/ca.pem"
	AlertTLSCertPath            = "/ssl/alert/server.crt"
	AlertTLSKeyPath             = "/ssl/alert/server.key"
	K8sServiceAccountRootCAPath = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
)

// multicluster constants
const (
	KaramdaNamespace        = "karmada-system"
	KaramdaAPIServer        = "karmada-apiserver"
	KaramadaAPIServerPort   = "5443"
	MultiClusterProxyScheme = "https"
	MultiClusterProxyHost   = KaramdaAPIServer + "." + KaramdaNamespace + ".svc.cluster.local:" + KaramadaAPIServerPort
	MultiClusterProxyURL    = "/apis/cluster.karmada.io/v1alpha1/clusters/{cluster}/proxy"

	MultiClusterPluginName = "multicluster"

	SingleClusterProxyScheme = "https"
	SingleClusterProxyHost   = "kubernetes.default.svc.cluster.local:443"
)

// default service/proxy
const (
	ServiceProxyURL  = "/api/v1/namespaces/{namespace}/services/{service}:{port}/proxy"
	DefaultHttpPort  = 80
	DefaultHttpsPort = 443
)

// OpenFuyaoAuthorization header
const (
	OpenFuyaoAuthHeader = "X-OpenFuyao-Authorization"
)

// cookie name for saving original request URL before OAuth redirect
const (
	CookieNameReferrer = "X-OpenFuyao-Referrer"
)

// oauth error string
const (
	OnlyAccessTokenExpiredErrorStr = "the access-token has expired"
)
