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

// volcano-config-service host constant
const (
	ResourcesPluralCluster          = "clusters"
	ConsoleServiceDefaultNamespace  = "fuyao-system"
	ConsoleServiceDefaultHost       = "console"
	ConsoleServiceDefaultAPIVersion = "v1beta1"
	ConsoleServiceDefaultOrgName    = "openfuyao.com"
)

// volcano-config-service host constant
const (
	VolcanoConfigServiceDefaultNamespace = "volcano-system"
	VolcanoConfigServiceConfigmap        = "volcano-scheduler-configmap"
	VolcanoConfigServiceConfigmapName    = "volcano-scheduler.conf"
)

// regular expression constant
const (
	MetadataNameRegExPattern = "[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*"
)

// restful response code
const (
	Success          = 200
	FileCreated      = 201
	NoContent        = 204
	ClientError      = 400
	ResourceNotFound = 404
	NotAcceptable    = 406
	ServerError      = 500
)

// numa-affinity-webhook time constant
const (
	TimeStart    = 2
	TimeInterval = 5
	TimeDuration = 2
)
