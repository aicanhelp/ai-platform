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

package constant

const (
	// ResourcesPluralCluster is resource collection
	ResourcesPluralCluster = "clusters"
)

// restful response code
const (
	Success          = 200
	FileCreated      = 201
	NoContent        = 204
	ClientError      = 400
	ResourceNotFound = 404
	ServerError      = 500
)

const (
	NodeRoleMasterLabel = "node-role.kubernetes.io/master"
	NodeRoleNodeLabel   = "node-role.kubernetes.io/node"
)

const (
	// PatchPath is path to store all upgrade patches
	PatchPath = "/bke/mount/source_registry/files/patches"
	// PatchKeyPrefix is prefix for patch key in configmap
	// since configmap does not support nested values, use this prefix to identify patch config map
	PatchKeyPrefix = "patch."
	// PatchValuePrefix is prefix for patch value in configmap
	// since configmap does not support nested values, use this prefix to identify patch config map
	PatchValuePrefix = "cm."
	// PatchNameSpace is the namespace of patch info
	PatchNameSpace = "openfuyao-patch"
	// DefaultRemoteDeployPath is the default manifest location, all online deploy files should be uploaded there
	DefaultRemoteDeployPath = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/latest/"
	// DefaultRemotePatchPath is the default manifest location, all online patch files should be uploaded there
	DefaultRemotePatchPath = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/"
)
