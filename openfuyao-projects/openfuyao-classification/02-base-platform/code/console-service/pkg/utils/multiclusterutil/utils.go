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

// Package multiclusterutil contains public function for multicluster routing
package multiclusterutil

import (
	"strings"

	"console-service/pkg/constant"
	"console-service/pkg/server/request"
)

// IsMultiClusterRequest checks whether the requestinfo is multi-cluster request
func IsMultiClusterRequest(info *request.RequestInfo) bool {
	return info.ClusterProxyHost != ""
}

// IsMultiClusterRequestURL checks whether the url is multi-cluster request
func IsMultiClusterRequestURL(url string) bool {
	return strings.Contains(url, constant.MultiClusterProxyHost)
}

// ClusterList represents the list for clueter objects
type ClusterList struct {
	Info map[string]*ClusterInformation `json:"info,omitempty"`
}

// ClusterInformation contains the cluster name of the specific cluster
type ClusterInformation struct {
	ClusterName string `json:"clustername,omitempty"`
}

// ReturnDummyHostCluster only return the clustername of host cluster when multi-cluster extension is not installed
func ReturnDummyHostCluster() *ClusterList {
	return &ClusterList{Info: map[string]*ClusterInformation{"host": {ClusterName: "host"}}}
}
