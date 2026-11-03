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

package multiclusterutil

import (
	"reflect"
	"testing"

	"console-service/pkg/server/request"
)

func TestReturnDummyHostCluster(t *testing.T) {
	want := &ClusterList{Info: map[string]*ClusterInformation{"host": {ClusterName: "host"}}}
	if got := ReturnDummyHostCluster(); !reflect.DeepEqual(got, want) {
		t.Errorf("ReturnDummyHostCluster() = %v, want %v", got, want)
	}
}

func TestIsMultiClusterRequest(t *testing.T) {
	info := &request.RequestInfo{
		ClusterProxyHost: "",
	}
	if got := IsMultiClusterRequest(info); got {
		t.Errorf("When ClusterProxyHost is empty, IsMultiClusterRequest() wants false, but true")
	}
}
