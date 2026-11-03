/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package utils

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetSupportPlatforms(t *testing.T) {
	platforms := GetSupportPlatforms()
	assert.Len(t, platforms, 3)
	assert.Contains(t, platforms, "centos")
	assert.Contains(t, platforms, "kylin")
	assert.Contains(t, platforms, "ubuntu")
}

func TestGetDefaultImageRepo(t *testing.T) {
	repo := GetDefaultImageRepo()
	assert.Equal(t, "deploy.bocloud.k8s:40443/kubernetes/", repo)
}

func TestGetKubeletConfPath(t *testing.T) {
	path := GetKubeletConfPath()
	expected := filepath.Join(KubeletConfigPath, KubeletConfigFileName)
	assert.Equal(t, expected, path)
}

func TestGetKubeletScriptPath(t *testing.T) {
	path := GetKubeletScriptPath()
	expected := filepath.Join(KubernetesDir, KubeletScriptName)
	assert.Equal(t, expected, path)
}

func TestGetKubeletServicePath(t *testing.T) {
	path := GetKubeletServicePath()
	expected := filepath.Join(SystemdDir, KubeletServiceUnitName)
	assert.Equal(t, expected, path)
}

func TestGetRunKubeletPreCreateDirs(t *testing.T) {
	dirs := GetRunKubeletPreCreateDirs()
	assert.Len(t, dirs, 13)
	assert.Contains(t, dirs, "/var/lib/kubelet")
	assert.Contains(t, dirs, "/etc/kubernetes")
	assert.Contains(t, dirs, "/var/lib/openFuyao/etcd")
}
