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

package mfutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	numZero       = 0
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFour       = 4
	numEight      = 8
	numTen        = 10
	numSixteen    = 16
	numOneHundred = 100
)

func TestGetAuditPolicyFilePath(t *testing.T) {
	result := GetAuditPolicyFilePath()
	assert.Equal(t, "/etc/kubernetes/audit-policy.yaml", result)
}

func TestGetDefaultManifestsPath(t *testing.T) {
	result := GetDefaultManifestsPath()
	assert.True(t, result == "/etc/kubernetes/manifests" || result == "\\etc\\kubernetes\\manifests")
}

func TestGetSchedulerPolicyFilePath(t *testing.T) {
	result := GetSchedulerPolicyFilePath()
	assert.True(t, result == "/etc/kubernetes/scheduler-policy-config.json" || result == "\\etc\\kubernetes\\scheduler-policy-config.json")
}

func TestGetSchedulerAdmissionConfigFilePath(t *testing.T) {
	result := GetSchedulerAdmissionConfigFilePath()
	assert.True(t, result == "/etc/kubernetes/gpu-admission.config" || result == "\\etc\\kubernetes\\gpu-admission.config")
}

func TestKubernetesConstants(t *testing.T) {
	assert.Equal(t, "/etc/kubernetes", KubernetesDir)
	assert.Equal(t, "manifests", ManifestsSubDirName)
	assert.Equal(t, "etcd", Etcd)
	assert.Equal(t, "kube-apiserver", KubeAPIServer)
	assert.Equal(t, "kube-controller-manager", KubeControllerManager)
	assert.Equal(t, "kube-scheduler", KubeScheduler)
	assert.Equal(t, "kubelet", Kubelet)
	assert.Equal(t, "audit-policy.yaml", AuditPolicyFileName)
	assert.Equal(t, "/var/lib/openFuyao/etcd", EtcdDataDir)
}

func TestHAConstants(t *testing.T) {
	assert.Equal(t, "haproxy", HAProxy)
	assert.Equal(t, "keepalived", Keepalived)
	assert.Equal(t, "ingress-Keepalived", IngressKeepalived)
	assert.Equal(t, "/etc/openFuyao/haproxy", HAProxyConfPath)
	assert.Equal(t, "haproxy.cfg", HAProxyConfName)
	assert.Equal(t, "/etc/openFuyao/keepalived", KeepAlivedConfPath)
	assert.Equal(t, "keepalived.conf", KeepAlivedConfName)
}
