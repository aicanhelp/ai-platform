/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package reset

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"

	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
)

func TestIsResidualK3sProcess(t *testing.T) {
	tests := []struct {
		name    string
		cmdline string
		want    bool
	}{
		{name: "k3s server", cmdline: "/usr/local/bin/k3s server --https-listen-port=36443", want: true},
		{name: "bin k3s", cmdline: "/bin/k3s server --token=xxx", want: true},
		{name: "k3s-server", cmdline: "/usr/bin/k3s-server", want: true},
		{name: "bke command", cmdline: "/usr/local/bin/bke reset", want: false},
		{name: "unrelated", cmdline: "/usr/bin/dockerd", want: false},
		{name: "empty", cmdline: "", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isResidualK3sProcess(tt.cmdline); got != tt.want {
				t.Fatalf("isResidualK3sProcess(%q)=%v, want %v", tt.cmdline, got, tt.want)
			}
		})
	}
}

func TestIsResidualKubeProcess(t *testing.T) {
	tests := []struct {
		name    string
		cmdline string
		want    bool
	}{
		{name: "kubelet", cmdline: "/usr/bin/kubelet --config=/var/lib/kubelet/config.yaml", want: true},
		{name: "kube-apiserver", cmdline: "/usr/local/bin/kube-apiserver --advertise-address=1.2.3.4", want: true},
		{name: "kube-controller-manager", cmdline: "kube-controller-manager --leader-elect=true", want: true},
		{name: "kube-scheduler", cmdline: "kube-scheduler --config=/etc/kubernetes/scheduler.conf", want: true},
		{name: "kube-proxy", cmdline: "/usr/bin/kube-proxy --cluster-cidr=10.244.0.0/16", want: true},
		{name: "etcd", cmdline: "/usr/local/bin/etcd --data-dir=/var/lib/etcd", want: true},
		{name: "k3s server", cmdline: "/usr/local/bin/k3s server --https-listen-port=36443", want: true},
		{name: "bke command", cmdline: "/usr/local/bin/bke reset --all --mount", want: false},
		{name: "unrelated", cmdline: "/usr/bin/dockerd", want: false},
		{name: "empty", cmdline: "", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isResidualKubeProcess(tt.cmdline); got != tt.want {
				t.Fatalf("isResidualKubeProcess(%q)=%v, want %v", tt.cmdline, got, tt.want)
			}
		})
	}
}

func TestRemoveContainerWithProcessFallbackNoRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })

	removeContainerWithProcessFallback("missing-container", isResidualKubeProcess)
}

func TestKillProcessTreeInvalidPID(t *testing.T) {
	killProcessTree(-1)
}
