/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package mfutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v3"
)

func TestDedupFlagsLastWinsAndKeepsOrder(t *testing.T) {
	args := []string{
		"kube-apiserver",
		"--v=0",
		"--secure-port=6443",
		"--v=4",
		"--profiling=false",
		"--secure-port=7443",
	}

	assert.Equal(t, []string{
		"kube-apiserver",
		"--v=4",
		"--secure-port=7443",
		"--profiling=false",
	}, DedupFlags(args))
}

func TestDedupFlagsKeepsBinaryAndNonFlags(t *testing.T) {
	args := []string{
		"kube-controller-manager",
		"positional",
		"--bind-address=127.0.0.1",
		"--bind-address=0.0.0.0",
		"-v=2",
	}

	assert.Equal(t, []string{
		"kube-controller-manager",
		"positional",
		"--bind-address=0.0.0.0",
		"-v=2",
	}, DedupFlags(args))
}

func TestDedupFlagsHandlesBoolFlags(t *testing.T) {
	args := []string{
		"kube-scheduler",
		"--leader-elect",
		"--leader-elect=false",
	}

	assert.Equal(t, []string{
		"kube-scheduler",
		"--leader-elect=false",
	}, DedupFlags(args))
}

func TestDedupPodCommandNode(t *testing.T) {
	data := []byte(`
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: kube-apiserver
    command:
    - kube-apiserver
    - --v=0
    - --authorization-mode=Node,RBAC
    - --v=5
    - --authorization-mode=AlwaysAllow
  - name: sidecar
    command:
    - sidecar
    - --listen=127.0.0.1
`)

	var doc yaml.Node
	assert.NoError(t, yaml.Unmarshal(data, &doc))
	assert.True(t, dedupPodCommandNode(&doc))

	spec := mappingValue(doc.Content[0], "spec")
	containers := mappingValue(spec, "containers")
	command := mappingValue(containers.Content[0], "command")
	got := make([]string, 0, len(command.Content))
	for _, item := range command.Content {
		got = append(got, item.Value)
	}

	assert.Equal(t, []string{
		"kube-apiserver",
		"--v=5",
		"--authorization-mode=AlwaysAllow",
	}, got)
}
