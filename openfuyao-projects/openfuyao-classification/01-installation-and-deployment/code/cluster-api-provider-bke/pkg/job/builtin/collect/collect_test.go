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

package collect

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	edocker "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	rt "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/runtime"
)

type mockExecutor struct {
	output string
	err    error
}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return m.output, m.err
}

func (m *mockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestCollectPluginName(t *testing.T) {
	p := &CollectPlugin{}
	assert.Equal(t, Name, p.Name())
}

func TestCollectPluginParam(t *testing.T) {
	p := &CollectPlugin{}
	params := p.Param()
	assert.NotNil(t, params)
	assert.Contains(t, params, "clusterName")
	assert.Contains(t, params, "namespace")
	assert.Contains(t, params, "clusterType")
	assert.Contains(t, params, "k8sCertDir")
	assert.Contains(t, params, "etcdCertDir")
}

func TestCollectPluginParamDefaults(t *testing.T) {
	p := &CollectPlugin{}
	params := p.Param()
	assert.Equal(t, "default", params["namespace"].Default)
	assert.NotEmpty(t, params["k8sCertDir"].Default)
	assert.NotEmpty(t, params["etcdCertDir"].Default)
}

func TestNewCollectPlugin(t *testing.T) {
	p := New(nil, nil)
	assert.NotNil(t, p)
	assert.Equal(t, Name, p.Name())
}

func TestCollectPluginExecuteMissingClusterName(t *testing.T) {
	p := &CollectPlugin{}
	commands := []string{Name}
	result, err := p.Execute(context.Background(), commands)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "clusterName")
	assert.Empty(t, result)
}

func TestCollectPluginNewWithNilClients(t *testing.T) {
	p := New(nil, nil)
	assert.NotNil(t, p)
	assert.Equal(t, Name, p.Name())
}

func TestCollectPluginFields(t *testing.T) {
	p := &CollectPlugin{
		clusterName: "test-cluster",
		clusterType: "bke",
		nameSpace:   "test-ns",
	}
	assert.Equal(t, "test-cluster", p.clusterName)
	assert.Equal(t, "bke", p.clusterType)
	assert.Equal(t, "test-ns", p.nameSpace)
}

func TestCollectPluginExecuteWithMockExecutor(t *testing.T) {
	mockExec := &mockExecutor{output: "", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"namespace=test-ns",
		"clusterType=bke",
		"k8sCertDir=/etc/kubernetes/pki",
		"etcdCertDir=/etc/kubernetes/pki/etcd",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestCollectPluginExecuteWithDefaultCertPath(t *testing.T) {
	mockExec := &mockExecutor{output: "", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestCollectPluginWithNilExecutor(t *testing.T) {
	p := New(nil, nil)
	commands := []string{
		Name,
		"clusterName=test-cluster",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestCollectPluginCollectMachineInfo(t *testing.T) {
	p := &CollectPlugin{}
	result := p.collectMachineInfo()
	assert.NotNil(t, result)
	assert.Equal(t, 3, len(result))
}

func TestCollectPluginCollectKubeletDataRootDir(t *testing.T) {
	p := &CollectPlugin{}
	dir := p.collectKubeletDataRootDir()
	assert.NotEmpty(t, dir)
}

func TestCollectPluginCollectKubeletDataRootDirWithMock(t *testing.T) {
	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	p.controllerRuntime = "docker"
	dir := p.collectKubeletDataRootDir()
	assert.Equal(t, "/var/lib/kubelet", dir)
}

func TestCollectPluginCollectKubeletDataRootDirWithMockError(t *testing.T) {
	mockExec := &mockExecutor{output: "", err: fmt.Errorf("mock error")}
	p := New(nil, mockExec)
	p.controllerRuntime = "docker"
	dir := p.collectKubeletDataRootDir()
	assert.NotEmpty(t, dir)
}

func TestCollectPluginCollectKubeletDataRootDirWithMockEmpty(t *testing.T) {
	mockExec := &mockExecutor{output: "", err: nil}
	p := New(nil, mockExec)
	p.controllerRuntime = "docker"
	dir := p.collectKubeletDataRootDir()
	assert.NotEmpty(t, dir)
}

func TestCollectPluginCollectKubeletDataRootDirContainerd(t *testing.T) {
	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	p.controllerRuntime = "containerd"
	dir := p.collectKubeletDataRootDir()
	assert.Equal(t, "/var/lib/kubelet", dir)
}

func TestCollectPluginCollectKubeletDataRootDirContainerdError(t *testing.T) {
	mockExec := &mockExecutor{output: "", err: fmt.Errorf("mock error")}
	p := New(nil, mockExec)
	p.controllerRuntime = "containerd"
	dir := p.collectKubeletDataRootDir()
	assert.NotEmpty(t, dir)
}

func TestExtractRootDirFromArgs(t *testing.T) {
	tests := []struct {
		name     string
		output   string
		expected string
	}{
		{
			name:     "empty output",
			output:   "",
			expected: initialize.DefaultKubeletRootDir,
		},
		{
			name:     "short output",
			output:   "[",
			expected: initialize.DefaultKubeletRootDir,
		},
		{
			name:     "valid output with root-dir",
			output:   "[--root-dir=/custom/kubelet]",
			expected: "/custom/kubelet",
		},
		{
			name:     "output without root-dir",
			output:   "[--network=host]",
			expected: initialize.DefaultKubeletRootDir,
		},
		{
			name:     "invalid split format",
			output:   "[--root-dir]",
			expected: initialize.DefaultKubeletRootDir,
		},
		{
			name:     "root-dir with trailing slash",
			output:   "[--root-dir=/var/lib/kubelet/]",
			expected: "/var/lib/kubelet",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractRootDirFromArgs(tt.output)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCollectPluginCollectDockerMachineInfoWithMock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, fmt.Errorf("config error")
	})

	p := &CollectPlugin{}
	runtimeResult, cgroup, dataRoot := p.collectDockerMachineInfo()
	assert.Equal(t, initialize.DefaultRuntime, runtimeResult)
	assert.Equal(t, initialize.DefaultCgroupDriver, cgroup)
	assert.Equal(t, initialize.DefaultCRIDockerDataRootDir, dataRoot)
}

func TestCollectPluginCollectDockerMachineInfoNilConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	p := &CollectPlugin{}
	runtimeResult, cgroup, dataRoot := p.collectDockerMachineInfo()
	assert.Equal(t, initialize.DefaultRuntime, runtimeResult)
	assert.Equal(t, initialize.DefaultCgroupDriver, cgroup)
	assert.Equal(t, initialize.DefaultCRIDockerDataRootDir, dataRoot)
}

func TestCollectPluginCollectContainerdMachineInfoWithMockError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(containerd.GetContainerdConfig, func(path string) (interface{}, error) {
		return nil, fmt.Errorf("config error")
	})

	p := &CollectPlugin{}
	runtimeResult, cgroup, dataRoot := p.collectContainerdMachineInfo()
	assert.Equal(t, initialize.DefaultRuntime, runtimeResult)
	assert.Equal(t, initialize.DefaultCgroupDriver, cgroup)
	assert.Equal(t, initialize.DefaultCRIContainerdDataRootDir, dataRoot)
}

func TestCollectPluginCollectContainerdMachineInfoNilConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(containerd.GetContainerdConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	p := &CollectPlugin{}
	runtimeResult, cgroup, dataRoot := p.collectContainerdMachineInfo()
	assert.Equal(t, initialize.DefaultRuntime, runtimeResult)
	assert.Equal(t, initialize.DefaultCgroupDriver, cgroup)
	assert.Equal(t, initialize.DefaultCRIContainerdDataRootDir, dataRoot)
}

func TestCollectPluginCollectMachineInfoWithDockerRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeDocker
	})

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	p := &CollectPlugin{}
	result := p.collectMachineInfo()
	assert.Len(t, result, 3)
}

func TestCollectPluginCollectMachineInfoWithContainerdRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeContainerd
	})

	patches.ApplyFunc(containerd.GetContainerdConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	p := &CollectPlugin{}
	result := p.collectMachineInfo()
	assert.Len(t, result, 3)
}

func TestCollectPluginCollectMachineInfoWithUnknownRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return "unknown"
	})

	p := &CollectPlugin{}
	result := p.collectMachineInfo()
	assert.Len(t, result, 3)
}

func TestCollectPluginCollectCertsWithBocloudCerts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(pkiutil.GetBocloudCertListWithoutEtcd, func() pkiutil.BocloudCertificates {
		return nil
	})

	patches.ApplyFunc(pkiutil.GetBocloudCertListForEtcd, func() pkiutil.BocloudCertificates {
		return nil
	})

	patches.ApplyFunc(pkiutil.UploadBocloudCertToClusterAPI, func(c client.Client, certSpec *pkiutil.BocloudCert, namespace, clusterName string) error {
		return nil
	})

	p := &CollectPlugin{
		nameSpace:   "test-ns",
		clusterName: "test-cluster",
	}
	err := p.collectCerts("/custom/pki", "/custom/etcd")
	assert.NoError(t, err)
}

func TestCollectPluginCollectCertsWithBKECerts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(pkiutil.GetCertsWithoutEtcd, func() pkiutil.Certificates {
		return pkiutil.Certificates{}
	})

	patches.ApplyFunc(pkiutil.GetEtcdCerts, func() pkiutil.Certificates {
		return pkiutil.Certificates{}
	})

	patches.ApplyFunc(pkiutil.UploadBKECertToClusterAPI, func(c client.Client, certSpec *pkiutil.BKECert, namespace, clusterName string) error {
		return nil
	})

	p := &CollectPlugin{
		nameSpace:   "test-ns",
		clusterName: "test-cluster",
	}
	err := p.collectCerts(pkiutil.GetDefaultPkiPath(), pkiutil.GetDefaultPkiPath())
	assert.NoError(t, err)
}

func TestCollectPluginCollectCertsWithUploadErrors(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(pkiutil.GetCertsWithoutEtcd, func() pkiutil.Certificates {
		return pkiutil.Certificates{}
	})

	patches.ApplyFunc(pkiutil.GetEtcdCerts, func() pkiutil.Certificates {
		return pkiutil.Certificates{}
	})

	patches.ApplyFunc(pkiutil.UploadBKECertToClusterAPI, func(c client.Client, certSpec *pkiutil.BKECert, namespace, clusterName string) error {
		return fmt.Errorf("upload error")
	})

	p := &CollectPlugin{
		nameSpace:   "test-ns",
		clusterName: "test-cluster",
	}
	err := p.collectCerts(pkiutil.GetDefaultPkiPath(), pkiutil.GetDefaultPkiPath())
	assert.NoError(t, err)
}

func TestCollectPluginExecuteWithDockerRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeDocker
	})

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"namespace=test-ns",
		"clusterType=bke",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.GreaterOrEqual(t, len(result), 4)
}

func TestCollectPluginExecuteWithContainerdRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeContainerd
	})

	patches.ApplyFunc(containerd.GetContainerdConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"namespace=test-ns",
		"clusterType=bke",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.GreaterOrEqual(t, len(result), 4)
}

func TestCollectPluginExecuteWithUnknownRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return ""
	})

	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"namespace=test-ns",
		"clusterType=bke",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.GreaterOrEqual(t, len(result), 4)
}

func TestCollectPluginExecuteWithTrailingSlash(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeDocker
	})

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"k8sCertDir=/etc/kubernetes/pki/",
		"etcdCertDir=/etc/kubernetes/pki/etcd/",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestCollectPluginParamValidation(t *testing.T) {
	p := &CollectPlugin{}
	params := p.Param()

	assert.True(t, params["clusterName"].Required)
	assert.False(t, params["namespace"].Required)
	assert.False(t, params["clusterType"].Required)
	assert.False(t, params["k8sCertDir"].Required)
	assert.False(t, params["etcdCertDir"].Required)

	assert.Equal(t, "default", params["namespace"].Default)
	assert.Equal(t, pkiutil.GetDefaultPkiPath(), params["k8sCertDir"].Default)
	assert.Equal(t, pkiutil.GetDefaultEtcdPkiPath(), params["etcdCertDir"].Default)
}

func TestCollectPluginExecuteWithBocloudClusterType(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeDocker
	})

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	patches.ApplyFunc(pkiutil.GetBocloudCertListWithoutEtcd, func() pkiutil.BocloudCertificates {
		return nil
	})

	patches.ApplyFunc(pkiutil.GetBocloudCertListForEtcd, func() pkiutil.BocloudCertificates {
		return nil
	})

	patches.ApplyFunc(pkiutil.UploadBocloudCertToClusterAPI, func(c client.Client, certSpec *pkiutil.BocloudCert, namespace, clusterName string) error {
		return nil
	})

	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"namespace=test-ns",
		"clusterType=bocloud",
		"k8sCertDir=/custom/pki",
		"etcdCertDir=/custom/etcd",
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "bocloud", result[3])
}

func TestCollectPluginExecuteWithBKECertPath(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(rt.DetectRuntime, func() string {
		return rt.ContainerRuntimeDocker
	})

	patches.ApplyFunc(edocker.GetDockerDaemonConfig, func(path string) (interface{}, error) {
		return nil, nil
	})

	patches.ApplyFunc(pkiutil.GetCertsWithoutEtcd, func() pkiutil.Certificates {
		return pkiutil.Certificates{}
	})

	patches.ApplyFunc(pkiutil.GetEtcdCerts, func() pkiutil.Certificates {
		return pkiutil.Certificates{}
	})

	patches.ApplyFunc(pkiutil.UploadBKECertToClusterAPI, func(c client.Client, certSpec *pkiutil.BKECert, namespace, clusterName string) error {
		return nil
	})

	mockExec := &mockExecutor{output: "[--root-dir=/var/lib/kubelet]", err: nil}
	p := New(nil, mockExec)
	commands := []string{
		Name,
		"clusterName=test-cluster",
		"namespace=test-ns",
		"clusterType=bke",
		"k8sCertDir=" + pkiutil.GetDefaultPkiPath(),
		"etcdCertDir=" + pkiutil.GetDefaultPkiPath(),
	}
	result, err := p.Execute(context.Background(), commands)
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "bke", result[3])
}
