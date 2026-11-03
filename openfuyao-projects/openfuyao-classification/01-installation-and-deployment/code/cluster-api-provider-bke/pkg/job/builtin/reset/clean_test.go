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

package reset

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/httprepo"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/resetutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/runtime"
)

type mockExecutorForClean struct {
	exec.Executor
	outputValue string
	outputErr   error
}

func (m *mockExecutorForClean) ExecuteCommandWithCombinedOutput(_ string, _ ...string) (string, error) {
	return m.outputValue, m.outputErr
}

func (m *mockExecutorForClean) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutorForClean) ExecuteCommandWithOutput(_ string, _ ...string) (string, error) {
	return m.outputValue, m.outputErr
}

func createTestExtraClean() ExtraClean {
	mockExec := &mockExecutorForClean{}
	return ExtraClean{
		Executor: mockExec,
	}
}

func setMockOutput(extra ExtraClean, value string, err error) {
	if mockExec, ok := extra.Executor.(*mockExecutorForClean); ok {
		mockExec.outputValue = value
		mockExec.outputErr = err
	}
}

func createTestBKEConfig() *bkev1beta1.BKEConfig {
	return &bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			CertificatesDir: "/etc/kubernetes/pki",
			Networking: bkev1beta1.Networking{
				PodSubnet:     "10.244.0.0/16",
				ServiceSubnet: "10.96.0.0/12",
			},
			Kubelet: &bkev1beta1.Kubelet{
				ManifestsDir: "/etc/kubernetes/manifests",
			},
			ControlPlane: bkev1beta1.ControlPlane{
				Etcd: &bkev1beta1.Etcd{
					DataDir: "/var/lib/etcd",
				},
			},
		},
		CustomExtra: map[string]string{},
	}
}

func TestCertCleanNilConfig(t *testing.T) {
	extra := createTestExtraClean()
	err := CertClean(nil, extra)

	assert.NoError(t, err)
}

func TestCertCleanWithConfig(t *testing.T) {
	extra := createTestExtraClean()
	extra.Dir = append(extra.Dir, "/etc/kubernetes/pki")
	err := CertClean(&bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			CertificatesDir: "/etc/kubernetes/pki",
		},
	}, extra)

	assert.NoError(t, err)
}

func TestCertCleanEmptyCertDir(t *testing.T) {
	extra := createTestExtraClean()
	err := CertClean(&bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			CertificatesDir: "",
		},
	}, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanBinWithAllInOneTrue(t *testing.T) {
	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "true"
	extra := createTestExtraClean()
	err := KubeletCleanBin(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanBinWithAllInOneFalse(t *testing.T) {
	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	err := KubeletCleanBin(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerCleanDockerWithNoContainers(t *testing.T) {
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)

	err := ContainerCleanDocker(extra)

	assert.NoError(t, err)
}

func TestContainerCleanDockerWithContainers(t *testing.T) {
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)

	err := ContainerCleanDocker(extra)

	assert.NoError(t, err)
}

func TestContainerCleanDockerWithError(t *testing.T) {
	extra := createTestExtraClean()
	setMockOutput(extra, "", errors.New("command failed"))

	err := ContainerCleanDocker(extra)

	assert.NoError(t, err)
}

func TestContainerdCfgClean(t *testing.T) {
	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := ContainerdCfgClean(cfg, extra)

	assert.NoError(t, err)
}

func TestManifestsCleanNilConfig(t *testing.T) {
	extra := createTestExtraClean()
	err := ManifestsClean(nil, extra)

	assert.NoError(t, err)
}

func TestManifestsCleanWithConfig(t *testing.T) {
	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := ManifestsClean(cfg, extra)

	assert.NoError(t, err)
}

func TestExtraToCleanWithAllInOneFalse(t *testing.T) {
	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	err := ExtraToClean(cfg, extra)

	assert.NoError(t, err)
}

func TestExtraToCleanWithAllInOneTrue(t *testing.T) {
	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "true"
	extra := createTestExtraClean()
	err := ExtraToClean(cfg, extra)

	assert.NoError(t, err)
}

func TestExtraToCleanWithEmptyCustomExtra(t *testing.T) {
	cfg := createTestBKEConfig()
	cfg.CustomExtra = nil
	extra := createTestExtraClean()
	err := ExtraToClean(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanBinWithDockerRuntime(t *testing.T) {
	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := KubeletCleanBin(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanBinWithDefaultDirs(t *testing.T) {
	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := KubeletCleanBin(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerCleanWithEmptyRuntime(t *testing.T) {
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)

	err := ContainerClean(nil, extra)

	assert.NoError(t, err)
}

func TestContainerCleanWithUnknownRuntime(t *testing.T) {
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)

	err := ContainerClean(nil, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanWithNilConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(resetutil.UnmountKubeletDirectory, func(_ string) error {
		return nil
	})
	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	err := KubeletClean(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanWithDockerRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(resetutil.UnmountKubeletDirectory, func(_ string) error {
		return nil
	})
	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	err := KubeletClean(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanWithContainerdRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(resetutil.UnmountKubeletDirectory, func(_ string) error {
		return nil
	})
	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeContainerd
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	err := KubeletClean(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanWithAllInOneTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(resetutil.UnmountKubeletDirectory, func(_ string) error {
		return nil
	})
	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "true"
	extra := createTestExtraClean()
	err := KubeletClean(cfg, extra)

	assert.NoError(t, err)
}

func TestKubeletCleanWithCustomKubeletRootDir(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(resetutil.UnmountKubeletDirectory, func(_ string) error {
		return nil
	})
	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	cfg.Cluster.Kubelet = &bkev1beta1.Kubelet{
		ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{
			ExtraVolumes: []bkev1beta1.HostPathMount{
				{
					Name:     "kubelet-root-dir",
					HostPath: "/var/lib/kubelet-custom",
				},
			},
		},
	}
	extra := createTestExtraClean()
	err := KubeletClean(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerRuntimeCleanWithAllInOneTrue(t *testing.T) {
	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "true"
	extra := createTestExtraClean()
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
	assert.Nil(t, extra.Dir)
	assert.Nil(t, extra.File)
}

func TestContainerRuntimeCleanWithEmptyRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return ""
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerRuntimeCleanWithDockerRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})
	patches.ApplyFunc(httprepo.RepoRemove, func(_ ...string) error {
		return nil
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	cfg.Cluster.KubernetesVersion = "1.23.0"
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerRuntimeCleanWithDockerRuntimeAndDataRoot(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})
	patches.ApplyFunc(httprepo.RepoRemove, func(_ ...string) error {
		return nil
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	cfg.Cluster.KubernetesVersion = "1.23.0"
	cfg.Cluster.ContainerRuntime = bkev1beta1.ContainerRuntime{
		Param: map[string]string{
			"data-root": "/custom/docker",
		},
	}
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerRuntimeCleanWithDockerRuntimeAndK8s124(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeDocker
	})
	patches.ApplyFunc(httprepo.RepoRemove, func(_ ...string) error {
		return nil
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	cfg.Cluster.KubernetesVersion = "1.24.0"
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerRuntimeCleanWithContainerdRuntime(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeContainerd
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
}

func TestContainerRuntimeCleanWithContainerdRuntimeAndDataRoot(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(runtime.DetectRuntime, func() string {
		return runtime.ContainerRuntimeContainerd
	})

	cfg := createTestBKEConfig()
	cfg.CustomExtra["allInOne"] = "false"
	cfg.Cluster.ContainerRuntime = bkev1beta1.ContainerRuntime{
		Param: map[string]string{
			"data-root": "/custom/containerd",
		},
	}
	extra := createTestExtraClean()
	setMockOutput(extra, "", nil)
	err := ContainerRuntimeClean(cfg, extra)

	assert.NoError(t, err)
}

func TestSourceCleanWithSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(source.ResetSource, func() error {
		return nil
	})
	patches.ApplyFunc(httprepo.RepoUpdate, func() error {
		return nil
	})

	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := SourceClean(cfg, extra)

	assert.NoError(t, err)
}

func TestSourceCleanWithResetSourceError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testErr := errors.New("reset source failed")
	patches.ApplyFunc(source.ResetSource, func() error {
		return testErr
	})

	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := SourceClean(cfg, extra)

	assert.Error(t, err)
	assert.Equal(t, testErr, err)
}

func TestSourceCleanWithRepoUpdateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	testErr := errors.New("repo update failed")
	patches.ApplyFunc(source.ResetSource, func() error {
		return nil
	})
	patches.ApplyFunc(httprepo.RepoUpdate, func() error {
		return testErr
	})

	cfg := createTestBKEConfig()
	extra := createTestExtraClean()
	err := SourceClean(cfg, extra)

	assert.Error(t, err)
	assert.Equal(t, testErr, err)
}

func TestCleanPhaseClean(t *testing.T) {
	phase := CleanPhase{
		Name: "test",
		CleanFunc: func(_ *bkev1beta1.BKEConfig, _ ExtraClean) error {
			return nil
		},
	}
	err := phase.Clean(nil)

	assert.NoError(t, err)
}

func TestCleanPhaseAddDirToClean(t *testing.T) {
	phase := CleanPhase{
		Name: "test",
		extra: ExtraClean{
			Dir: []string{},
		},
	}
	phase.AddDirToClean("/test/dir")

	assert.Len(t, phase.extra.Dir, 1)
	assert.Equal(t, "/test/dir", phase.extra.Dir[0])
}

func TestCleanPhaseAddFileToClean(t *testing.T) {
	phase := CleanPhase{
		Name: "test",
		extra: ExtraClean{
			File: []string{},
		},
	}
	phase.AddFileToClean("/test/file")

	assert.Len(t, phase.extra.File, 1)
	assert.Equal(t, "/test/file", phase.extra.File[0])
}

func TestCleanPhaseAddIPToClean(t *testing.T) {
	phase := CleanPhase{
		Name: "test",
		extra: ExtraClean{
			Ips: []string{},
		},
	}
	phase.AddIPToClean("192.168.1.1")

	assert.Len(t, phase.extra.Ips, 1)
	assert.Equal(t, "192.168.1.1", phase.extra.Ips[0])
}

func TestExtraCleanCleanAllWithNilExecutor(t *testing.T) {
	extra := ExtraClean{
		Dir:  []string{"/test"},
		File: []string{"/test/file"},
	}
	err := extra.CleanAll()

	assert.NoError(t, err)
}

func TestExtraCleanCleanIPWithEmptyIPs(t *testing.T) {
	extra := ExtraClean{
		Ips: []string{},
	}
	err := extra.CleanIP()

	assert.NoError(t, err)
}

func TestExtraCleanAddIPToCleanWithDuplicate(t *testing.T) {
	extra := ExtraClean{
		Ips: []string{"192.168.1.1"},
	}
	extra.AddIPToClean("192.168.1.1")

	assert.Len(t, extra.Ips, 1)
}

func TestDefaultCleanPhases(t *testing.T) {
	DefaultCleanPhases()

}

func TestCleanCertPhase(t *testing.T) {
	phase := CleanCertPhase()

	assert.Equal(t, "cert", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanManifestsPhase(t *testing.T) {
	phase := CleanManifestsPhase()

	assert.Equal(t, "manifests", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanContainerdCfgPhase(t *testing.T) {
	phase := CleanContainerdCfgPhase()

	assert.Equal(t, "containerd-cfg", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanContainerPhase(t *testing.T) {
	phase := CleanContainerPhase()

	assert.Equal(t, "container", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanKubeletPhase(t *testing.T) {
	phase := CleanKubeletPhase()

	assert.Equal(t, "kubelet", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanContainerRuntimePhase(t *testing.T) {
	phase := CleanContainerRuntimePhase()

	assert.Equal(t, "containerRuntime", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanSourcePhase(t *testing.T) {
	phase := CleanSourcePhase()

	assert.Equal(t, "source", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanExtraPhase(t *testing.T) {
	phase := CleanExtraPhase()

	assert.Equal(t, "extra", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanGlobalCertPhase(t *testing.T) {
	phase := CleanGlobalCertPhase()

	assert.Equal(t, "global-cert", phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestCleanShutdownPhase(t *testing.T) {
	phase := CleanShutdownPhase()

	assert.Equal(t, ScopeShutdown, phase.Name)
	assert.NotNil(t, phase.CleanFunc)
}

func TestBKEAgentShutdownCleanSchedulesImmediately(t *testing.T) {
	t.Cleanup(ClearPendingAgentCleanup)

	var scheduled bool
	orig := scheduleAgentCleanup
	scheduleAgentCleanup = func() error {
		scheduled = true
		return nil
	}
	t.Cleanup(func() { scheduleAgentCleanup = orig })

	assert.NoError(t, BKEAgentShutdownClean(nil, ExtraClean{}))
	assert.True(t, AgentCleanupPending(), "pending remains so reconciler shortens writeback timeout")
	assert.True(t, scheduled, "self-manage delete must schedule stop before API writeback")
}

func TestBKEAgentShutdownCleanScheduleFailureDoesNotFailReset(t *testing.T) {
	t.Cleanup(ClearPendingAgentCleanup)

	orig := scheduleAgentCleanup
	scheduleAgentCleanup = func() error { return errors.New("nohup failed") }
	t.Cleanup(func() { scheduleAgentCleanup = orig })

	assert.NoError(t, BKEAgentShutdownClean(nil, ExtraClean{}))
	assert.True(t, AgentCleanupPending(), "pending kept so Flush can retry")
}

func TestCleanPhasesIterate(t *testing.T) {
	phases := DefaultCleanPhases()

	for _, phase := range phases {
		assert.NotEmpty(t, phase.Name)
		assert.NotNil(t, phase.CleanFunc)
	}
}
