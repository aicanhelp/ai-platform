/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package manifests

import (
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"context"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

const (
	numZero       = 0
	numOne        = 1
	numTwo        = 2
	numThree      = 3
	numFour       = 4
	numFive       = 5
	numSix        = 6
	numSeven      = 7
	numEight      = 8
	numTen        = 10
	numSeventeen  = 17
	numTwentyFour = 24
	numOneHundred = 100

	testNamespace   = "test-namespace"
	testName        = "test-name"
	testManifestDir = "/etc/kubernetes/manifests"
	testEtcdDataDir = "/var/lib/etcd"
	testGPUEnable   = "true"
	testCheck       = "true"
	testScopeAll    = "kube-apiserver,kube-controller-manager,kube-scheduler,etcd"
)

type testExecutor struct {
	output           string
	shouldErr        bool
	errMsg           string
	outputMap        map[string]string
	errCommandPrefix string
}

func (e *testExecutor) WithContext(_ context.Context) exec.Executor {
	return e
}

func (e *testExecutor) ExecuteCommand(command string, arg ...string) error {
	if e.shouldErr {
		return errors.New(e.errMsg)
	}
	return nil
}

func (e *testExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (e *testExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	if e.shouldErr {
		return "", errors.New(e.errMsg)
	}
	if val, ok := e.outputMap[command]; ok {
		return val, nil
	}
	return e.output, nil
}

func (e *testExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	if val, ok := e.outputMap[command]; ok {
		return val, nil
	}
	if e.shouldErr && e.errCommandPrefix != "" && len(command) >= len(e.errCommandPrefix) && command[0:len(e.errCommandPrefix)] == e.errCommandPrefix {
		return "", errors.New(e.errMsg)
	}
	if e.shouldErr {
		return "", errors.New(e.errMsg)
	}
	return e.output, nil
}

func (e *testExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return e.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (e *testExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (e *testExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "", nil
}

func (e *testExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "", nil
}

func (e *testExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

func TestName(t *testing.T) {
	mp := &ManifestPlugin{}
	assert.Equal(t, Name, mp.Name())
}

func TestNew(t *testing.T) {
	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{}

	mp := New(bootScope, executor)

	assert.NotNil(t, mp)
	assert.Equal(t, Name, mp.Name())

	manifestPlugin := mp.(*ManifestPlugin)
	assert.Equal(t, bootScope, manifestPlugin.bootScope)
	assert.Equal(t, executor, manifestPlugin.exec)
}

func TestNewWithNilBootScope(t *testing.T) {
	var bootScope *mfutil.BootScope
	executor := &testExecutor{}

	mp := New(bootScope, executor)

	assert.NotNil(t, mp)

	manifestPlugin := mp.(*ManifestPlugin)
	assert.Nil(t, manifestPlugin.bootScope)
	assert.Equal(t, executor, manifestPlugin.exec)
}

func TestParam(t *testing.T) {
	mp := &ManifestPlugin{}
	params := mp.Param()

	assert.NotNil(t, params)
	assert.Contains(t, params, "bkeConfig")
	assert.Contains(t, params, "scope")
	assert.Contains(t, params, "check")
	assert.Contains(t, params, "gpuEnable")
	assert.Contains(t, params, "manifestDir")
	assert.Contains(t, params, "etcdDataDir")
}

func TestParamDefaultValues(t *testing.T) {
	mp := &ManifestPlugin{}
	params := mp.Param()

	assert.Equal(t, "false", params["check"].Default)
	assert.Equal(t, "false", params["gpuEnable"].Default)
	assert.Equal(t, mfutil.GetDefaultManifestsPath(), params["manifestDir"].Default)
	assert.Equal(t, mfutil.EtcdDataDir, params["etcdDataDir"].Default)
}

func TestParamRequiredFields(t *testing.T) {
	mp := &ManifestPlugin{}
	params := mp.Param()

	assert.False(t, params["bkeConfig"].Required)
	assert.False(t, params["scope"].Required)
	assert.False(t, params["check"].Required)
	assert.False(t, params["gpuEnable"].Required)
	assert.False(t, params["manifestDir"].Required)
	assert.False(t, params["etcdDataDir"].Required)
}

func TestDefaultScope(t *testing.T) {
	mp := &ManifestPlugin{}
	params := mp.Param()

	expected := mfutil.KubeAPIServer + "," + mfutil.KubeControllerManager + "," +
		mfutil.KubeScheduler + "," + mfutil.Etcd

	assert.Equal(t, expected, params["scope"].Default)
}

func TestNewBootScopeWithoutBkeConfig(t *testing.T) {
	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	parseCommands := map[string]string{}

	err := mp.newBootScope(parseCommands)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bkeConfig param is required")
}

func TestExecuteWithBootScope(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		output:    "",
		shouldErr: false,
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=kube-apiserver,etcd",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.KubeAPIServer || item == mfutil.Etcd
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.APIServerComponent(),
			mfutil.SchedulerComponent(),
			mfutil.ControllerComponent(),
			mfutil.EtcdComponent(),
		}
	})

	patches.ApplyFunc(mfutil.GenerateManifestYaml, func(components mfutil.Components, bootScope *mfutil.BootScope) error {
		return nil
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteWithoutBootScopeAndWithoutBkeConfig(t *testing.T) {
	executor := &testExecutor{}
	mp := &ManifestPlugin{
		bootScope: nil,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
	}

	result, err := mp.Execute(context.Background(), commands)

	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestExecuteEtcdDataDirNotExist(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		output:    "",
		shouldErr: false,
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=etcd",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	existCheck := map[string]bool{
		testEtcdDataDir: false,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.Etcd
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return existCheck[path]
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.EtcdComponent(),
		}
	})

	patches.ApplyFunc(mfutil.GenerateManifestYaml, func(components mfutil.Components, bootScope *mfutil.BootScope) error {
		return nil
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestExecuteEtcdDataDirOwnerChangeFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		outputMap: map[string]string{
			"id etcd": "etcd:x:1001:1001:etcd user:/var/lib/etcd:/sbin/nologin",
		},
		shouldErr: true,
		errMsg:    "chown failed",
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=etcd",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.Etcd
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.EtcdComponent(),
		}
	})

	patches.ApplyFunc(mfutil.GenerateManifestYaml, func(components mfutil.Components, bootScope *mfutil.BootScope) error {
		return nil
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "chown failed")
	assert.Nil(t, result)
}

func TestExecuteGenerateManifestFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		output:    "",
		shouldErr: false,
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=kube-apiserver",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.KubeAPIServer
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.APIServerComponent(),
		}
	})

	patches.ApplyFunc(mfutil.GenerateManifestYaml, func(components mfutil.Components, bootScope *mfutil.BootScope) error {
		return errors.New("generate manifest failed")
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "generate manifest failed")
	assert.Nil(t, result)
}

func TestExecuteMkdirFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		output:    "",
		shouldErr: true,
		errMsg:    "mkdir failed",
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=etcd",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.Etcd
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.EtcdComponent(),
		}
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "mkdir failed")
	assert.Nil(t, result)
}

func TestExecuteCreateUserFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		outputMap: map[string]string{
			"id etcd": "",
		},
		shouldErr: true,
		errMsg:    "useradd failed",
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=etcd",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.Etcd
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.EtcdComponent(),
		}
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "useradd failed")
	assert.Nil(t, result)
}

func TestExecuteRestartKubeletWarn(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bootScope := &mfutil.BootScope{}
	executor := &testExecutor{
		outputMap: map[string]string{
			"/usr/bin/sh":            "",
			"/usr/bin/sh -c id etcd": "etcd:x:1001:1001:etcd user:/var/lib/etcd:/sbin/nologin",
			"/usr/bin/sh -c chown -R etcd:etcd /var/lib/etcd":                                                      "",
			"/usr/bin/sh -c if [ -f /usr/lib/systemd/system/kubelet.service ]; then systemctl restart kubelet; fi": "",
		},
		shouldErr:        true,
		errMsg:           "restart kubelet failed",
		errCommandPrefix: "/usr/bin/sh -c if [ -f",
	}
	mp := &ManifestPlugin{
		bootScope: bootScope,
		exec:      executor,
	}

	commands := []string{
		"Manifests",
		"scope=kube-apiserver",
		"manifestDir=" + testManifestDir,
		"etcdDataDir=" + testEtcdDataDir,
	}

	patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
		return item == mfutil.KubeAPIServer
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(mfutil.GetDefaultComponentList, func() mfutil.Components {
		return mfutil.Components{
			mfutil.APIServerComponent(),
		}
	})

	patches.ApplyFunc(mfutil.GenerateManifestYaml, func(components mfutil.Components, bootScope *mfutil.BootScope) error {
		return nil
	})

	result, err := mp.Execute(context.Background(), commands)

	assert.NoError(t, err)
	assert.Nil(t, result)
}
