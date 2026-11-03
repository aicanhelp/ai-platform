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

package kubelet

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/blang/semver/v4"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

func SliceRemoveStrings(slice []string, empty string) []string {
	result := []string{}
	for _, s := range slice {
		if s != empty {
			result = append(result, s)
		}
	}
	return result
}

const (
	numZero    = 0
	numOne     = 1
	numTwo     = 2
	numThree   = 3
	numFour    = 4
	numFive    = 5
	numSix     = 6
	numSeven   = 7
	numEight   = 8
	numNine    = 9
	numTen     = 10
	numSixteen = 16

	numOneHundred     = 100
	numOneTwentySeven = 127
)

const (
	numTwoHundred            = 200
	testScriptWaitTimeout    = 100 * time.Millisecond
	testKubeletScriptPath    = "/etc/kubernetes/kubelet.sh"
	testKubernetesDir        = "/etc/kubernetes"
	testContainerKubeletName = "test-kubelet"
	testKubeletRegistryImg   = "registry.io/kubelet:v1.23.17"
	testPauseRegistryImg     = "registry.io/pause:3.9"
	testKubeletHostIP        = "192.168.1.100"
)

func TestNewRunKubeletCommand(t *testing.T) {
	cmd := NewRunKubeletCommand()

	assert.NotNil(t, cmd)
	assert.Equal(t, "containerd", cmd.ContainerRuntime)
	assert.Equal(t, "kubelet", cmd.ContainerName)
	assert.Equal(t, "deploy.bocloud.k8s:40443/kubernetes/kubelet:v1.23.17", cmd.KubeletImage)
	assert.Equal(t, "/etc/kubernetes/admin.conf", cmd.KubeConfigPath)
	assert.Empty(t, cmd.ExtraVolumes)
	assert.Empty(t, cmd.ExtraKubeletArgs)
	assert.Equal(t, V12317, cmd.k8sVersion)
}

func TestValidateWithEmptyKubeletImage(t *testing.T) {
	cmd := &RunKubeletCommand{
		KubeletImage: "",
		PauseImage:   "pause:image",
		HostIP:       "192.168.1.1",
		k8sVersion:   V12317,
	}

	err := cmd.validate()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kubelet image is empty")
}

func TestValidateWithVersionLessThanV121(t *testing.T) {
	cmd := &RunKubeletCommand{
		KubeletImage:     "kubelet:image",
		PauseImage:       "pause:image",
		HostIP:           "192.168.1.1",
		k8sVersion:       mustParseSemver("v1.20.0"),
		ContainerRuntime: "containerd",
	}

	err := cmd.validate()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "kubernetes version")
}

func TestValidateBoundaryVersions(t *testing.T) {
	testCases := []struct {
		name    string
		version semver.Version
		valid   bool
	}{
		{name: "v1.20", version: mustParseSemver("v1.20.0"), valid: false},
		{name: "v1.21", version: V121, valid: true},
		{name: "v1.23", version: V12317, valid: true},
		{name: "v1.24", version: V124, valid: true},
		{name: "v1.27", version: V127, valid: true},
		{name: "v1.28", version: mustParseSemver("v1.28.0"), valid: true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := &RunKubeletCommand{
				KubeletImage:     "kubelet:image",
				PauseImage:       "pause:image",
				HostIP:           "192.168.1.1",
				k8sVersion:       tc.version,
				ContainerRuntime: "containerd",
			}

			err := cmd.validate()

			if tc.valid {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
			}
		})
	}
}

func mustParseSemver(v string) semver.Version {
	ver, _ := semver.ParseTolerant(v)
	return ver
}

func TestGetVolumeArgsWithPlatformError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(host.PlatformInformation,
		func() (platform string, family string, version string, err error) {
			return "", "", "", assert.AnError
		})

	cmd := &RunKubeletCommand{
		ContainerRuntime: "containerd",
		rootDirPath:      "/var/lib/kubelet",
	}
	args := cmd.getVolumeArgs()

	assert.NotEmpty(t, args)
	assert.Contains(t, strings.Join(args, " "), "--volume /etc/os-release:/etc/os-release")
}

func TestSetContainerName(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetContainerName("test-kubelet")

	assert.Equal(t, "test-kubelet", result.ContainerName)
	assert.Equal(t, "test-kubelet", cmd.ContainerName)
}

func TestSetContainerNameEmpty(t *testing.T) {
	cmd := &RunKubeletCommand{
		ContainerName: "original",
	}

	result := cmd.SetContainerName("")

	assert.Equal(t, "", result.ContainerName)
	assert.Equal(t, "", cmd.ContainerName)
}

func TestSetContainerNameReturnsSelf(t *testing.T) {
	cmd := &RunKubeletCommand{}

	result := cmd.SetContainerName("kubelet-test")

	assert.NotNil(t, result)
}

func TestSetContainerRuntime(t *testing.T) {
	cmd := NewRunKubeletCommand()

	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{name: "containerd", input: "containerd", expected: "containerd"},
		{name: "docker", input: "docker", expected: "docker"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := cmd.SetContainerRuntime(tc.input)
			assert.Equal(t, tc.expected, result.ContainerRuntime)
		})
	}
}

func TestSetKubeletImage(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetKubeletImage("my-registry.io/kubelet:v1.24.0")

	assert.Equal(t, "my-registry.io/kubelet:v1.24.0", result.KubeletImage)
}

func TestSetKubeletImageEmpty(t *testing.T) {
	cmd := &RunKubeletCommand{
		KubeletImage: "original:image",
	}

	result := cmd.SetKubeletImage("")

	assert.Equal(t, "", result.KubeletImage)
}

func TestSetPauseImage(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetPauseImage("registry.io/pause:3.9")

	assert.Equal(t, "registry.io/pause:3.9", result.PauseImage)
}

func TestSetHostIP(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetHostIP("10.0.0.100")

	assert.Equal(t, "10.0.0.100", result.HostIP)
}

func TestSetHostIPEmpty(t *testing.T) {
	cmd := &RunKubeletCommand{
		HostIP: "192.168.1.1",
	}

	result := cmd.SetHostIP("")

	assert.Equal(t, "", result.HostIP)
}

func TestSetKubeConfigPath(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetKubeConfigPath("/etc/kubernetes/kubeconfig.conf")

	assert.Equal(t, "/etc/kubernetes/kubeconfig.conf", result.KubeConfigPath)
}

func TestSetExtraVolumes(t *testing.T) {
	cmd := NewRunKubeletCommand()
	volumes := []string{"/data:/data", "/config:/config"}

	result := cmd.SetExtraVolumes(volumes)

	assert.Equal(t, volumes, result.ExtraVolumes)
	assert.Len(t, result.ExtraVolumes, numTwo)
}

func TestSetExtraVolumesEmpty(t *testing.T) {
	cmd := &RunKubeletCommand{
		ExtraVolumes: []string{"/old:/old"},
	}

	result := cmd.SetExtraVolumes(nil)

	assert.Nil(t, result.ExtraVolumes)
}

func TestSetExtraKubeletArgs(t *testing.T) {
	cmd := NewRunKubeletCommand()
	args := []string{"--v=3", "--feature-gates=XYZ=true"}

	result := cmd.SetExtraKubeletArgs(args)

	assert.Equal(t, args, result.ExtraKubeletArgs)
	assert.Len(t, result.ExtraKubeletArgs, numTwo)
}

func TestSetExtraKubeletArgsEmpty(t *testing.T) {
	cmd := &RunKubeletCommand{
		ExtraKubeletArgs: []string{"--existing"},
	}

	result := cmd.SetExtraKubeletArgs([]string{})

	assert.Empty(t, result.ExtraKubeletArgs)
}

func TestSetK8sVersion(t *testing.T) {
	cmd := NewRunKubeletCommand()

	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{name: "v1.21", input: "v1.21.0", expected: "1.21.0"},
		{name: "v1.23", input: "v1.23.17", expected: "1.23.17"},
		{name: "v1.24", input: "v1.24.0", expected: "1.24.0"},
		{name: "v1.27", input: "v1.27.0", expected: "1.27.0"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := cmd.SetK8sVersion(tc.input)
			assert.Equal(t, tc.expected, result.k8sVersion.String())
		})
	}
}

func TestSetK8sVersionInvalid(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetK8sVersion("invalid-version")

	assert.Equal(t, "0.0.0", result.k8sVersion.String())
}

func TestSetRootDirPath(t *testing.T) {
	cmd := NewRunKubeletCommand()

	result := cmd.SetRootDirPath("/var/lib/custom-kubelet")

	assert.Equal(t, "/var/lib/custom-kubelet", result.rootDirPath)
}

func TestGetCmd(t *testing.T) {
	testCases := []struct {
		name     string
		runtime  string
		expected string
	}{
		{name: "docker", runtime: "docker", expected: "docker"},
		{name: "containerd", runtime: "containerd", expected: "nerdctl"},
		{name: "unknown defaults to nerdctl", runtime: "unknown", expected: "nerdctl"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := &RunKubeletCommand{
				ContainerRuntime: tc.runtime,
			}
			assert.Equal(t, tc.expected, cmd.getCmd())
		})
	}
}

func TestGetCmdArgs(t *testing.T) {
	t.Run("docker returns correct args", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "docker",
			ContainerName:    "my-kubelet",
		}
		args := cmd.getCmdArgs()

		assert.Contains(t, args, "run")
		assert.Contains(t, args, "--detach")
		assert.Contains(t, args, "--name=my-kubelet")
		assert.Contains(t, args, "--net=host")
		assert.Contains(t, args, "--pid=host")
		assert.Contains(t, args, "--user=root")
		assert.Contains(t, args, "--privileged")
		assert.Contains(t, args, "--restart=always")
		assert.NotContains(t, args, "--namespace=")
	})

	t.Run("containerd returns correct args", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			ContainerName:    "my-kubelet",
		}
		args := cmd.getCmdArgs()

		assert.Contains(t, args, "--namespace=k8s.io")
		assert.Contains(t, args, "--insecure-registry")
		assert.Contains(t, args, "run")
	})
}

func TestGetKubeletArgs(t *testing.T) {
	t.Run("basic args", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			k8sVersion:       V124,
			KubeConfigPath:   "/etc/kubernetes/admin.conf",
			HostIP:           "192.168.1.100",
			rootDirPath:      "/var/lib/kubelet",
			PauseImage:       "pause:3.9",
		}
		args := cmd.getKubeletArgs()

		assert.Equal(t, "kubelet", args[0])
		assert.Contains(t, args, "--v=0")
		foundHostnameOverride := false
		for _, arg := range args {
			if strings.HasPrefix(arg, "--hostname-override=") {
				foundHostnameOverride = true
				break
			}
		}
		assert.True(t, foundHostnameOverride)
	})

	t.Run("containerd v1.26 adds remote runtime args", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			k8sVersion:       mustParseSemver("v1.26.0"),
			KubeConfigPath:   "/etc/kubernetes/admin.conf",
			HostIP:           "192.168.1.100",
			rootDirPath:      "/var/lib/kubelet",
		}
		args := cmd.getKubeletArgs()

		assert.Contains(t, args, "--container-runtime=remote")
	})

	t.Run("docker v1.24 adds cri-dockerd endpoint", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "docker",
			k8sVersion:       V124,
			KubeConfigPath:   "/etc/kubernetes/admin.conf",
			HostIP:           "192.168.1.100",
			rootDirPath:      "/var/lib/kubelet",
		}
		args := cmd.getKubeletArgs()

		assert.Contains(t, args, "--container-runtime-endpoint=unix:///var/run/cri-dockerd.sock")
	})

	t.Run("version < v1.24 adds CNI args", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			k8sVersion:       mustParseSemver("v1.23.0"),
			KubeConfigPath:   "/etc/kubernetes/admin.conf",
			HostIP:           "192.168.1.100",
			rootDirPath:      "/var/lib/kubelet",
		}
		args := cmd.getKubeletArgs()

		assert.Contains(t, args, "--network-plugin=cni")
	})

	t.Run("version < v1.27 adds pause image", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			k8sVersion:       V124,
			KubeConfigPath:   "/etc/kubernetes/admin.conf",
			HostIP:           "192.168.1.100",
			rootDirPath:      "/var/lib/kubelet",
			PauseImage:       "pause:3.9",
		}
		args := cmd.getKubeletArgs()

		assert.Contains(t, args, "--pod-infra-container-image=pause:3.9")
	})

	t.Run("extra args are appended", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			k8sVersion:       V124,
			KubeConfigPath:   "/etc/kubernetes/admin.conf",
			HostIP:           "192.168.1.100",
			rootDirPath:      "/var/lib/kubelet",
			ExtraKubeletArgs: []string{"--test-flag"},
		}
		args := cmd.getKubeletArgs()

		assert.Contains(t, args, "--test-flag")
	})
}

func TestGetVolumeArgs(t *testing.T) {
	t.Run("returns non-empty volume list", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			rootDirPath:      "/var/lib/kubelet",
		}
		args := cmd.getVolumeArgs()

		assert.NotEmpty(t, args)
		for _, arg := range args {
			assert.True(t, strings.HasPrefix(arg, "--volume "))
		}
	})

	t.Run("containerd adds cgroup volume", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			rootDirPath:      "/var/lib/kubelet",
		}
		args := cmd.getVolumeArgs()

		assert.Contains(t, strings.Join(args, " "), "--volume /sys/fs/cgroup:/sys/fs/cgroup")
	})

	t.Run("extra volumes are included", func(t *testing.T) {
		cmd := &RunKubeletCommand{
			ContainerRuntime: "containerd",
			rootDirPath:      "/var/lib/kubelet",
			ExtraVolumes:     []string{"/data:/data"},
		}
		args := cmd.getVolumeArgs()

		assert.Contains(t, strings.Join(args, " "), "--volume /data:/data")
	})
}

func TestCommand(t *testing.T) {
	t.Run("valid config returns command", func(t *testing.T) {
		cmd := NewRunKubeletCommand()
		cmd.SetContainerName("kubelet")
		cmd.SetKubeletImage("kubelet:v1.23")
		cmd.SetPauseImage("pause:v3.9")
		cmd.SetHostIP("192.168.1.1")
		cmd.SetK8sVersion("v1.23.17")

		command, args, err := cmd.Command()

		assert.NoError(t, err)
		assert.NotEmpty(t, command)
		assert.NotEmpty(t, args)
		assert.Equal(t, "nerdctl", command)
	})

	t.Run("invalid config returns error", func(t *testing.T) {
		cmd := NewRunKubeletCommand()
		cmd.SetKubeletImage("")

		_, _, err := cmd.Command()

		assert.Error(t, err)
	})
}

func TestValidateWithUnsupportedRuntime(t *testing.T) {
	cmd := &RunKubeletCommand{
		KubeletImage:     "kubelet:image",
		PauseImage:       "pause:image",
		HostIP:           "192.168.1.1",
		k8sVersion:       V12317,
		ContainerRuntime: "unsupported",
	}

	err := cmd.validate()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not supported")
}

func TestValidateWithValidConfigs(t *testing.T) {
	testCases := []struct {
		name       string
		containerd string
	}{
		{name: "containerd", containerd: "containerd"},
		{name: "docker", containerd: "docker"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := &RunKubeletCommand{
				KubeletImage:     "kubelet:image",
				PauseImage:       "pause:image",
				HostIP:           "192.168.1.1",
				k8sVersion:       V12317,
				ContainerRuntime: tc.containerd,
			}

			err := cmd.validate()

			assert.NoError(t, err)
		})
	}
}

func TestAllDefaultVolumesAreValid(t *testing.T) {
	for i, vol := range defaultKubeletVolumes {
		assert.NotEmpty(t, vol, "defaultKubeletVolumes[%d] should not be empty", i)
		assert.Contains(t, vol, ":", "defaultKubeletVolumes[%d] should contain ':'", i)
	}
}

func TestAllDefaultArgsAreValid(t *testing.T) {
	for i, arg := range defaultKubeletArgs {
		assert.NotEmpty(t, arg, "defaultKubeletArgs[%d] should not be empty", i)
	}
}

func TestExportKubeletScriptRefreshFalseAndFileExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testKubeletScriptPath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	cmd := NewRunKubeletCommand()
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")

	err := cmd.ExportKubeletScript(false)

	assert.NoError(t, err)
}

func TestExportKubeletScriptValidateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testKubeletScriptPath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	cmd := NewRunKubeletCommand()
	cmd.SetKubeletImage("")

	err := cmd.ExportKubeletScript(true)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "validate kubelet command failed")
}

func TestExportKubeletScriptContainerRuntimeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testKubeletScriptPath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := &RunKubeletCommand{
		ContainerRuntime: "unsupported-runtime",
		ContainerName:    testContainerKubeletName,
		KubeletImage:     testKubeletRegistryImg,
		PauseImage:       testPauseRegistryImg,
		HostIP:           testKubeletHostIP,
		k8sVersion:       V12317,
		rootDirPath:      "/var/lib/kubelet",
	}

	err := cmd.ExportKubeletScript(true)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not supported")
}

func TestExportKubeletScriptMkdirAllError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testKubeletScriptPath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return os.ErrPermission
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := NewRunKubeletCommand()
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")

	err := cmd.ExportKubeletScript(true)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "create")
}

func TestExportKubeletScriptOpenFileError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testKubeletScriptPath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return nil, os.ErrPermission
	})

	cmd := NewRunKubeletCommand()
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")

	err := cmd.ExportKubeletScript(true)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "open kubelet script file failed")
}

func TestExportKubeletScriptDockerRuntimeSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tempFile, err := ioutil.TempFile("", "kubelet-test-*.sh")
	assert.NoError(t, err)
	tempFilePath := tempFile.Name()
	tempFile.Close()

	defer os.Remove(tempFilePath)

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return tempFilePath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := NewRunKubeletCommand()
	cmd.ContainerRuntime = "docker"
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.24.0")

	err = cmd.ExportKubeletScript(true)

	assert.NoError(t, err)
}

func TestExportKubeletScriptContainerdRuntimeSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tempFile, err := ioutil.TempFile("", "kubelet-test-*.sh")
	assert.NoError(t, err)
	tempFilePath := tempFile.Name()
	tempFile.Close()

	defer os.Remove(tempFilePath)

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return tempFilePath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := NewRunKubeletCommand()
	cmd.ContainerRuntime = "containerd"
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")

	err = cmd.ExportKubeletScript(true)

	assert.NoError(t, err)
}

func TestExportKubeletScriptRefreshTrueForcesExport(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tempFile, err := ioutil.TempFile("", "kubelet-test-*.sh")
	assert.NoError(t, err)
	tempFilePath := tempFile.Name()
	tempFile.Close()

	defer os.Remove(tempFilePath)

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return tempFilePath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := NewRunKubeletCommand()
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")

	err = cmd.ExportKubeletScript(true)

	assert.NoError(t, err)
}

func TestExportKubeletScriptWithExtraVolumes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	tempFile, err := ioutil.TempFile("", "kubelet-test-*.sh")
	assert.NoError(t, err)
	tempFilePath := tempFile.Name()
	tempFile.Close()

	defer os.Remove(tempFilePath)

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return tempFilePath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := NewRunKubeletCommand()
	cmd.ContainerRuntime = "containerd"
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")
	cmd.SetExtraVolumes([]string{"/data:/data", "/config:/config"})
	cmd.SetExtraKubeletArgs([]string{"--test-flag"})

	err = cmd.ExportKubeletScript(true)

	assert.NoError(t, err)
}

func TestExportKubeletScriptDifferentVersions(t *testing.T) {
	testVersions := []struct {
		version string
	}{
		{"v1.21.0"},
		{"v1.23.17"},
		{"v1.24.0"},
		{"v1.27.0"},
	}

	for _, tc := range testVersions {
		t.Run(tc.version, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			tempFile, err := ioutil.TempFile("", "kubelet-test-*.sh")
			assert.NoError(t, err)
			tempFilePath := tempFile.Name()
			tempFile.Close()

			defer os.Remove(tempFilePath)

			patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
				return tempFilePath
			})

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return false
			})

			patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
				return nil
			})

			patches.ApplyFunc(utils.HostName, func() string {
				return "test-node"
			})

			patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
				result := []string{}
				for _, s := range slice {
					if s != empty {
						result = append(result, s)
					}
				}
				return result
			})

			cmd := NewRunKubeletCommand()
			cmd.ContainerRuntime = "containerd"
			cmd.SetContainerName(testContainerKubeletName)
			cmd.SetKubeletImage(testKubeletRegistryImg)
			cmd.SetPauseImage(testPauseRegistryImg)
			cmd.SetHostIP(testKubeletHostIP)
			cmd.SetK8sVersion(tc.version)

			err = cmd.ExportKubeletScript(true)

			assert.NoError(t, err)
		})
	}
}

func TestExportKubeletScriptGeneratesBothCommands(t *testing.T) {
	t.Skip("skip unstable UT: writes kubelet script under /etc/kubernetes and requires host permissions")

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(utils.GetKubeletScriptPath, func() string {
		return testKubeletScriptPath
	})

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return nil
	})

	patches.ApplyFunc(utils.HostName, func() string {
		return "test-node"
	})

	patches.ApplyFunc(SliceRemoveStrings, func(slice []string, empty string) []string {
		result := []string{}
		for _, s := range slice {
			if s != empty {
				result = append(result, s)
			}
		}
		return result
	})

	cmd := NewRunKubeletCommand()
	cmd.ContainerRuntime = "containerd"
	cmd.SetContainerName(testContainerKubeletName)
	cmd.SetKubeletImage(testKubeletRegistryImg)
	cmd.SetPauseImage(testPauseRegistryImg)
	cmd.SetHostIP(testKubeletHostIP)
	cmd.SetK8sVersion("v1.23.17")

	err := cmd.ExportKubeletScript(true)

	assert.NoError(t, err)
}
