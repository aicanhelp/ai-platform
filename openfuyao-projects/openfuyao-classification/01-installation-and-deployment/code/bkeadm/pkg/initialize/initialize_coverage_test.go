/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package initialize

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/docker/docker/api/types"
	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configsource "gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	k8sexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/kubelet"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/bkeagent"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/bkeconfig"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/clusterapi"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/repository"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/syscompat"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/timezone"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func patchValidationSuccess(patches *gomonkey.Patches) {
	patches.ApplyFunc(utils.DiskUsage, func(string) (uint64, uint64) {
		return 0, 200 * 1024 * 1024 * 1024
	})
	patches.ApplyFunc(utils.Exists, func(string) bool { return false })
	patches.ApplyFunc(utils.CheckPorts, func([]string) error { return nil })
}

func TestNodeInfo(t *testing.T) {
	op := &Options{InstallConsole: true}
	op.nodeInfo()

	op.InstallConsole = false
	op.nodeInfo()
}

func TestValidateSuccess(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchValidationSuccess(patches)

	op := &Options{
		Domain:         "registry.example.com",
		HostIP:         testHostIP.String(),
		KubernetesPort: "36443",
		ConsolePort:    "30010",
		ImageRepoPort:  "443",
		ChartRepoPort:  "8443",
		YumRepoPort:    "8080",
	}
	err := op.Validate()
	assert.NoError(t, err)
}

func TestValidateDiskSpaceFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.DiskUsage, func(string) (uint64, uint64) { return 0, 0 })
	patches.ApplyFunc(utils.Exists, func(string) bool { return false })

	op := &Options{}
	err := op.Validate()
	assert.Error(t, err)
}

func patchNoContainerRuntime(patches *gomonkey.Patches) {
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
	patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })
}

func TestValidatePortsFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchValidationSuccess(patches)
	patchNoContainerRuntime(patches)
	patches.ApplyFunc(utils.CheckPorts, func([]string) error {
		return errors.New("port in use")
	})

	op := &Options{
		KubernetesPort: "36443",
		ConsolePort:    "30010",
		ImageRepoPort:  "443",
		ChartRepoPort:  "8443",
		YumRepoPort:    "8080",
	}
	err := op.validatePorts()
	assert.Error(t, err)
}

func TestValidatePortsSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchNoContainerRuntime(patches)
	patches.ApplyFunc(utils.CheckPorts, func([]string) error { return nil })

	op := &Options{
		KubernetesPort: "36443",
		ConsolePort:    "30010",
		ImageRepoPort:  "443",
		ChartRepoPort:  "8443",
		YumRepoPort:    "8080",
	}
	err := op.validatePorts()
	assert.NoError(t, err)
}

func TestFilterDockerContainerPortsAllContainers(t *testing.T) {
	op := &Options{
		KubernetesPort: "36443",
		ImageRepoPort:  "443",
		ChartRepoPort:  "8443",
		YumRepoPort:    "8080",
	}
	ports := []string{"36443", "443", "8443", "8080", "9999"}

	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	defer func() { global.Docker = originalDocker }()

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*docker.Client).ContainerExists, func(_ *docker.Client, name string) (types.ContainerJSON, bool) {
		switch name {
		case utils.LocalKubernetesName, utils.LocalImageRegistryName,
			utils.LocalChartRegistryName, utils.LocalYumRegistryName:
			return types.ContainerJSON{}, true
		default:
			return types.ContainerJSON{}, false
		}
	})

	filtered := op.filterDockerContainerPorts(append([]string{}, ports...))
	assert.Equal(t, []string{"9999"}, filtered)
}

func TestConfigLocalSourceOffline(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	var capturedURL string
	patches.ApplyFunc(configsource.SetSource, func(url string) error {
		capturedURL = url
		return nil
	})

	op := &Options{}
	op.configLocalSource()
	assert.Contains(t, capturedURL, "file://")
}

func TestConfigLocalSourceSkipsWhenOnline(t *testing.T) {
	called := false
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(configsource.SetSource, func(string) error {
		called = true
		return nil
	})

	op := &Options{OtherRepo: "cr.example.com:443/repo"}
	op.configLocalSource()
	assert.False(t, called)
}

func TestPrepareEnvironmentSuccess(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syscompat.SetHosts, func(string, string) error { return nil })
	patches.ApplyFunc(repository.RepoInit, func(repository.OtherRepo, *repository.CertificateConfig) error {
		return nil
	})
	patches.ApplyFunc(repository.DecompressionSystemSourceFile, func() error { return nil })
	patches.ApplyFunc(repository.SourceInit, func(repository.OtherRepo) error { return nil })
	patches.ApplyFunc(syscompat.RepoUpdate, func() error { return nil })
	patches.ApplyFunc(syscompat.Compat, func() error { return nil })
	patches.ApplyFunc(syscompat.SetForwardAccept, func() error { return nil })
	patches.ApplyFunc(syscompat.SetSysctl, func() {})

	op := &Options{
		Domain: "registry.example.com",
		HostIP: testHostIP.String(),
	}
	err := op.prepareEnvironment()
	assert.NoError(t, err)
}

func TestEnsureContainerServerSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(repository.PrepareRepositoryDependOn, func(string) error { return nil })
	patches.ApplyFunc(repository.VerifyContainerdFile, func(string) (repository.ContainerdFileResult, error) {
		return repository.ContainerdFileResult{
			FilePath:       "/tmp/runtime",
			ContainerdList: []string{"containerd-amd64.tar.gz"},
			CniPluginList:  []string{"cni-amd64.tar.gz"},
		}, nil
	})
	patches.ApplyFunc(infrastructure.RuntimeInstall, func(infrastructure.RuntimeConfig) error { return nil })

	op := &Options{
		Runtime:        "containerd",
		RuntimeStorage: "/var/lib/containerd",
		Domain:         "registry.example.com",
		ImageRepoPort:  "443",
		HostIP:         testHostIP.String(),
	}
	err := op.ensureContainerServer()
	assert.NoError(t, err)
}

func TestEnsureRepositorySuccess(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{Repo: "cr.example.com/repo"}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(repository.LoadLocalRepository, func() error { return nil })
	patches.ApplyFunc(repository.ContainerServer, func(string, string, string, string) error { return nil })
	patches.ApplyFunc(repository.YumServer, func(string, string, string, string, string) error { return nil })
	patches.ApplyFunc(repository.ChartServer, func(string, string, string, string, string) error { return nil })

	op := &Options{ImageRepoPort: "443", YumRepoPort: "8080", ChartRepoPort: "8443"}
	err := op.ensureRepository()
	assert.NoError(t, err)
}

func TestEnsureRepositoryWithLocalImage(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(repository.LoadLocalImage, func() error { return nil })
	patches.ApplyFunc(repository.LoadLocalRepository, func() error { return nil })
	patches.ApplyFunc(repository.ContainerServer, func(string, string, string, string) error { return nil })
	patches.ApplyFunc(repository.SyncLocalImage, func(string) error { return nil })
	patches.ApplyFunc(repository.YumServer, func(string, string, string, string, string) error { return nil })
	patches.ApplyFunc(repository.ChartServer, func(string, string, string, string, string) error { return nil })

	op := &Options{
		ImageFilePath: "/data/images",
		ImageRepoPort: "443",
		YumRepoPort:   "8080",
		ChartRepoPort: "8443",
	}
	err := op.ensureRepository()
	assert.NoError(t, err)
}

func TestGenerateDeployCMOfflineWithImageFilePath(t *testing.T) {
	oldWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	defer func() { global.Workspace = oldWorkspace }()

	patchesDir := filepath.Join(global.Workspace, utils.LocalPatchDirectory)
	require.NoError(t, os.MkdirAll(patchesDir, utils.DefaultDirPermission))
	require.NoError(t, os.WriteFile(
		filepath.Join(patchesDir, "openfuyao-v1.0.0.yaml"),
		[]byte("patch"),
		utils.DefaultFilePermission,
	))

	op := &Options{
		FS:            afero.NewOsFs(),
		ImageFilePath: "/data/images",
		OFVersion:     "v1.0.0",
		SetPatchConfigFn: func(string, string, string) error {
			return nil
		},
	}
	err := op.generateDeployCM()
	assert.NoError(t, err)
}

func TestGenerateDeployCMOfflineNoRepo(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{}

	patchesDir := filepath.Join(t.TempDir(), "patches")
	require.NoError(t, os.MkdirAll(patchesDir, utils.DefaultDirPermission))
	require.NoError(t, os.WriteFile(
		filepath.Join(patchesDir, "openfuyao-v2.0.0.yaml"),
		[]byte("patch"),
		utils.DefaultFilePermission,
	))

	oldWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	defer func() { global.Workspace = oldWorkspace }()

	targetDir := filepath.Join(global.Workspace, utils.PatchDataDirectory)
	require.NoError(t, os.MkdirAll(targetDir, utils.DefaultDirPermission))
	require.NoError(t, os.WriteFile(
		filepath.Join(targetDir, "openfuyao-v2.0.0.yaml"),
		[]byte("patch"),
		utils.DefaultFilePermission,
	))

	op := &Options{
		FS:        afero.NewOsFs(),
		OFVersion: "v2.0.0",
		SetPatchConfigFn: func(string, string, string) error {
			return nil
		},
	}
	err := op.generateDeployCM()
	assert.NoError(t, err)
}

func TestGenerateDeployCMOnline(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{Repo: "cr.example.com/repo", Image: "online.tar"}

	oldWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	defer func() { global.Workspace = oldWorkspace }()

	indexYAML := `
- openFuyaoVersion: v3.0.0
  filePath: patches/v3.0.0.yaml
`
	fs := afero.NewMemMapFs()
	op := &Options{
		FS:         fs,
		OFVersion:  "v3.0.0",
		VersionUrl: "https://repo.example.com/versions",
		DownloadFunc: func(url, dest string) error {
			if filepath.Base(dest) == "index.yaml" {
				return afero.WriteFile(fs, dest, []byte(indexYAML), utils.DefaultFilePermission)
			}
			return nil
		},
		SetPatchConfigFn: func(string, string, string) error { return nil },
	}
	err := op.generateDeployCM()
	assert.NoError(t, err)
}

func TestOnlineGenerateDeployCMFailure(t *testing.T) {
	oldWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	defer func() { global.Workspace = oldWorkspace }()

	op := &Options{
		FS:         afero.NewMemMapFs(),
		VersionUrl: "https://repo.example.com/versions",
		DownloadFunc: func(string, string) error {
			return errors.New("download failed")
		},
	}
	err := op.onlineGenerateDeployCM()
	assert.Error(t, err)
}

func TestGetClusterAPIVersionMissingDataKey(t *testing.T) {
	op := &Options{
		OFVersion:  "v1.0.0",
		ClusterAPI: "default-api",
		K8sClient: &k8sexec.MockK8sClient{
			ConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "cm.v1.0.0"},
				Data:       map[string]string{"other-key": "data"},
			},
		},
	}
	m, p := op.getClusterAPIVersion("v1.0.0", "default-api")
	assert.Equal(t, "default-api", m)
	assert.Equal(t, "default-api", p)
}

func TestGetClusterAPIVersionBadYAML(t *testing.T) {
	op := &Options{
		OFVersion:  "v1.0.0",
		ClusterAPI: "default-api",
		K8sClient: &k8sexec.MockK8sClient{
			ConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "cm.v1.0.0"},
				Data:       map[string]string{"v1.0.0": "not: valid: yaml: ["},
			},
		},
	}
	m, p := op.getClusterAPIVersion("v1.0.0", "default-api")
	assert.Equal(t, "default-api", m)
	assert.Equal(t, "default-api", p)
}

func TestGetClusterAPIVersionNewK8sClient(t *testing.T) {
	originalK8s := global.K8s
	defer func() { global.K8s = originalK8s }()
	global.K8s = nil

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(k8sexec.NewKubernetesClient, func(string) (k8sexec.KubernetesClient, error) {
		return nil, errors.New("no kubeconfig")
	})

	op := &Options{OFVersion: "v1.0.0", ClusterAPI: "fallback"}
	m, p := op.getClusterAPIVersion("v1.0.0", "fallback")
	assert.Equal(t, "fallback", m)
	assert.Equal(t, "fallback", p)
}

func TestPrepareClusterConfigDataSuccess(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{Source: "http://mirror.example.com:8080"}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(string) bool { return false })

	op := &Options{
		Domain:          "registry.example.com",
		HostIP:          testHostIP.String(),
		ImageRepoPort:   "443",
		ChartRepoPort:   "8443",
		YumRepoPort:     "8080",
		ClusterAPI:      "v1.0.0",
		Runtime:         "containerd",
		NtpServer:       "pool.ntp.org",
		AgentHealthPort: "8081",
	}
	op.setGlobalCustomExtra()

	data, repos, err := op.prepareClusterConfigData()
	require.NoError(t, err)
	require.Len(t, repos, 3)
	assert.Equal(t, "registry.example.com", data["domain"])
	assert.Equal(t, testHostIP.String(), data["host"])
	assert.NotEmpty(t, data["kubernetesVersion"])
	assert.NotEmpty(t, repos[0].Domain)
}

func TestCreateClusterConfigFileWithFileSet(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(bkeconfig.SetKubernetesConfig, func(map[string]string, string, string) error {
		return nil
	})

	op := &Options{File: "/tmp/cluster.yaml"}
	op.createClusterConfigFile(
		map[string]string{"domain": "example.com"},
		v1beta1.Repo{Domain: "img.example.com"},
		v1beta1.Repo{Domain: "yum.example.com"},
		v1beta1.Repo{Domain: "chart.example.com"},
	)
}

func TestProcessPatchFilesWithValidPatch(t *testing.T) {
	patchesDir := t.TempDir()
	require.NoError(t, os.WriteFile(
		filepath.Join(patchesDir, "openfuyao-v1.0.0.yaml"),
		[]byte("patch-content"),
		utils.DefaultFilePermission,
	))

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(bkeconfig.SetPatchConfig, func(string, string, string) error {
		return nil
	})

	op := &Options{}
	result := op.ProcessPatchFiles(patchesDir)
	require.NotNil(t, result)
	assert.NotEmpty(t, result)
}

func TestInitializeValidationFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.DiskUsage, func(string) (uint64, uint64) { return 0, 0 })
	patches.ApplyFunc(utils.Exists, func(string) bool { return false })

	op := &Options{}
	op.Initialize()
}

func TestEnsureClusterAPIWithOFVersion(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{}

	oldWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	defer func() { global.Workspace = oldWorkspace }()

	targetDir := filepath.Join(global.Workspace, utils.PatchDataDirectory)
	require.NoError(t, os.MkdirAll(targetDir, utils.DefaultDirPermission))
	require.NoError(t, os.WriteFile(
		filepath.Join(targetDir, "openfuyao-v1.0.0.yaml"),
		[]byte("patch"),
		utils.DefaultFilePermission,
	))

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.StartLocalKubernetes, func(k3s.Config, string) error { return nil })
	patches.ApplyFunc(syscompat.SetForwardAccept, func() error { return nil })
	patches.ApplyFunc(containerd.ApplyContainerdCfg, func(string) error { return nil })
	patches.ApplyFunc(kubelet.ApplyKubeletCfg, func() error { return nil })
	patches.ApplyFunc(bkeagent.InstallBKEAgentCRD, func() error { return nil })
	patches.ApplyFunc(clusterapi.DeployClusterAPI, func(string, string, string) error { return nil })

	op := &Options{
		Domain:        "registry.example.com",
		ImageRepoPort: "443",
		ConsolePort:   "31000",
		ClusterAPI:    "v1.0.0",
		OFVersion:     "v1.0.0",
		FS:            afero.NewOsFs(),
		SetPatchConfigFn: func(string, string, string) error {
			return nil
		},
		K8sClient: &k8sexec.MockK8sClient{},
	}
	err := op.ensureClusterAPI()
	assert.NoError(t, err)
}

func TestConfigurePrivateRegistryWithCA(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{Repo: "cr.example.com:8443/openfuyao/k8s"}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(repository.SetupCACertificate, func(*repository.CertificateConfig) error {
		return nil
	})

	op := &Options{
		ImageRepoTLSVerify: true,
		ImageRepoCAFile:    "/etc/certs/ca.crt",
	}
	cfg := op.buildClientAuthConfig()
	err := op.configurePrivateRegistry(cfg)
	assert.NoError(t, err)
}

func TestInitializeSuccessMinimal(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()
	oc = repository.OtherRepo{}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patchValidationSuccess(patches)
	patches.ApplyFunc(timezone.SetTimeZone, func() error { return nil })
	patches.ApplyFunc(timezone.NTPServer, func(string, string, bool) (string, error) {
		return "pool.ntp.org", nil
	})
	patches.ApplyFunc(syscompat.SetHosts, func(string, string) error { return nil })
	patches.ApplyFunc(repository.RepoInit, func(repository.OtherRepo, *repository.CertificateConfig) error {
		return nil
	})
	patches.ApplyFunc(repository.DecompressionSystemSourceFile, func() error { return nil })
	patches.ApplyFunc(repository.SourceInit, func(repository.OtherRepo) error { return nil })
	patches.ApplyFunc(syscompat.RepoUpdate, func() error { return nil })
	patches.ApplyFunc(syscompat.Compat, func() error { return nil })
	patches.ApplyFunc(syscompat.SetForwardAccept, func() error { return nil })
	patches.ApplyFunc(syscompat.SetSysctl, func() {})
	patches.ApplyFunc(repository.PrepareRepositoryDependOn, func(string) error { return nil })
	patches.ApplyFunc(repository.VerifyContainerdFile, func(string) (repository.ContainerdFileResult, error) {
		return repository.ContainerdFileResult{
			FilePath:       "/tmp/runtime",
			ContainerdList: []string{"containerd-amd64.tar.gz"},
			CniPluginList:  []string{"cni-amd64.tar.gz"},
		}, nil
	})
	patches.ApplyFunc(infrastructure.RuntimeInstall, func(infrastructure.RuntimeConfig) error { return nil })
	patches.ApplyFunc(repository.LoadLocalRepository, func() error { return nil })
	patches.ApplyFunc(repository.ContainerServer, func(string, string, string, string) error { return nil })
	patches.ApplyFunc(repository.YumServer, func(string, string, string, string, string) error { return nil })
	patches.ApplyFunc(repository.ChartServer, func(string, string, string, string, string) error { return nil })
	patches.ApplyFunc(infrastructure.StartLocalKubernetes, func(k3s.Config, string) error { return nil })
	patches.ApplyFunc(containerd.ApplyContainerdCfg, func(string) error { return nil })
	patches.ApplyFunc(kubelet.ApplyKubeletCfg, func() error { return nil })
	patches.ApplyFunc(bkeagent.InstallBKEAgentCRD, func() error { return nil })
	patches.ApplyFunc(clusterapi.DeployClusterAPI, func(string, string, string) error { return nil })
	patches.ApplyFunc(bkeconfig.SetKubernetesConfig, func(map[string]string, string, string) error {
		return nil
	})
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return path != filepath.Join(global.Workspace, utils.ImageDataDirectory)
	})

	op := &Options{
		Domain:          "registry.example.com",
		HostIP:          testHostIP.String(),
		KubernetesPort:  "36443",
		ConsolePort:     "30010",
		ImageRepoPort:   "443",
		ChartRepoPort:   "8443",
		YumRepoPort:     "8080",
		ClusterAPI:      "v1.0.0",
		Runtime:         "containerd",
		RuntimeStorage:  "/var/lib/containerd",
		InstallConsole:  false,
		AgentHealthPort: "8081",
		K8sClient:         &k8sexec.MockK8sClient{},
	}
	op.Initialize()
}
