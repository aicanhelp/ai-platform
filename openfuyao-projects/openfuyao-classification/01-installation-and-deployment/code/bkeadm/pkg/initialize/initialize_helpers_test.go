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
	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"

	"github.com/docker/docker/api/types"

	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	k8sexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/repository"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/timezone"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestExtractVersionFromFilenameExtended(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		expected string
	}{
		{name: "latest tag", filename: "VersionConfig-latest.yaml", expected: "latest"},
		{name: "version with rc suffix", filename: "Core-VersionConfig-v26.03-rc.1.yaml", expected: "v26.03-rc.1"},
		{name: "openfuyao patch version", filename: "openfuyao-v1.0.0.yaml", expected: "v1.0.0"},
		{name: "plain version file", filename: "v2.0.0.yaml", expected: "v2.0.0"},
		{name: "invalid name", filename: "README.md", expected: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, extractVersionFromFilename(tt.filename))
		})
	}
}

func TestVersionLessExtended(t *testing.T) {
	tests := []struct {
		v1, v2   string
		expected bool
	}{
		{"v1.10.0", "v1.2.0", true},
		{"v1.2", "v1.2.0", true},
		{"1.0.0", "v2.0.0", true},
	}
	for _, tt := range tests {
		t.Run(tt.v1+"_vs_"+tt.v2, func(t *testing.T) {
			assert.Equal(t, tt.expected, versionLess(tt.v1, tt.v2))
		})
	}
}

func TestParseHTTPSourceURL(t *testing.T) {
	tests := []struct {
		name       string
		source     string
		wantHost   string
		wantPort   string
		wantPrefix string
	}{
		{
			name:       "host and port",
			source:     "http://mirror.example.com:8080",
			wantHost:   "mirror.example.com",
			wantPort:   "8080",
			wantPrefix: "",
		},
		{
			name:       "default http port",
			source:     "http://mirror.example.com",
			wantHost:   "mirror.example.com",
			wantPort:   "80",
			wantPrefix: "",
		},
		{
			name:       "host port and prefix",
			source:     "http://repo.example.com:40080/openFuyao/binary",
			wantHost:   "repo.example.com",
			wantPort:   "40080",
			wantPrefix: "openFuyao/binary",
		},
		{
			name:       "ip with trailing slash prefix",
			source:     "http://192.168.1.120:8080/packages/runtime/",
			wantHost:   "192.168.1.120",
			wantPort:   "8080",
			wantPrefix: "packages/runtime",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			host, port, prefix := parseHTTPSourceURL(tt.source)
			assert.Equal(t, tt.wantHost, host)
			assert.Equal(t, tt.wantPort, port)
			assert.Equal(t, tt.wantPrefix, prefix)
		})
	}
}

func TestResolveHTTPSourceHost(t *testing.T) {
	t.Run("ip host returns empty domain", func(t *testing.T) {
		domain, ip := resolveHTTPSourceHost("10.0.0.1")
		assert.Empty(t, domain)
		assert.Empty(t, ip)
	})

	t.Run("hostname resolves via LoopIP", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(utils.LoopIP, func(host string) ([]string, error) {
			return []string{"203.0.113.1"}, nil
		})

		orig := global.CustomExtra["httpIp"]
		defer func() { global.CustomExtra["httpIp"] = orig }()

		domain, ip := resolveHTTPSourceHost("repo.example.com")
		assert.Equal(t, "repo.example.com", domain)
		assert.Equal(t, "203.0.113.1", ip)
		assert.Equal(t, "repo.example.com", global.CustomExtra["httpIp"])
	})

	t.Run("LoopIP failure keeps domain", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(utils.LoopIP, func(host string) ([]string, error) {
			return nil, errors.New("lookup failed")
		})

		domain, ip := resolveHTTPSourceHost("unknown.example.com")
		assert.Equal(t, "unknown.example.com", domain)
		assert.Empty(t, ip)
	})
}

func TestApplyDefaultRepoHostOverride(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()

	oc = repository.OtherRepo{
		Repo:   configinit.DefaultImageRepo + ":443/openfuyao/bke-online-installed",
		RepoIP: "203.0.113.10",
	}
	hostIP, domain := applyDefaultRepoHostOverride("192.168.1.1", "deploy.local")
	assert.Equal(t, "203.0.113.10", hostIP)
	assert.Equal(t, configinit.DefaultImageRepo, domain)

	oc = repository.OtherRepo{Repo: "cr.example.com:5000/openfuyao/"}
	hostIP, domain = applyDefaultRepoHostOverride("192.168.1.1", "deploy.local")
	assert.Equal(t, "192.168.1.1", hostIP)
	assert.Equal(t, "deploy.local", domain)
}

func TestResolveOnlineModeHostIP(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()

	op := &Options{HostIP: testHostIP.String(), Domain: "registry.example.com"}

	t.Run("otherRepo is IP", func(t *testing.T) {
		op.OtherRepo = "203.0.113.5:5000/openfuyao/"
		assert.Equal(t, "203.0.113.5", op.resolveOnlineModeHostIP(op.Domain))
	})

	t.Run("domain matches otherRepo host uses repo IP", func(t *testing.T) {
		oc = repository.OtherRepo{RepoIP: "203.0.113.20"}
		op.OtherRepo = "registry.example.com:443/openfuyao/"
		op.Domain = "registry.example.com"
		assert.Equal(t, "203.0.113.20", op.resolveOnlineModeHostIP(op.Domain))
	})

	t.Run("fallback bootstrap IP", func(t *testing.T) {
		oc = repository.OtherRepo{}
		op.OtherRepo = "other.example.com:443/openfuyao/"
		op.Domain = "registry.example.com"
		assert.Equal(t, testHostIP.String(), op.resolveOnlineModeHostIP(op.Domain))
	})
}

func TestFlattenImagesAndExtractVersions(t *testing.T) {
	assert.Nil(t, flattenImages(nil))

	cfg := &build.BuildConfig{
		Repos: []build.Repo{
			{
				SubImages: []build.SubImage{
					{
						Images: []build.Image{
							{Name: "bke-manifests", Tag: []string{"v9.9.9"}},
							{Name: "cluster-api-provider-bke", Tag: []string{"v8.8.8"}},
							{Name: "other", Tag: []string{"v1.0.0"}},
						},
					},
				},
			},
		},
	}
	assert.Len(t, flattenImages(cfg), 3)
	assert.Equal(t, "v9.9.9", findImageTag(cfg, "bke-manifests", "default"))
	assert.Equal(t, "default", findImageTag(cfg, "missing", "default"))

	manifests, provider := extractVersionsFromConfig(cfg, "fallback")
	assert.Equal(t, "v9.9.9", manifests)
	assert.Equal(t, "v8.8.8", provider)
}

func TestGetClusterAPIVersion(t *testing.T) {
	patchYAML := `
repos:
  - subImages:
      - images:
          - name: bke-manifests
            tag: ["v1.1.1"]
          - name: cluster-api-provider-bke
            tag: ["v2.2.2"]
`
	op := &Options{
		OFVersion:  "v1.0.0",
		ClusterAPI: "default-api",
		K8sClient: &k8sexec.MockK8sClient{
			ConfigMap: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "cm.v1.0.0"},
				Data:       map[string]string{"v1.0.0": patchYAML},
			},
		},
	}
	manifests, provider := op.getClusterAPIVersion("v1.0.0", "default-api")
	assert.Equal(t, "v1.1.1", manifests)
	assert.Equal(t, "v2.2.2", provider)

	op.K8sClient = &k8sexec.MockK8sClient{GetErr: errors.New("not found")}
	m, p := op.getClusterAPIVersion("v1.0.0", "default-api")
	assert.Equal(t, "default-api", m)
	assert.Equal(t, "default-api", p)
}

func TestOfflineGenerateDeployCM(t *testing.T) {
	patchesDir := filepath.Join(t.TempDir(), "patches")
	const patchFileName = "openfuyao-v1.0.0.yaml"

	require.NoError(t, os.MkdirAll(patchesDir, utils.DefaultDirPermission))
	require.NoError(t, os.WriteFile(
		filepath.Join(patchesDir, patchFileName),
		[]byte("patch"),
		utils.DefaultFilePermission,
	))

	fs := afero.NewOsFs()
	entries, err := afero.ReadDir(fs, patchesDir)
	require.NoError(t, err)
	require.Len(t, entries, 1)
	require.Equal(t, patchFileName, entries[0].Name())
	require.Equal(t, "v1.0.0", extractVersionFromFilename(entries[0].Name()))

	var capturedVersion, capturedPath, capturedKey string
	op := &Options{
		FS:        fs,
		OFVersion: "v1.0.0",
		SetPatchConfigFn: func(version, path, key string) error {
			capturedVersion = version
			capturedPath = path
			capturedKey = key
			return nil
		},
	}
	err = op.offlineGenerateDeployCM(patchesDir)
	require.NoError(t, err)
	assert.Equal(t, "v1.0.0", capturedVersion)
	assert.Contains(t, capturedPath, patchFileName)
	assert.Contains(t, capturedKey, "v1.0.0")

	op.OFVersion = "v9.9.9"
	err = op.offlineGenerateDeployCM(patchesDir)
	assert.Error(t, err)
}

func TestProcessIndexYAML(t *testing.T) {
	indexYAML := `
- openFuyaoVersion: v1.0.0
  filePath: ./patches/v1.0.0.yaml
`
	var setCalled bool
	op := &Options{OFVersion: "v1.0.0"}
	err := op.processIndexYAML(
		func(url, dest string) error {
			assert.Contains(t, url, "v1.0.0.yaml")
			return nil
		},
		func(version, path, key string) error {
			setCalled = true
			assert.Equal(t, "v1.0.0", version)
			return nil
		},
		[]byte(indexYAML),
		"https://repo.example.com",
		"/tmp/patches",
	)
	require.NoError(t, err)
	assert.True(t, setCalled)
}

func TestParseYAMLBytesToSliceMapTrimsFilePath(t *testing.T) {
	data := `
- openFuyaoVersion: v1.0.0
  filePath: ./files/patch.yaml
`
	result, err := parseYAMLBytesToSliceMap([]byte(data))
	require.NoError(t, err)
	require.Len(t, result, 1)
	assert.Equal(t, "files/patch.yaml", result[0]["filePath"])
}

func TestFilterContainerPorts(t *testing.T) {
	op := &Options{
		KubernetesPort: "36443",
		ConsolePort:    "30010",
		ImageRepoPort:  "443",
		ChartRepoPort:  "8443",
		YumRepoPort:    "8080",
	}
	ports := []string{"36443", "30010", "443", "8443", "8080"}

	t.Run("docker filters existing containers", func(t *testing.T) {
		originalDocker := global.Docker
		global.Docker = &docker.Client{}
		defer func() { global.Docker = originalDocker }()

		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
		patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return false })
		patches.ApplyFunc((*docker.Client).ContainerExists, func(_ *docker.Client, name string) (types.ContainerJSON, bool) {
			if name == utils.LocalKubernetesName || name == utils.LocalImageRegistryName {
				return types.ContainerJSON{}, true
			}
			return types.ContainerJSON{}, false
		})

		filtered := op.filterExistingContainerPorts(append([]string{}, ports...))
		assert.NotContains(t, filtered, "36443")
		assert.NotContains(t, filtered, "443")
	})

	t.Run("containerd filters k8s and console ports", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })
		patches.ApplyFunc(infrastructure.IsContainerd, func() bool { return true })
		patches.ApplyFunc(econd.ContainerExists, func(name string) (econd.NerdContainerInfo, bool) {
			if name == utils.LocalKubernetesName {
				return econd.NerdContainerInfo{}, true
			}
			return econd.NerdContainerInfo{}, false
		})

		filtered := op.filterExistingContainerPorts(append([]string{}, ports...))
		assert.NotContains(t, filtered, "36443")
		assert.NotContains(t, filtered, "30010")
	})
}

func TestPrepareImageRepoConfigBranches(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()

	op := &Options{Domain: "local.domain", HostIP: testHostIP.String(), ImageRepoPort: "5000"}
	repo := op.prepareImageRepoConfig()
	assert.Equal(t, "local.domain", repo.Domain)
	assert.Equal(t, bkecommon.ImageRegistryKubernetes, repo.Prefix)

	oc = repository.OtherRepo{
		Repo:   "cr.example.com:8443/openfuyao/k8s",
		RepoIP: "203.0.113.1",
	}
	repo = op.prepareImageRepoConfig()
	assert.Equal(t, "cr.example.com", repo.Domain)
	assert.Equal(t, "8443", repo.Port)
	assert.Equal(t, "openfuyao/k8s", repo.Prefix)

	oc = repository.OtherRepo{Image: "online-image.tar"}
	repo = op.prepareImageRepoConfig()
	assert.Equal(t, "default", repo.Domain)
	assert.Empty(t, repo.Prefix)
}

func TestEnsureConsoleAllDisabled(t *testing.T) {
	op := &Options{InstallConsole: false}
	err := op.ensureConsoleAll()
	assert.NoError(t, err)
}

func TestGetStartKubernetesVersion(t *testing.T) {
	t.Run("missing registry dir uses manifest version", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(utils.Exists, func(path string) bool { return false })

		op := &Options{}
		version, err := op.getStartKubernetesVersion()
		require.NoError(t, err)
		assert.NotEmpty(t, version)
	})

	t.Run("reads kubectl versions from dir", func(t *testing.T) {
		dir := t.TempDir()
		registryDir := filepath.Join(dir, "mount", "source_registry", "files")
		require.NoError(t, os.MkdirAll(registryDir, utils.DefaultDirPermission))
		require.NoError(t, os.WriteFile(filepath.Join(registryDir, "kubectl-v1.28.3-amd64"), []byte("x"), utils.DefaultFilePermission))
		require.NoError(t, os.WriteFile(filepath.Join(registryDir, "kubectl-v1.30.1-amd64"), []byte("x"), utils.DefaultFilePermission))

		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(utils.Exists, func(path string) bool {
			return path == filepath.Join(dir, "mount", "source_registry", "files")
		})
		origWorkspace := global.Workspace
		global.Workspace = dir
		defer func() { global.Workspace = origWorkspace }()

		op := &Options{}
		version, err := op.getStartKubernetesVersion()
		require.NoError(t, err)
		assert.Equal(t, "v1.28.3", version)
	})
}

func TestValidateDiskSpace(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(string) bool { return false })
	patches.ApplyFunc(utils.DiskUsage, func(string) (uint64, uint64) {
		return 0, utils.MinDiskSpace * 1024 * 1024 * 1024
	})

	op := &Options{}
	err := op.validateDiskSpace()
	assert.NoError(t, err)

	patches.ApplyFunc(utils.DiskUsage, func(string) (uint64, uint64) {
		return 0, 1 * 1024 * 1024 * 1024
	})
	err = op.validateDiskSpace()
	assert.Error(t, err)
}

func TestConfigurePrivateRegistry(t *testing.T) {
	orig := oc
	defer func() { oc = orig }()

	oc = repository.OtherRepo{Repo: "cr.example.com:443/openfuyao/"}
	op := &Options{ImageRepoTLSVerify: true, ImageRepoCAFile: "/tmp/ca.crt"}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(repository.SetupCACertificate, func(*repository.CertificateConfig) error {
		return nil
	})

	err := op.configurePrivateRegistry(op.buildClientAuthConfig())
	assert.NoError(t, err)

	oc = repository.OtherRepo{Repo: "invalid-repo-without-port"}
	err = op.configurePrivateRegistry(op.buildClientAuthConfig())
	assert.NoError(t, err)
}

func TestLogAuthModeTLSOnly(t *testing.T) {
	op := &Options{ImageRepoTLSVerify: true}
	op.logAuthMode()
}

func TestSetTimezoneSkipsNTPWhenDisabled(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	calledNTPServer := false
	patches.ApplyFunc(timezone.SetTimeZone, func() error { return nil })
	patches.ApplyFunc(timezone.NTPServer, func(string, string, bool) (string, error) {
		calledNTPServer = true
		return "ntp.example.com", nil
	})

	op := &Options{EnableNTP: false, NtpServer: "ntp.example.com"}
	err := op.setTimezone()
	require.NoError(t, err)
	assert.False(t, calledNTPServer)
	assert.Empty(t, op.NtpServer)
}
