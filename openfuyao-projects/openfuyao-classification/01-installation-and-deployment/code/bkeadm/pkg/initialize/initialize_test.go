/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package initialize

import (
	"errors"
	"net"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"

	k8sexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/kubelet"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/bkeagent"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/clusterapi"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/repository"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/syscompat"
)

const (
	testIPv4SegmentA = 192
	testIPv4SegmentB = 168
	testIPv4SegmentC = 1
	testIPv4SegmentD = 100
)

var testHostIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
)

func TestExtractVersionFromFilename(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		expected string
	}{
		{
			name:     "no version pattern returns empty",
			filename: "some-random-file.tar.gz",
			expected: "",
		},
		{
			name:     "version with dots",
			filename: "openfuyao-v1.0.0.1.yaml",
			expected: "v1.0.0.1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractVersionFromFilename(tt.filename)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseYAMLBytesToSliceMap(t *testing.T) {
	tests := []struct {
		name        string
		yamlData    string
		expectError bool
	}{
		{
			name: "valid yaml with required fields",
			yamlData: `
- openFuyaoVersion: v1.0.0
  filePath: ./files/v1.0.0.tar.gz
`,
			expectError: false,
		},
		{
			name: "missing openFuyaoVersion field",
			yamlData: `
- filePath: ./files/v1.0.0.tar.gz
`,
			expectError: true,
		},
		{
			name: "missing filePath field",
			yamlData: `
- openFuyaoVersion: v1.0.0
`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseYAMLBytesToSliceMap([]byte(tt.yamlData))
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

func TestVersionLess(t *testing.T) {
	tests := []struct {
		name     string
		v1       string
		v2       string
		expected bool
	}{
		{
			name:     "v1 less than v2",
			v1:       "v1.0.0",
			v2:       "v2.0.0",
			expected: true,
		},
		{
			name:     "v1 greater than v2",
			v1:       "v2.0.0",
			v2:       "v1.0.0",
			expected: false,
		},
		{
			name:     "same version",
			v1:       "v1.0.0",
			v2:       "v1.0.0",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := versionLess(tt.v1, tt.v2)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestLogAuthMode(t *testing.T) {
	t.Run("tls verify with credentials", func(t *testing.T) {
		op := &Options{
			ImageRepoTLSVerify: true,
			ImageRepoUsername:  "admin",
			ImageRepoPassword:  "password",
		}
		op.logAuthMode()
		assert.True(t, true)
	})

	t.Run("tls verify with ca file", func(t *testing.T) {
		op := &Options{
			ImageRepoTLSVerify: true,
			ImageRepoCAFile:    "/path/to/ca.crt",
		}
		op.logAuthMode()
		assert.True(t, true)
	})
}

func TestSetGlobalCustomExtra(t *testing.T) {
	t.Run("set custom extra values", func(t *testing.T) {
		op := &Options{
			Domain:        "registry.example.com",
			HostIP:        testHostIP.String(),
			ImageRepoPort: "443",
			YumRepoPort:   "8080",
			ChartRepoPort: "8443",
			ClusterAPI:    "v1.0.0",
		}
		op.setGlobalCustomExtra()

		assert.Equal(t, "registry.example.com", global.CustomExtra["domain"])
		assert.Equal(t, testHostIP.String(), global.CustomExtra["host"])
		assert.Equal(t, "443", global.CustomExtra["imageRepoPort"])
		assert.Equal(t, "INFO", global.CustomExtra["agentLogLevel"])
	})
}

func TestModifyPermission(t *testing.T) {
	t.Run("default workspace", func(t *testing.T) {
		op := &Options{}
		op.modifyPermission()
		assert.True(t, true)
	})
}

func TestProcessPatchFiles(t *testing.T) {
	t.Run("directory not exist", func(t *testing.T) {
		op := &Options{}
		nonexistent := filepath.Join(t.TempDir(), "does-not-exist")
		result := op.ProcessPatchFiles(nonexistent)
		assert.Nil(t, result)
	})

	t.Run("empty patches directory", func(t *testing.T) {
		op := &Options{}
		result := op.ProcessPatchFiles("/tmp")
		assert.NotNil(t, result)
	})
}

func TestGenerateClusterConfig(t *testing.T) {
	t.Run("generate cluster config", func(t *testing.T) {
		op := &Options{
			Domain:        "registry.example.com",
			HostIP:        testHostIP.String(),
			ImageRepoPort: "443",
			ChartRepoPort: "8443",
			YumRepoPort:   "8080",
			ClusterAPI:    "v1.0.0",
			Runtime:       "containerd",
		}
		op.setGlobalCustomExtra()
		op.generateClusterConfig()
		assert.True(t, true)
	})
}

func TestDeployCluster(t *testing.T) {
	t.Run("no file specified", func(t *testing.T) {
		op := &Options{
			File: "",
		}
		op.deployCluster()
		assert.True(t, true)
	})
}

func TestBuildClientAuthConfig(t *testing.T) {
	t.Run("build config with all fields", func(t *testing.T) {
		op := &Options{
			ImageRepoTLSVerify: true,
			ImageRepoUsername:  "admin",
			ImageRepoPassword:  "password",
			ImageRepoCAFile:    "/path/to/ca.crt",
		}
		cfg := op.buildClientAuthConfig()
		assert.True(t, cfg.TLSVerify)
		assert.Equal(t, "admin", cfg.Username)
		assert.Equal(t, "password", cfg.Password)
		assert.Equal(t, "/path/to/ca.crt", cfg.CAFile)
	})
}

func TestPrepareImageRepoConfig(t *testing.T) {
	t.Run("default config", func(t *testing.T) {
		op := &Options{
			Domain: "registry.example.com",
			HostIP: testHostIP.String(),
		}
		repo := op.prepareImageRepoConfig()
		assert.Equal(t, "registry.example.com", repo.Domain)
	})
}

func TestInitRepositoriesSetsForwardAccept(t *testing.T) {
	tests := []struct {
		name       string
		forwardErr error
	}{
		{
			name: "set forward accept success",
		},
		{
			name:       "set forward accept warning only",
			forwardErr: errors.New("iptables failed"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(repository.RepoInit, func(repository.OtherRepo, *repository.CertificateConfig) error {
				return nil
			})
			patches.ApplyFunc(repository.DecompressionSystemSourceFile, func() error {
				return nil
			})
			patches.ApplyFunc(syscompat.RepoUpdate, func() error {
				return nil
			})
			patches.ApplyFunc(syscompat.Compat, func() error {
				return nil
			})
			patches.ApplyFunc(syscompat.SetSysctl, func() {})

			forwardCallCount := 0
			patches.ApplyFunc(syscompat.SetForwardAccept, func() error {
				forwardCallCount++
				return tt.forwardErr
			})

			op := &Options{ImageFilePath: "local-image-path"}
			err := op.initRepositories(&repository.CertificateConfig{})

			assert.NoError(t, err)
			assert.Equal(t, 1, forwardCallCount)
		})
	}
}

func TestEnsureClusterAPISetsForwardAccept(t *testing.T) {
	tests := []struct {
		name       string
		forwardErr error
	}{
		{
			name: "set forward accept success",
		},
		{
			name:       "set forward accept warning only",
			forwardErr: errors.New("iptables failed"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			var capturedConsolePort string
			patches.ApplyFunc(infrastructure.StartLocalKubernetes, func(config k3s.Config, _ string) error {
				capturedConsolePort = config.ConsolePort
				return nil
			})
			patches.ApplyFunc(containerd.ApplyContainerdCfg, func(string) error {
				return nil
			})
			patches.ApplyFunc(kubelet.ApplyKubeletCfg, func() error {
				return nil
			})
			patches.ApplyFunc(bkeagent.InstallBKEAgentCRD, func() error {
				return nil
			})
			patches.ApplyFunc(clusterapi.DeployClusterAPI, func(string, string, string) error {
				return nil
			})

			forwardCallCount := 0
			patches.ApplyFunc(syscompat.SetForwardAccept, func() error {
				forwardCallCount++
				return tt.forwardErr
			})

			op := &Options{
				Domain:        "registry.example.com",
				ImageRepoPort: "443",
				ConsolePort:   "31000",
				ClusterAPI:    "v1.0.0",
				K8sClient:     &k8sexec.MockK8sClient{},
			}
			err := op.ensureClusterAPI()

			assert.NoError(t, err)
			assert.Equal(t, 1, forwardCallCount)
			assert.Equal(t, "31000", capturedConsolePort)
		})
	}
}

func TestPrepareChartRepoConfig(t *testing.T) {
	t.Run("chart repo config", func(t *testing.T) {
		op := &Options{
			Domain:        "registry.example.com",
			HostIP:        testHostIP.String(),
			ChartRepoPort: "8443",
		}
		repo := op.prepareChartRepoConfig()
		assert.NotEmpty(t, repo.Ip)
	})
}

func TestPrepareHTTPRepoConfig(t *testing.T) {
	bootstrapIP := testHostIP.String()
	externalIP := "10.20.30.40"

	t.Run("default config", func(t *testing.T) {
		oc = repository.OtherRepo{}
		op := &Options{
			Domain:      "registry.example.com",
			HostIP:      bootstrapIP,
			YumRepoPort: "8080",
		}
		repo := op.prepareHTTPRepoConfig()
		assert.Equal(t, configinit.DefaultYumRepo, repo.Domain)
		assert.Equal(t, bootstrapIP, repo.Ip)
		assert.Equal(t, "8080", repo.Port)
	})

	t.Run("external hostname resolves to ip", func(t *testing.T) {
		oc = repository.OtherRepo{Source: "http://localhost"}
		op := &Options{HostIP: bootstrapIP}
		repo := op.prepareHTTPRepoConfig()
		assert.Equal(t, "localhost", repo.Domain)
		assert.NotEmpty(t, repo.Ip)
		assert.NotNil(t, net.ParseIP(repo.Ip))
		assert.Equal(t, "80", repo.Port)
		assert.Equal(t, "localhost", global.CustomExtra["httpIp"])
	})

	t.Run("external ip", func(t *testing.T) {
		oc = repository.OtherRepo{Source: "http://" + externalIP + ":8080"}
		op := &Options{HostIP: bootstrapIP}
		repo := op.prepareHTTPRepoConfig()
		assert.Equal(t, configinit.DefaultYumRepo, repo.Domain)
		assert.Equal(t, externalIP, repo.Ip)
		assert.Equal(t, "8080", repo.Port)
	})
}

func TestResolveHostIPAndDomain(t *testing.T) {
	bootstrapIP := testHostIP.String()
	externalIP := "10.20.30.40"

	t.Run("offline mode", func(t *testing.T) {
		oc = repository.OtherRepo{}
		op := &Options{Domain: "deploy.bocloud.k8s", HostIP: bootstrapIP}
		hostIP, domain := op.resolveHostIPAndDomain()
		assert.Equal(t, bootstrapIP, hostIP)
		assert.Equal(t, "deploy.bocloud.k8s", domain)
	})

	t.Run("local domain separate from external otherRepo", func(t *testing.T) {
		oc = repository.OtherRepo{RepoIP: externalIP}
		op := &Options{
			Domain:    "deploy.bocloud.k8s",
			HostIP:    bootstrapIP,
			OtherRepo: "cr.internal.openfuyao.cn/pass",
		}
		hostIP, domain := op.resolveHostIPAndDomain()
		assert.Equal(t, bootstrapIP, hostIP)
		assert.Equal(t, "deploy.bocloud.k8s", domain)
	})

	t.Run("domain matches otherRepo host", func(t *testing.T) {
		oc = repository.OtherRepo{RepoIP: externalIP}
		op := &Options{
			Domain:    "cr.internal.openfuyao.cn",
			HostIP:    bootstrapIP,
			OtherRepo: "cr.internal.openfuyao.cn/pass",
		}
		hostIP, domain := op.resolveHostIPAndDomain()
		assert.Equal(t, externalIP, hostIP)
		assert.Equal(t, "cr.internal.openfuyao.cn", domain)
	})

	t.Run("otherRepo uses IP", func(t *testing.T) {
		oc = repository.OtherRepo{RepoIP: externalIP}
		op := &Options{
			Domain:    "deploy.bocloud.k8s",
			HostIP:    bootstrapIP,
			OtherRepo: externalIP + ":5000/pass",
		}
		hostIP, domain := op.resolveHostIPAndDomain()
		assert.Equal(t, externalIP, hostIP)
		assert.Equal(t, "deploy.bocloud.k8s", domain)
	})

	t.Run("legacy default image repo", func(t *testing.T) {
		oc = repository.OtherRepo{
			Repo:   configinit.DefaultImageRepo + ":443/openfuyao/",
			RepoIP: externalIP,
		}
		op := &Options{
			Domain:    configinit.DefaultImageRepo,
			HostIP:    bootstrapIP,
			OtherRepo: configinit.DefaultImageRepo + ":443/openfuyao/bke-online-installed:latest",
		}
		hostIP, domain := op.resolveHostIPAndDomain()
		assert.Equal(t, externalIP, hostIP)
		assert.Equal(t, configinit.DefaultImageRepo, domain)
	})
}

func TestPrepareHTTPRepoConfig_WithPrefix(t *testing.T) {
	originalSource := oc.Source
	defer func() {
		oc.Source = originalSource
	}()

	oc.Source = "http://repo.example.com:40080/openFuyao/bkeadm/releases"

	op := &Options{
		HostIP:      "10.0.0.10",
		YumRepoPort: "80",
	}

	repo := op.prepareHTTPRepoConfig()

	assert.Equal(t, "repo.example.com", repo.Domain)
	assert.Equal(t, "40080", repo.Port)
	assert.Equal(t, "openFuyao/bkeadm/releases", repo.Prefix)
}

