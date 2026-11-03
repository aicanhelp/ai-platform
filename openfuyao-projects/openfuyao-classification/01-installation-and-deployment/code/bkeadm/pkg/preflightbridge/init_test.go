/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package preflightbridge

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestInitFlagSpecsComplete(t *testing.T) {
	specs := InitFlagSpecs()
	// The spec must include the key flags so bke init and bke preflight init
	// cannot drift.
	names := make(map[string]bool)
	for _, s := range specs {
		names[s.Name] = true
	}
	for _, want := range []string{"file", "hostIP", "kubernetesPort", "consolePort",
		"imageRepoPort", "yumRepoPort", "chartRepoPort", "ntpServer", "runtime",
		"installConsole", "enableNTP", "confirm", "agentHealthPort"} {
		assert.True(t, names[want], "flag %q must be in InitFlagSpecs", want)
	}
}

func TestResolveInitOptionsDefaults(t *testing.T) {
	// Minimal input: only hostIP (required). Everything else defaults.
	o, err := ResolveInitOptions(map[string]string{"hostIP": "192.0.2.10"})
	require.NoError(t, err)
	assert.Equal(t, "192.0.2.10", o.HostIP)
	assert.Equal(t, configinit.DefaultImageRepo, o.Domain)
	assert.Equal(t, atoi(utils.DefaultKubernetesPort), o.KubernetesPort)
	assert.Equal(t, atoi(utils.DefaultConsolePort), o.ConsolePort)
	assert.Equal(t, "containerd", o.Runtime)
	assert.True(t, o.InstallConsole)
	assert.True(t, o.EnableNTP)
}

func TestResolveInitOptionsExplicitWins(t *testing.T) {
	o, err := ResolveInitOptions(map[string]string{
		"hostIP": "192.0.2.10", "kubernetesPort": "36444", "runtime": "docker",
	})
	require.NoError(t, err)
	assert.Equal(t, 36444, o.KubernetesPort)
	assert.Equal(t, "docker", o.Runtime)
	// docker runtime -> docker data root default.
	assert.Equal(t, configinit.DefaultCRIDockerDataRootDir, o.RuntimeStorage)
}

func TestMarkExplicitInitPortPropagatesOnlyAdapterMetadata(t *testing.T) {
	options, err := ResolveInitOptions(map[string]string{"hostIP": "192.0.2.10"})
	require.NoError(t, err)
	MarkExplicitInitPort(&options, "kubernetesPort")
	MarkExplicitInitPort(&options, "unknown")
	input := options.ToAdapterInput()
	assert.Equal(t, map[string]bool{"kubernetes": true}, input.Init.ExplicitPorts)
	input.Init.ExplicitPorts["kubernetes"] = false
	assert.True(t, options.ExplicitPorts["kubernetes"], "adapter input must not alias resolver metadata")
}

func TestResolveInitOptionsRuntimeStorageDefault(t *testing.T) {
	o, err := ResolveInitOptions(map[string]string{"hostIP": "1.2.3.4"})
	require.NoError(t, err)
	assert.Equal(t, configinit.DefaultCRIContainerdDataRootDir, o.RuntimeStorage)
}

func TestValidateInitInputMissingHostIP(t *testing.T) {
	o := InitOptions{}
	err := ValidateInitInput(o)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "host IP")
}

func TestValidateInitInputBadPort(t *testing.T) {
	o := InitOptions{HostIP: "1.2.3.4", KubernetesPort: 70000, ConsolePort: 30010,
		ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
		AgentHealthPort: 58080, Domain: "x"}
	err := ValidateInitInput(o)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "kubernetes port")
}

func TestValidateInitInputImageFileMutualExclusion(t *testing.T) {
	o := InitOptions{HostIP: "1.2.3.4", KubernetesPort: 36443, ConsolePort: 30010,
		ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
		AgentHealthPort: 58080, Domain: "x", ImageFilePath: "/img.tar.gz",
		InstallConsole: true, Runtime: "containerd", RuntimeStorage: "/var/lib/containerd"}
	err := ValidateInitInput(o)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mutually exclusive")
}

func TestValidateInitInputOK(t *testing.T) {
	o := InitOptions{HostIP: "1.2.3.4", KubernetesPort: 36443, ConsolePort: 30010,
		ImageRepoPort: 40443, ChartRepoPort: 38080, YumRepoPort: 40080,
		AgentHealthPort: 58080, Domain: "x", Runtime: "containerd", RuntimeStorage: "/var/lib/containerd"}
	assert.NoError(t, ValidateInitInput(o))
}

func TestToAdapterInputStripsExecutionOnly(t *testing.T) {
	o := InitOptions{HostIP: "1.2.3.4", KubernetesPort: 36443, Domain: "x",
		ImageRepoUsername: "u", ImageRepoPassword: "p"}
	in := o.ToAdapterInput()
	// Auth is reported as a boolean, credentials never carried.
	assert.True(t, in.Init.ImageRepoAuth)
	assert.Empty(t, in.Init.ImageRepoCAFile) // no CA file set
	// No password field on the DTO at all.
	// confirm is execution-only and absent.
}

func TestConfirmIsExecutionOnly(t *testing.T) {
	for _, s := range InitFlagSpecs() {
		if s.Name == "confirm" {
			assert.True(t, s.ExecutionOnly)
			return
		}
	}
	t.Fatal("confirm flag not found")
}

func TestApplyInitFileOptionsProjectsOperationSources(t *testing.T) {
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	cluster.Spec.ClusterConfig.Cluster.ImageRepo = configv1beta1.Repo{Ip: "192.0.2.20", Port: "5443", Prefix: "images"}
	cluster.Spec.ClusterConfig.Cluster.HTTPRepo = configv1beta1.Repo{Domain: "rpm.example.test", Port: "8080"}
	cluster.Spec.ClusterConfig.Cluster.ChartRepo = configv1beta1.Repo{Domain: "chart.example.test", Port: "9443", Prefix: "charts"}
	cluster.Spec.ClusterConfig.Cluster.NTPServer = "ntp.example.test"
	b, err := yaml2.Marshal(cluster)
	require.NoError(t, err)
	file := filepath.Join(t.TempDir(), "cluster.yaml")
	require.NoError(t, os.WriteFile(file, b, 0o600))

	options, err := ResolveInitOptions(map[string]string{"file": file, "hostIP": "192.0.2.10"})
	require.NoError(t, err)
	require.NoError(t, ApplyInitFileOptions(&options))
	assert.Equal(t, "192.0.2.20:5443/images", options.OtherRepo)
	assert.Equal(t, "http://rpm.example.test:8080", options.OtherSource)
	assert.Equal(t, "chart.example.test:9443/charts", options.OtherChart)
	assert.Equal(t, 8080, options.YumRepoPort)
	assert.True(t, options.ExplicitPorts["yum"])
	assert.Equal(t, "ntp.example.test", options.NtpServer)
}

// TestApplyInitFileOptionsDomainAndKeepCLIValues freezes the file-override
// precedence: domain-based repos override, hostIP-equal repos keep CLI values,
// and chart repo IP always projects.
func TestApplyInitFileOptionsDomainAndKeepCLIValues(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "cluster.yaml")
	cluster := &configv1beta1.BKECluster{}
	cluster.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	// Domain-based image repo overrides an empty CLI value.
	cluster.Spec.ClusterConfig.Cluster.ImageRepo = configv1beta1.Repo{
		Domain: "registry.example.com", Port: "443", Prefix: "openfuyao",
	}
	// HTTP repo with domain overrides the package source.
	cluster.Spec.ClusterConfig.Cluster.HTTPRepo = configv1beta1.Repo{
		Domain: "repo.example.com", Port: "40080",
	}
	// Chart repo with IP projects.
	cluster.Spec.ClusterConfig.Cluster.ChartRepo = configv1beta1.Repo{
		Ip: "192.0.2.30", Port: "443", Prefix: "chart",
	}
	raw, err := yaml2.Marshal(cluster)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(file, raw, 0o600))

	options := InitOptions{HostIP: "192.0.2.10", File: file}
	require.NoError(t, ApplyInitFileOptions(&options))
	assert.Equal(t, "registry.example.com:443/openfuyao", options.OtherRepo)
	assert.Equal(t, "http://repo.example.com:40080", options.OtherSource)
	assert.Equal(t, "192.0.2.30:443/chart", options.OtherChart)

	// A repo IP equal to hostIP keeps the CLI value instead of overriding.
	keepCLI := &configv1beta1.BKECluster{}
	keepCLI.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	keepCLI.Spec.ClusterConfig.Cluster.ImageRepo = configv1beta1.Repo{Ip: "192.0.2.10", Port: "5443"}
	rawKeep, err := yaml2.Marshal(keepCLI)
	require.NoError(t, err)
	fileKeep := filepath.Join(dir, "keep.yaml")
	require.NoError(t, os.WriteFile(fileKeep, rawKeep, 0o600))
	keepOptions := InitOptions{HostIP: "192.0.2.10", OtherRepo: "cli-value:5000", File: fileKeep}
	require.NoError(t, ApplyInitFileOptions(&keepOptions))
	assert.Equal(t, "cli-value:5000", keepOptions.OtherRepo)

	// A nil options value and a missing file are both no-ops.
	require.NoError(t, ApplyInitFileOptions(nil))
	require.NoError(t, ApplyInitFileOptions(&InitOptions{}))

	// An invalid HTTP repo port is a structured error.
	badPort := &configv1beta1.BKECluster{}
	badPort.Spec.ClusterConfig = &configv1beta1.BKEConfig{}
	badPort.Spec.ClusterConfig.Cluster.HTTPRepo = configv1beta1.Repo{Domain: "x", Port: "abc"}
	rawBad, err := yaml2.Marshal(badPort)
	require.NoError(t, err)
	fileBad := filepath.Join(dir, "bad.yaml")
	require.NoError(t, os.WriteFile(fileBad, rawBad, 0o600))
	badOptions := InitOptions{File: fileBad}
	assert.ErrorContains(t, ApplyInitFileOptions(&badOptions), "integer")
}

// TestRepoEndpointDefaults freezes the endpoint normalization helper.
func TestRepoEndpointDefaults(t *testing.T) {
	assert.Equal(t, "host:443", repoEndpoint("host", "", "443", "", ""))
	assert.Equal(t, "https://host:8443/prefix", repoEndpoint("host", "8443", "443", "prefix", "https"))
	assert.Equal(t, "http://[2001:db8::1]:80", repoEndpoint("[2001:db8::1]", "", "80", "", "http"))
}
