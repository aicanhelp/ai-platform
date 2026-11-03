/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package containerd

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"text/template"
	"time"

	"github.com/pkg/errors"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"k8s.io/apimachinery/pkg/util/wait"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/pkg/config"
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed config.toml
	configToml string
	//go:embed containerd_crd.yaml
	containerdCrd []byte
	//go:embed containerd_default.yaml
	containerdDefault       []byte
	defaultRuntime          = "runc"
	defaultInstallDirectory = "/"
	cniDirectory            = "/opt/cni/bin"
)

const (
	// containerdReadyTimeout is the timeout for waiting containerd to be ready
	containerdReadyTimeout = 2 * time.Minute
	// containerdPollInterval is the interval for polling containerd readiness
	containerdPollInterval = 5 * time.Second
)

func applyContainerdCrd() error {
	var err error
	if global.K8s == nil {
		global.K8s, err = k8s.NewKubernetesClient("")
		if err != nil {
			return err
		}
	}

	containerdCrdFile := fmt.Sprintf("%s/tmpl/containerd_crd.yaml", global.Workspace)
	err = os.WriteFile(containerdCrdFile, containerdCrd, utils.DefaultFilePermission)
	if err != nil {
		return err
	}
	time.Sleep(1 * time.Second)

	log.Info("Install Containerd CRD...")
	err = global.K8s.InstallYaml(containerdCrdFile, nil, "")
	if err != nil {
		return err
	}

	return nil
}

func applyContainerdDefault(domain string) error {
	runtimeParam := map[string]string{}
	sandbox, offline := config.GenerateControllerParam(domain)
	runtimeParam["sandbox"] = sandbox
	runtimeParam["offline"] = offline

	tmpl, err := template.New("containerd").Parse(string(containerdDefault))
	if err != nil {
		return fmt.Errorf("parse containerd default failed: %v", err)
	}

	var buf bytes.Buffer
	if err = tmpl.Execute(&buf, runtimeParam); err != nil {
		return fmt.Errorf("render containerd default failed: %v", err)
	}

	conf := &configv1beta1.ContainerdConfig{}
	if err = yaml2.Unmarshal(buf.Bytes(), conf); err != nil {
		return fmt.Errorf("unmarshal containerd default failed: %v", err)
	}

	if err = k8s.CreateNamespace(global.K8s, conf.Namespace); err != nil {
		return err
	}

	containerdDefaultFile := fmt.Sprintf("%s/tmpl/containerd_default.yaml", global.Workspace)
	err = os.WriteFile(containerdDefaultFile, containerdDefault, utils.DefaultFilePermission)
	if err != nil {
		return err
	}
	time.Sleep(1 * time.Second)

	log.Infof("Submit containerd default yaml to the cluster")
	err = global.K8s.InstallYaml(containerdDefaultFile, runtimeParam, "")
	if err != nil {
		return fmt.Errorf("install containerd default: %w", err)
	}
	log.Infof("Submit the containerd configuration to the cluster")

	return nil
}

func ApplyContainerdCfg(domain string) error {
	err := applyContainerdCrd()
	if err != nil {
		return fmt.Errorf("apply containerd crd failed: %v", err)
	}

	err = applyContainerdDefault(domain)
	if err != nil {
		return fmt.Errorf("apply containerd default failed: %v", err)
	}

	log.Info("Apply containerd crd and default success")

	return nil
}

func getPlatform() string {
	switch runtime.GOARCH {
	case "amd64":
		return "linux/amd64"
	case "arm64":
		return "linux/arm64"
	case "arm":
		return "linux/arm/v7"
	default:
		return "linux/amd64"
	}
}

func executeTemplateWithFile(tplContent, tplName string, data interface{}, file *os.File) error {
	// 解析模板
	tmpl, err := template.New(tplName).Parse(tplContent)
	if err != nil {
		return fmt.Errorf("parse template %s failed: %w", tplName, err)
	}

	// 执行模板
	if err := tmpl.Execute(file, data); err != nil {
		return fmt.Errorf("execute template %s failed: %w", tplName, err)
	}

	return nil
}

func Install(domain, port, runtimeStorage, containerdFile, caFile string) error {
	err := utils.UnTar(containerdFile, defaultInstallDirectory)
	if err != nil {
		return fmt.Errorf("untar containerd package: %w", err)
	}
	runtimeParam := map[string]string{
		"runtime":  defaultRuntime,
		"port":     port,
		"repo":     fmt.Sprintf("%s:%s", domain, port),
		"dataRoot": runtimeStorage,
		"caFile":   caFile, // 传入用户指定的CA证书路径
	}
	sandbox, offline := config.GenerateControllerParam(fmt.Sprintf("%s:%s", domain, port))
	runtimeParam["sandbox"] = sandbox
	runtimeParam["offline"] = offline
	runtimeParam["platform"] = getPlatform()
	log.Infof("containerd sandbox image: %s", runtimeParam["sandbox"])

	// Render configuration file
	if err = writeConfigToDisk(runtimeParam); err != nil {
		return fmt.Errorf("write containerd config: %w", err)
	}
	log.Info("containerd configuration written to /etc/containerd/config.toml")
	// 新增：创建 hosts.toml 配置
	if err = createHostsTOML(runtimeParam); err != nil {
		return fmt.Errorf("create containerd hosts.toml: %w", err)
	}
	// enable and start containerd
	if err = global.Command.ExecuteCommand("systemctl", "enable", "containerd"); err != nil {
		return fmt.Errorf("enable containerd service: %w", err)
	}

	// Start the containerd
	if err = global.Command.ExecuteCommand("systemctl", "start", "containerd"); err != nil {
		return fmt.Errorf("start containerd service: %w", err)
	}
	log.Info("wait for containerd to start")
	if err = waitContainerdReady(); err != nil {
		return fmt.Errorf("wait for containerd ready: %w", err)
	}
	log.Info("containerd is ready")
	return nil
}

func waitContainerdReady() error {
	ctx, cancel := context.WithTimeout(context.Background(), containerdReadyTimeout)
	defer cancel()
	err := wait.PollImmediateUntil(containerdPollInterval, func() (bool, error) {
		log.Info("Waiting for containerd to be ready")
		_, err := econd.NewContainedClient()
		if err == nil {
			return true, nil
		}
		log.Debugf("containerd is not available: %v", err)
		return false, nil
	}, ctx.Done())
	if err != nil {
		return errors.Wrapf(err, "failed to wait containerd available")
	}
	return nil
}

func writeConfigToDisk(runtimeParam map[string]string) error {
	// Render configuration file
	f, err := os.OpenFile(
		fmt.Sprintf("%s%s", defaultInstallDirectory, "etc/containerd/config.toml"),
		os.O_WRONLY|os.O_CREATE, utils.DefaultFilePermission)
	if err != nil {
		return err
	}
	defer func() {
		if err := f.Close(); err != nil {
			log.Warnf("Failed to close config.toml: %v", err)
		}
	}()
	tpl, err := template.New("config.toml").Parse(configToml)
	if err != nil {
		log.Warnf("Failed to parse config.toml: %v", err)
	}
	return tpl.Execute(f, runtimeParam)
}

func createOfflineSpecialHostsTOML(certsDir, port string) error {
	offlineRegistry := fmt.Sprintf("127.0.0.1:%s", port)
	registryDir := filepath.Join(certsDir, offlineRegistry)
	hostsTOMLPath := filepath.Join(registryDir, "hosts.toml")

	if err := os.MkdirAll(registryDir, utils.DefaultDirPermission); err != nil {
		return fmt.Errorf("create %s dir failed: %v", offlineRegistry, err)
	}

	// 使用模板渲染 hosts.toml 内容
	data := struct {
		Port string
	}{
		Port: port,
	}

	hostsTpl := `server = "https://127.0.0.1:{{.Port}}"
[host."https://127.0.0.1:{{.Port}}"]
  capabilities = ["pull", "resolve", "push"]
  skip_verify = true
`

	f, err := os.OpenFile(hostsTOMLPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, utils.DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("create %s hosts.toml failed: %v", offlineRegistry, err)
	}
	defer func() {
		if closeErr := f.Close(); closeErr != nil {
			log.Warnf("Failed to close %s hosts.toml: %v", offlineRegistry, closeErr)
		}
	}()

	if err = executeTemplateWithFile(hostsTpl, "offlineSpecialHosts", data, f); err != nil {
		return fmt.Errorf("render %s hosts.toml failed: %w", offlineRegistry, err)
	}

	log.Infof("Created offline special hosts.toml: %s", hostsTOMLPath)
	return nil
}

// getRegistryList 获取需要配置的 registry 列表
func getRegistryList(repo, repoWithNoPort, offline string) []string {
	registries := []string{repo, repoWithNoPort}
	if offline == "true" {
		publicRegistries := []string{
			"docker.io", "registry.k8s.io", "k8s.gcr.io", "ghcr.io", "quay.io", "gcr.io", "cr.openfuyao.cn", "hub.oepkgs.net",
		}
		registries = append(registries, publicRegistries...)
	}
	return registries
}

// createRegistryHostsTOML 为单个 registry 创建 hosts.toml 文件
func createRegistryHostsTOML(registry, repo, offline, caFile, certsDir string) error {
	registryDir := filepath.Join(certsDir, registry)
	if err := os.MkdirAll(registryDir, utils.DefaultDirPermission); err != nil {
		return fmt.Errorf("create %s dir failed: %v", registry, err)
	}

	data := struct {
		Repo     string
		Registry string
		Offline  string
		CAFile   string
	}{
		Repo:     repo,
		Registry: registry,
		Offline:  offline,
		CAFile:   caFile,
	}

	hostsTpl := `# 私有镜像仓配置：{{.Registry}}
server = "https://{{.Registry}}"  # 直接指向私有仓库的域名和端口
[host."https://{{.Repo}}"]
  capabilities = ["pull", "resolve", "push"]  # 支持拉取、解析、推送
  {{if .CAFile}}ca = "/etc/containerd/certs.d/{{.Registry}}/ca.crt"{{end}}  # 若有CA证书，指定证书路径
  skip_verify = {{if .CAFile}}false{{else}}true{{end}}  # 有CA证书则不跳过验证，否则跳过
`

	hostsPath := filepath.Join(registryDir, "hosts.toml")
	f, err := os.OpenFile(hostsPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, utils.DefaultFilePermission)
	if err != nil {
		return fmt.Errorf("create %s hosts.toml failed: %v", registry, err)
	}
	defer f.Close()
	if err = executeTemplateWithFile(hostsTpl, "baseHosts", data, f); err != nil {
		return fmt.Errorf("process template for %s failed: %w", registry, err)
	}
	return nil
}

func createHostsTOML(runtimeParam map[string]string) error {
	repo := runtimeParam["repo"]
	repoWithNoPort := strings.Split(repo, ":")[0]
	offline := runtimeParam["offline"]
	certsDir := "/etc/containerd/certs.d"
	caFile := runtimeParam["caFile"]
	port := runtimeParam["port"]

	if err := createOfflineSpecialHostsTOML(certsDir, port); err != nil {
		return fmt.Errorf("offline special registry config failed: %v", err)
	}

	registries := getRegistryList(repo, repoWithNoPort, offline)
	for _, registry := range registries {
		if err := createRegistryHostsTOML(registry, repo, offline, caFile, certsDir); err != nil {
			return err
		}
	}

	if offline == "true" {
		log.Infof("Offline mode configured: public traffic redirects to %s", repo)
	}
	return nil
}

func CniPluginInstall(cniPluginFile string) error {
	bridge := fmt.Sprintf("%s/bridge", cniDirectory)
	if utils.Exists(bridge) {
		return nil
	}

	if err := os.MkdirAll(cniDirectory, utils.DefaultDirPermission); err != nil {
		return fmt.Errorf("create CNI directory: %w", err)
	}
	err := utils.UnTar(cniPluginFile, cniDirectory)
	if err != nil {
		return err
	}
	return nil
}
