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

package containerd

import (
	"archive/tar"
	"compress/gzip"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"text/template"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	econd "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name              = "InstallContainerd"
	eight             = 8
	successStatusCode = 200
	RwRR              = 0644
	Rwxxx             = 0711
)

var (
	//go:embed config.toml
	configToml              string
	defaultRepo             = fmt.Sprintf("%s:%s", utils.DefaultImageRepo, utils.DefaultImageRepoPort)
	defaultSandbox          = fmt.Sprintf("%s/kubernetes/%s", defaultRepo, "pause:3.9")
	defaultRuntime          = "runc"
	defaultInstallDirectory = "/"
	defaultDataRoot         = "/var/lib/containerd"
	containerdCertsDir      = "/etc/containerd/certs.d"
)

type ContainerdPlugin struct {
	exec             exec.Executor
	serviceDropInDir string
}

func New(exec exec.Executor) plugin.Plugin {
	return &ContainerdPlugin{
		exec: exec,
	}
}

func (cp *ContainerdPlugin) Name() string {
	return Name
}

func (cp *ContainerdPlugin) getPlatform() string {
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

func (cp *ContainerdPlugin) createHostsTOML(runtimeParam map[string]string) error {
	repo := runtimeParam["repo"]
	offline := runtimeParam["repoInsecure"]
	registries := []string{repo}

	if offline == "true" {
		publicRegistries := strings.Split(runtimeParam["insecureRegistries"], ",")
		registries = append(registries, publicRegistries...)
		log.Info("Offline mode: configuring public registry redirects")
	}

	for _, registry := range registries {
		registryDir := filepath.Join(containerdCertsDir, registry)
		if err := os.MkdirAll(registryDir, utils.RwxRxRx); err != nil {
			return fmt.Errorf("create %s dir failed: %w", registry, err)
		}

		data := struct {
			Repo     string
			Registry string
			Offline  string
		}{Repo: repo, Registry: registry, Offline: offline}

		hostsTpl := `server = "https://{{.Registry}}"
[host."https://{{.Repo}}"]
  capabilities = ["pull", "resolve", "push"]
  skip_verify = true
`
		hostsPath := filepath.Join(registryDir, "hosts.toml")
		f, err := os.OpenFile(hostsPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, utils.RwRR)
		if err != nil {
			return fmt.Errorf("create %s hosts.toml failed: %w", registry, err)
		}

		if err := executeTemplateWithFile(hostsTpl, "baseHosts", data, f); err != nil {
			if closeErr := f.Close(); closeErr != nil {
				return errors.Join(
					fmt.Errorf("process template for %s failed: %w", registry, err),
					fmt.Errorf("close file failed: %w", closeErr),
				)
			}
			return fmt.Errorf("process template for %s failed: %w", registry, err)
		}

		if err = f.Close(); err != nil {
			return fmt.Errorf("close hosts.toml fail: %w", err)
		}
		log.Infof("Created base hosts.toml: %s", hostsPath)
	}

	if offline == "true" {
		log.Infof("Offline mode configured: public traffic redirects to %s", repo)
	}

	return nil
}

func createTempScript(content string) (string, error) {
	tmpFile, err := os.CreateTemp("", "containerd-script-*.sh")
	if err != nil {
		return "", fmt.Errorf("create temporary file failed: %w", err)
	}
	defer tmpFile.Close()

	if !strings.HasPrefix(strings.TrimSpace(content), "#!") {
		content = "#!/bin/bash\n" + content
	}

	if _, err = tmpFile.WriteString(content); err != nil {
		return "", fmt.Errorf("write content filed: %w", err)
	}

	return tmpFile.Name(), nil
}

func (cp *ContainerdPlugin) executeScript(script *bkev1beta1.ScriptConfig) error {
	var scriptContent string
	var scriptPath string

	if script.Content != "" {
		scriptContent = script.Content
		tmpFile, err := createTempScript(scriptContent)
		if err != nil {
			return fmt.Errorf("failed to create temporary script: %w", err)
		}
		defer os.Remove(tmpFile)
		scriptPath = tmpFile
	} else if script.Path != "" {
		if _, err := os.Stat(script.Path); os.IsNotExist(err) {
			return fmt.Errorf("script file does not exist: %s", script.Path)
		}
		scriptPath = script.Path
	} else {
		return nil
	}

	if err := os.Chmod(scriptPath, utils.RwxRxRx); err != nil {
		return fmt.Errorf("failed to set script permissions: %w", err)
	}

	out, err := cp.exec.ExecuteCommandWithCombinedOutput(script.Interpreter, append([]string{scriptPath}, script.Args...)...)
	if err != nil {
		log.Warnf("execute shell script failed, err: %v, out: %s", err, out)
	}
	return nil
}

func generateOverrideService(service *bkev1beta1.ServiceConfig, serviceDropInDir string) error {
	generate := NewServiceDropInGenerator(serviceDropInDir)
	if generate == nil {
		return fmt.Errorf("new service drop in generator failed")
	}
	return generate.GenerateServiceDropIn(service)
}

func renderConfigToml(main *bkev1beta1.MainConfig, runtimeParam map[string]string) error {
	if runtimeParam == nil {
		return fmt.Errorf("runtime param is nil")
	}
	if main.SandboxImage != "" {
		runtimeParam["sandbox"] = main.SandboxImage
	}
	if main.Root != "" {
		runtimeParam["dataRoot"] = main.Root
	}
	if main.State != "" {
		runtimeParam["dataState"] = main.State
	}
	if main.ConfigPath != "" {
		runtimeParam["configPath"] = main.ConfigPath
	}
	if main.MetricsAddress != "" {
		runtimeParam["metricsAddress"] = main.MetricsAddress
	}
	// Render configuration file
	if err := writeConfigToDisk(runtimeParam); err != nil {
		return err
	}
	return nil
}

func generateHostsToml(registry *bkev1beta1.RegistryConfig) error {
	generate := NewHostsTOMLGenerator(registry.ConfigPath)
	if generate == nil {
		return fmt.Errorf("new hosts toml generator failed")
	}
	return generate.GenerateMultipleHostsTOML(registry.Configs)
}

// 根据cr配置containerd
func (cp *ContainerdPlugin) generateContainerdCfg(runtimeParam map[string]string) error {
	cc, err := plugin.GetContainerdConfig(runtimeParam["containerdConfig"])
	if err != nil {
		return fmt.Errorf("get containerd config: %w", err)
	}
	if cc.Script != nil {
		// script shell script execution configuration
		if err = cp.executeScript(cc.Script); err != nil {
			return fmt.Errorf("execute script failed: %w", err)
		}
	}
	if cc.Service != nil {
		// 处理containerd.service
		if err = generateOverrideService(cc.Service, cp.serviceDropInDir); err != nil {
			return fmt.Errorf("generate containerd override service failed: %w", err)
		}
	}
	if cc.Main != nil {
		// 处理config.toml
		if err = renderConfigToml(cc.Main, runtimeParam); err != nil {
			return fmt.Errorf("render containerd config toml failed: %w", err)
		}
	}
	if cc.Registry != nil {
		// 处理hosts.toml
		if err = generateHostsToml(cc.Registry); err != nil {
			return fmt.Errorf("generate containerd hosts toml failed: %w", err)
		}
	}
	return nil
}

func (cp *ContainerdPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"url":                {Key: "url", Value: "", Required: true, Default: "", Description: "containerd.tar.gz download address"},
		"repo":               {Key: "repo", Value: "", Required: false, Default: defaultRepo, Description: "Image repository address"},
		"sandbox":            {Key: "sandbox", Value: "", Required: false, Default: defaultSandbox, Description: "Pod sandbox"},
		"runtime":            {Key: "runtime", Value: "", Required: false, Default: defaultRuntime, Description: "Container runtime"},
		"dataRoot":           {Key: "dataRoot", Value: "", Required: false, Default: defaultDataRoot, Description: "Specify the data directory"},
		"directory":          {Key: "directory", Value: "", Required: false, Default: defaultInstallDirectory, Description: "Specify the unzip directory"},
		"insecureRegistries": {Key: "insecureRegistries", Value: "", Required: false, Default: "", Description: "Specify the insecure registries, split by ','"},
		"containerdConfig":   {Key: "containerdConfig", Value: "NameSpace:Name", Required: false, Default: "", Description: "Specify the containerd config, example ns:name"},
		"immutableOS":        {Key: "immutableOS", Value: "", Required: false, Default: "false", Description: "immutable OS mode, skip containerd download and extract"},
	}
}

// Execute Install and start Containerd example ["InstallContainerd", "url=http://deploy.bocloud.k8s:40080", "sandbox=deploy.bocloud.k8s:40443/kubernetes/pause:3.5.1"]
func (cp *ContainerdPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	cpc := *cp
	cpc.exec = exec.Bound(cp.exec, ctx)
	cp = &cpc

	var result []string
	// Parse command
	runtimeParam, err := plugin.ParseCommands(cp, commands)
	if err != nil {
		return result, err
	}
	if !strings.HasSuffix(runtimeParam["directory"], "/") {
		runtimeParam["directory"] = runtimeParam["directory"] + "/"
	}

	// 不可变 OS：containerd 镜像内置，跳过下载解压
	if runtimeParam["immutableOS"] != "true" {
		tarFile := path.Join(os.TempDir(), fmt.Sprintf("containerd-%s.tar.gz", econd.GenerateID()[:eight]))
		defer os.Remove(tarFile)
		if err = downloadTar(runtimeParam["url"], tarFile); err != nil {
			return result, err
		}
		if err = unTar(tarFile, runtimeParam["directory"]); err != nil {
			return result, err
		}
	} else {
		log.Info("skip containerd download in immutable OS mode")
	}

	runtimeParam["platform"] = cp.getPlatform()
	if err = cp.configureContainerd(runtimeParam); err != nil {
		return result, err
	}
	if res, err := cp.startContainerdService(); err != nil {
		return res, err
	}
	if err = econd.WaitContainerdReady(); err != nil {
		return nil, err
	}
	return result, nil
}

func (cp *ContainerdPlugin) configureContainerd(runtimeParam map[string]string) error {
	if runtimeParam == nil {
		return fmt.Errorf("runtime param is nil")
	}
	if runtimeParam["containerdConfig"] == "" {
		return cp.configureContainerdLegacy(runtimeParam)
	}
	// 新逻辑，从cr中获取containerd获取数据配置containerd
	if err := cp.generateContainerdCfg(runtimeParam); err != nil {
		return fmt.Errorf("Failed to generate containerd config from cr: %w", err)
	}
	return nil
}

// configureContainerdLegacy handles legacy containerd configuration
func (cp *ContainerdPlugin) configureContainerdLegacy(runtimeParam map[string]string) error {
	if runtimeParam == nil {
		return fmt.Errorf("runtime param is nil")
	}
	if !utils.Exists(runtimeParam["dataRoot"]) {
		if err := os.MkdirAll(runtimeParam["dataRoot"], Rwxxx); err != nil {
			return err
		}
	}
	if runtimeParam["insecureRegistries"] != "" {
		registries := strings.Split(runtimeParam["insecureRegistries"], ",")
		registriesLen := len(registries)
		for i := range registriesLen {
			if registries[i] == runtimeParam["repo"] && runtimeParam["repo"] != "cr.openfuyao.cn" && i >= 0 && i < len(registries) {
				registries = append(registries[:i], registries[i+1:]...)
				runtimeParam["insecureRegistries"] = strings.Join(registries, ",")
				runtimeParam["repoInsecure"] = "true"
				break
			}
		}
	}
	if err := writeConfigToDisk(runtimeParam); err != nil {
		return err
	}
	if err := cp.createHostsTOML(runtimeParam); err != nil {
		return fmt.Errorf("Failed to create hosts.toml: %w", err)
	}
	return nil
}

func (cp *ContainerdPlugin) startContainerdService() ([]string, error) {
	out, err := cp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl daemon-reload")
	if err != nil {
		log.Warnf("daemon-reload failed, err: %v, out: %s", err, out)
	}

	out, err = cp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl enable containerd")
	if err != nil {
		log.Warnf("enable containerd failed, err: %v, out: %s", err, out)
	}

	out, err = cp.exec.ExecuteCommandWithCombinedOutput("sh", "-c", "systemctl restart containerd")
	if err != nil {
		errorMsg := fmt.Sprintf("start containerd failed, err: %v, out: %s", err, out)
		return []string{errorMsg}, fmt.Errorf("start containerd failed, err: %w, out: %s", err, out)
	}

	return []string{}, nil
}

func downloadTar(url, tar string) error {
	newFile, err := os.OpenFile(tar, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, RwRR)
	if err != nil {
		return err
	}
	defer newFile.Close()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	if resp.StatusCode != successStatusCode {
		return fmt.Errorf("http request failed, status code: %d", resp.StatusCode)
	}

	defer resp.Body.Close()

	numBytesWritten, err := io.Copy(newFile, resp.Body)
	if err != nil {
		return err
	}
	log.Infof("Downloaded %d byte file.\n", numBytesWritten)
	return nil
}

// ensureDirectory ensure directory exists or not.
func ensureDirectory(path string, mode os.FileMode) error {
	if utils.IsDir(path) {
		return nil
	}
	if err := os.MkdirAll(path, mode); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", path, err)
	}
	return nil
}

// extractFile extract a regular file.
func extractFile(tr *tar.Reader, path string, mode os.FileMode) error {
	// Create the file
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, mode)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", path, err)
	}
	defer file.Close()

	// Copy file contents
	n, err := io.Copy(file, tr)
	if err != nil {
		return fmt.Errorf("failed to write file %s: %w", path, err)
	}

	log.Infof("extracted: %s, wrote %d bytes", path, n)
	return nil
}

func unTar(src, dst string) error {
	// Open the tar.gz file
	fr, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer fr.Close()

	// Create gzip reader
	gr, err := gzip.NewReader(fr)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gr.Close()

	// Create tar reader
	tr := tar.NewReader(gr)

	// Process each file in the archive
	for {
		hdr, err := tr.Next()
		switch {
		case err == io.EOF:
			return nil // Finished successfully
		case err != nil:
			return fmt.Errorf("tar read error: %w", err)
		case hdr == nil:
			continue // Skip nil headers
		default:
			// Continue processing when err == nil and hdr != nil
		}

		// Construct full destination path
		targetPath := filepath.Join(dst, hdr.Name)

		// Handle based on file type
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := ensureDirectory(targetPath, utils.RwxRxRx); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := extractFile(tr, targetPath, os.FileMode(hdr.Mode)); err != nil {
				return err
			}
		default:
			log.Warnf("unhandled file type %d for %s", hdr.Typeflag, hdr.Name)
		}
	}
}

func ensureRunTime() bool {
	_, err := econd.NewContainedClient()
	if err != nil {
		return false
	}
	return true
}

// ensureContainerdConfigDir 确保 /etc/containerd 目录存在，仅记录错误不中断
func ensureContainerdConfigDir(dir string) {
	if err := os.MkdirAll(dir, utils.RwxRxRx); err != nil {
		log.Warnf("create containerd config dir %s failed: %v", dir, err)
	}
}

func writeConfigToDisk(runtimeParam map[string]string) error {
	// 确保 /etc/containerd 目录存在（immutable OS 模式跳过下载解压时目录可能不存在）
	ensureContainerdConfigDir(filepath.Join(runtimeParam["directory"], "etc/containerd"))
	// Render configuration file
	f, err := os.OpenFile(fmt.Sprintf("%s%s", runtimeParam["directory"], "etc/containerd/config.toml"), os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	tpl, err := template.New("config.toml").Funcs(commonFuncMaps()).Parse(configToml)
	if err != nil {
		return err
	}
	return tpl.Execute(f, runtimeParam)
}

func commonFuncMaps() template.FuncMap {
	return template.FuncMap{
		"split": func(s string, sep string) []string {
			return strings.Split(s, sep)
		},
		"default": func(value, defaultValue interface{}) interface{} {
			if value == nil || value == "" {
				return defaultValue
			}
			return value
		},
	}
}
