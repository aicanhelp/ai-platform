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

package dockerd

import (
	_ "embed"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/registry"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed tlscert.sh
	tlsCertScript string
)

var defaultConfigFile = "/etc/docker/daemon.json"

// containsHost 检查 hosts 列表中是否包含指定的 host
func containsHost(hosts []interface{}, target string) bool {
	for _, h := range hosts {
		if h == target {
			return true
		}
	}
	return false
}

// buildTLSConfig 构建 Docker TLS 配置
func buildTLSConfig() map[string]interface{} {
	return map[string]interface{}{
		"tls":       true,
		"tlsverify": true,
		"tlscacert": "/etc/docker/certs/ca.pem",
		"tlscert":   "/etc/docker/certs/server-cert.pem",
		"tlskey":    "/etc/docker/certs/server-key.pem",
		"hosts":     []string{"tcp://0.0.0.0:2376", "unix:///var/run/docker.sock"},
	}
}

// writeDockerDaemonConfig 将配置写入 daemon.json
func writeDockerDaemonConfig(cfg interface{}) error {
	data, err := json.MarshalIndent(cfg, "", " ")
	if err != nil {
		return errors.Wrapf(err, "marshal docker daemon config failed")
	}
	err = os.WriteFile(defaultConfigFile, data, utils.DefaultFilePermission)
	if err != nil {
		return errors.Wrapf(err, "write docker daemon config file %s failed", defaultConfigFile)
	}
	return nil
}

const (
	// minRuncVersionLen is the minimum length of runc version string
	minRuncVersionLen = 6
	// minRequiredRuncVersion is the minimum required runc version
	minRequiredRuncVersion = "1.1.12"
)

type configMirror struct {
	ExecOptions  []string `json:"exec-opts,omitempty"`
	GraphDriver  string   `json:"storage-driver,omitempty"`
	GraphOptions []string `json:"storage-opts,omitempty"`
	DataRoot     string   `json:"data-root,omitempty"`
	LogConfig
	registry.ServiceOptions
	CommonUnixConfig
}

// LogConfig represents the default log configuration.
// It includes json tags to deserialize configuration from a file
// using the same names that the flags in the command line use.
type LogConfig struct {
	Type   string            `json:"log-driver,omitempty"`
	Config map[string]string `json:"log-opts,omitempty"`
}

// CommonUnixConfig defines configuration of a docker daemon that is
// common across Unix platforms.
type CommonUnixConfig struct {
	Runtimes          map[string]container.HostConfig `json:"runtimes,omitempty"`
	DefaultRuntime    string                          `json:"default-runtime,omitempty"`
	DefaultInitBinary string                          `json:"default-init,omitempty"`
}

func initConfig() *configMirror {
	return &configMirror{
		ExecOptions: []string{"native.cgroupdriver=systemd"},
		GraphDriver: "overlay2",
		LogConfig: LogConfig{
			Type: "json-file",
			Config: map[string]string{
				"max-size": "100m",
			},
		},
		ServiceOptions:   registry.ServiceOptions{},
		CommonUnixConfig: CommonUnixConfig{},
	}
}

// createNewDockerConfig 创建新的 Docker 配置
func createNewDockerConfig(domain string, runtimeStorage string) (bool, error) {
	mirror := initConfig()
	mirror.InsecureRegistries = append(mirror.InsecureRegistries, domain)
	mirror.DataRoot = runtimeStorage
	h, _ := host.Info()
	if (strings.ToLower(h.Platform) == "centos" && strings.HasPrefix(h.PlatformVersion, "8")) ||
		strings.ToLower(h.Platform) == "kylin" {
		mirror.GraphDriver = "overlay2"
		mirror.GraphOptions = []string{}
	}
	b, err := json.MarshalIndent(mirror, "", " ")
	if err != nil {
		return false, err
	}
	err = os.WriteFile(defaultConfigFile, b, utils.DefaultFilePermission)
	if err != nil {
		return false, err
	}
	return true, nil
}

// updateInsecureRegistries 更新不安全镜像仓库配置
func updateInsecureRegistries(conf map[string]interface{}, domain string) bool {
	if conf == nil {
		return false
	}
	ir := make([]string, 0)
	if v, ok := conf["insecure-registries"]; ok {
		aInterface, ok := v.([]interface{})
		if !ok {
			panic("insecure-registries must be []interface{}")
		}
		for _, v1 := range aInterface {
			str, ok := v1.(string)
			if !ok {
				panic("insecure-registries element must be string")
			}
			ir = append(ir, str)
		}
	}
	if !utils.ContainsString(ir, domain) {
		ir = append(ir, domain)
		conf["insecure-registries"] = ir
		return true
	}
	conf["insecure-registries"] = ir
	return false
}

// updateExecOpts 更新执行选项配置
func updateExecOpts(conf map[string]interface{}) bool {
	if conf == nil {
		return false
	}
	eo := make([]string, 0)
	if v, ok := conf["exec-opts"]; ok {
		bInterface, ok := v.([]interface{})
		if !ok {
			panic("exec-opts must be []interface{}")
		}
		for _, v1 := range bInterface {
			str, ok := v1.(string)
			if !ok {
				panic("exec-opts element must be string")
			}
			eo = append(eo, str)
		}
	}
	if !utils.ContainsString(eo, "native.cgroupdriver=systemd") {
		eo = append(eo, "native.cgroupdriver=systemd")
		conf["exec-opts"] = eo
		return true
	}
	conf["exec-opts"] = eo
	return false
}

// updateExistingDockerConfig 更新已存在的 Docker 配置
func updateExistingDockerConfig(domain string, runtimeStorage string) (bool, error) {
	restartFlag := false
	conf := make(map[string]interface{})
	fs, err := os.ReadFile(defaultConfigFile)
	if err != nil {
		return false, err
	}
	if err = json.Unmarshal(fs, &conf); err != nil {
		return false, err
	}

	if updateInsecureRegistries(conf, domain) {
		restartFlag = true
	}
	if updateExecOpts(conf) {
		restartFlag = true
	}

	if dataRoot, ok := conf["data-root"]; ok && dataRoot.(string) != runtimeStorage {
		conf["data-root"] = runtimeStorage
		restartFlag = true
	}

	b, err := json.MarshalIndent(conf, "", " ")
	if err != nil {
		return false, err
	}
	err = os.WriteFile(defaultConfigFile, b, utils.DefaultFilePermission)
	if err != nil {
		return false, err
	}
	return restartFlag, nil
}

func initDockerConfig(domain string, runtimeStorage string) (bool, error) {
	if !utils.Exists(runtimeStorage) {
		if err := os.MkdirAll(runtimeStorage, utils.DefaultDirPermission); err != nil {
			return false, err
		}
	}

	if !utils.Exists(defaultConfigFile) {
		return createNewDockerConfig(domain, runtimeStorage)
	}

	return updateExistingDockerConfig(domain, runtimeStorage)
}

func ensureRuncVersion() bool {
	output, err := global.Command.ExecuteCommandWithOutput("sudo", "runc", "-v")
	if err != nil {
		return false
	}
	runcVersion := strings.Split(output, "\n")
	if len(runcVersion) == 0 {
		return false
	}
	vs := strings.Split(runcVersion[0], " ")
	// 比较字符串
	if len(vs[len(vs)-1]) > minRuncVersionLen && vs[len(vs)-1] >= minRequiredRuncVersion {
		return false
	} else {
		runc := filepath.Join(global.Workspace, "mount", "source_registry", "files", "runc-"+runtime.GOARCH)
		// 将runc拷贝到/usr/bin/runc
		err = utils.CopyFile(runc, "/usr/bin/runc")
		if err != nil {
			log.Debugf("copy runc %s to /usr/bin/runc failed: %v", runc, err)
			return true
		}
		if err := global.Command.ExecuteCommand("sudo", "chmod", "+x", "/usr/bin/runc"); err != nil {
			log.Warnf("chmod +x /usr/bin/runc failed: %v", err)
		}
		return true
	}
}

func ensureDockerCertsDir() error {
	if !utils.Exists("/etc/docker/certs") {
		err := os.MkdirAll("/etc/docker/certs", utils.DefaultDirPermission)
		if err != nil {
			return errors.Wrapf(err, "create docker certs dir failed")
		}
	}
	return nil
}

func generateDockerTlsCert(tlsHost string) error {
	err := os.WriteFile("/etc/docker/certs/tlscert.sh", []byte(tlsCertScript), utils.DefaultDirPermission)
	if err != nil {
		return errors.Wrapf(err, "write tlscert.sh failed")
	}

	executor := &exec.CommandExecutor{}
	output, err := executor.ExecuteCommandWithCombinedOutput(
		"/bin/sh", "-c", "cd /etc/docker/certs && ./tlscert.sh "+tlsHost)
	if err != nil {
		return errors.Wrapf(err, "generate docker tls cert failed, output: %s, err: %v", output, err)
	}

	output, err = executor.ExecuteCommandWithCombinedOutput(
		"/bin/sh", "-c", "echo 'export DOCKER_CONFIG=/etc/docker/certs' >> /etc/profile")
	if err != nil {
		log.Warnf("export DOCKER_CONFIG=/etc/docker/certs to /etc/profile failed, output: %s, err: %v", output, err)
	}
	log.Debugf("export DOCKER_CONFIG=/etc/docker/certs to /etc/profile, output: %s", output)

	output, err = executor.ExecuteCommandWithCombinedOutput("/bin/bash", "-c", "source /etc/profile")
	if err != nil {
		log.Warnf("source /etc/profile failed, output: %s, err: %v", output, err)
	}
	return nil
}

func readOrCreateDaemonConfig() (interface{}, bool, error) {
	if !utils.Exists(defaultConfigFile) {
		if !utils.Exists(filepath.Dir(defaultConfigFile)) {
			err := os.MkdirAll(filepath.Dir(defaultConfigFile), utils.DefaultDirPermission)
			if err != nil {
				return nil, false, errors.Wrapf(err, "create docker daemon config file %s failed", defaultConfigFile)
			}
		}

		log.Infof("docker daemon config file %s not found, create it", defaultConfigFile)
		f, err := os.OpenFile(defaultConfigFile, os.O_RDONLY|os.O_CREATE, utils.DefaultFilePermission)
		if err != nil {
			return nil, false, errors.Wrapf(err, "create docker daemon config file %s failed", defaultConfigFile)
		}
		defer f.Close()
		return buildTLSConfig(), true, nil
	}

	f, err := os.ReadFile(defaultConfigFile)
	if err != nil {
		return nil, false, errors.Wrapf(err, "read docker daemon config file %s failed", defaultConfigFile)
	}
	if len(f) == 0 {
		return buildTLSConfig(), true, nil
	}

	var cfg interface{}
	err = json.Unmarshal(f, &cfg)
	if err != nil {
		return nil, false, errors.Wrapf(err, "unmarshal docker daemon config failed")
	}
	return cfg, false, nil
}

func addTlsConfigToMap(v map[string]interface{}) {
	if v == nil {
		return
	}
	v["tls"] = true
	v["tlsverify"] = true
	v["tlscacert"] = "/etc/docker/certs/ca.pem"
	v["tlscert"] = "/etc/docker/certs/server-cert.pem"
	v["tlskey"] = "/etc/docker/certs/server-key.pem"

	if hosts := v["hosts"]; hosts != nil {
		if h, ok := hosts.([]interface{}); ok {
			if !containsHost(h, "unix:///var/run/docker.sock") {
				h = append(h, "unix:///var/run/docker.sock")
			}
			if !containsHost(h, "tcp://0.0.0.0:2376") {
				h = append(h, "tcp://0.0.0.0:2376")
			}
			v["hosts"] = h
		}
	} else {
		v["hosts"] = []interface{}{
			"unix:///var/run/docker.sock",
			"tcp://0.0.0.0:2376",
		}
	}
}

func configDockerTls(tlsHost string) error {
	log.Infof("config docker tls, tls host: %s", tlsHost)
	if tlsHost == "" {
		tlsHost = "127.0.0.1"
	}

	if err := ensureDockerCertsDir(); err != nil {
		return err
	}
	if err := generateDockerTlsCert(tlsHost); err != nil {
		return err
	}

	cfg, isNew, err := readOrCreateDaemonConfig()
	if err != nil {
		return err
	}
	if isNew {
		return writeDockerDaemonConfig(cfg)
	}

	if v, ok := cfg.(map[string]interface{}); ok {
		addTlsConfigToMap(v)
	}
	return writeDockerDaemonConfig(cfg)
}
