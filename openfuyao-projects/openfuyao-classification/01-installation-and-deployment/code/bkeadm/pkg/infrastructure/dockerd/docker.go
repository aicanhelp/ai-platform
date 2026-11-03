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
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

const (
	OverrideDockerConfig = `[Service]
ExecStart=
ExecStart=/usr/bin/dockerd
`
)

// waitForDockerConnection waits for Docker to become available
func waitForDockerConnection(maxRetries int, interval time.Duration) bool {
	for i := 0; i < maxRetries; i++ {
		time.Sleep(interval)
		if ensureDockerConnect() {
			return true
		}
	}
	return false
}

// getPackageManager returns the appropriate package manager for the platform
func getPackageManager(platform string) string {
	switch platform {
	case "ubuntu", "debian":
		return "apt"
	case "centos", "kylin", "redhat", "fedora":
		return "yum"
	default:
		log.Debug("unknown platform, default yum package manager")
		return "yum"
	}
}

// installDockerPackage installs Docker using the appropriate method
func installDockerPackage(platform, dockerdFile, pkgManager string) error {
	if platform == "kylin" && utils.Exists(dockerdFile) {
		return utils.UnTar(dockerdFile, "/")
	}

	result, err := global.Command.ExecuteCommandWithOutput(
		"sh", "-c", fmt.Sprintf("%s -y install docker-ce", pkgManager))
	if err != nil {
		return fmt.Errorf("install docker-ce: %w, output: %s", err, result)
	}
	return nil
}

// createSystemdDockerOverride creates the systemd override configuration for Docker
func createSystemdDockerOverride() error {
	configDir := "/etc/systemd/system/docker.service.d"
	configFile := configDir + "/docker.conf"

	if utils.Exists(configFile) {
		return nil
	}

	if !utils.Exists(configDir) {
		if err := os.MkdirAll(configDir, utils.DefaultDirPermission); err != nil {
			return fmt.Errorf("create docker systemd config dir %s: %w", configDir, err)
		}
	}

	if err := os.WriteFile(configFile, []byte(OverrideDockerConfig), utils.DefaultFilePermission); err != nil {
		return fmt.Errorf("write docker systemd config %s: %w", configFile, err)
	}
	return nil
}

// startDockerService enables and starts the Docker service
func startDockerService() error {
	// Enable docker
	result, err := global.Command.ExecuteCommandWithCombinedOutput("sh", "-c", "sudo systemctl enable docker")
	if err != nil {
		log.Warnf("systemctl enable docker error %s", result)
	}

	// Daemon reload
	result, err = global.Command.ExecuteCommandWithCombinedOutput("sh", "-c", "sudo systemctl daemon-reload")
	if err != nil {
		log.Warnf("systemctl daemon-reload error %s", result)
	}

	// Start docker
	result, err = global.Command.ExecuteCommandWithCombinedOutput("sh", "-c", "sudo systemctl start docker")
	if err != nil {
		return fmt.Errorf("start docker service: %w, output: %s", err, result)
	}
	return nil
}

// updateExistingDocker 更新已存在的 Docker 配置并重启（如需要）
func updateExistingDocker(domain, runtimeStorage, hostIp string) error {
	flag, err := initDockerConfig(domain, runtimeStorage)
	if err != nil {
		return err
	}
	if err = configDockerTls(hostIp); err != nil {
		return err
	}
	flag2 := ensureRuncVersion()
	if flag || flag2 {
		result, err := global.Command.ExecuteCommandWithCombinedOutput("systemctl", "restart", "docker")
		if err != nil {
			return fmt.Errorf("restart docker service: %w, output: %s", err, result)
		}
		waitForDockerConnection(utils.DockerConnectionMaxRetries, utils.DockerConnectionRetrySeconds*time.Second)
	}
	return nil
}

// installNewDocker 安装新的 Docker
func installNewDocker(domain, runtimeStorage, dockerdFile, hostIp string) error {
	log.Info("install docker...")

	platform, _, _, err := host.PlatformInformation()
	if err != nil {
		return fmt.Errorf("get host platform information: %w", err)
	}

	pkgManager := getPackageManager(platform)
	if err = installDockerPackage(platform, dockerdFile, pkgManager); err != nil {
		return err
	}

	if err = os.Mkdir("/etc/docker", utils.DefaultDirPermission); err != nil && !os.IsExist(err) {
		log.Warnf("Failed to create /etc/docker: %v", err)
	}

	if _, err = initDockerConfig(domain, runtimeStorage); err != nil {
		return fmt.Errorf("init docker config for domain %s: %w", domain, err)
	}
	ensureRuncVersion()

	if err = configDockerTls(hostIp); err != nil {
		return err
	}

	if err = createSystemdDockerOverride(); err != nil {
		return err
	}

	if err = startDockerService(); err != nil {
		return err
	}

	if waitForDockerConnection(utils.DockerConnectionMaxRetries, utils.DockerConnectionRetrySeconds*time.Second) {
		return nil
	}

	return errors.New("docker installation failed")
}

// EnsureDockerServer ensures Docker service is running and properly configured with the specified parameters
func EnsureDockerServer(domain string, runtimeStorage string, dockerdFile string, hostIp string) error {
	if ensureDockerConnect() {
		return updateExistingDocker(domain, runtimeStorage, hostIp)
	}
	return installNewDocker(domain, runtimeStorage, dockerdFile, hostIp)
}

func ensureDockerConnect() bool {
	if global.Docker == nil {
		global.Docker, _ = docker.NewDockerClient()
	}
	if global.Docker != nil {
		_, err := global.Docker.GetClient().Ping(context.Background())
		if err == nil {
			return true
		}
	}
	return false
}
