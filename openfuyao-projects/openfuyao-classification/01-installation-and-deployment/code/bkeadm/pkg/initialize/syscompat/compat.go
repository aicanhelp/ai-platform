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

package syscompat

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

// Compat Compatible with multiple system Settings
func Compat() error {
	h, _ := host.Info()
	stopFirewall()

	output, err := verifyAndInstallIptables(h.Platform)
	if err != nil {
		return err
	}

	if strings.Contains(output, "nf_tables") {
		return switchToLegacyIptables(h.Platform)
	}
	return nil
}

func SetForwardAccept() error {
	log.Info("set iptables FORWARD policy to ACCEPT")
	if err := global.Command.ExecuteCommand("iptables", "-P", "FORWARD", "ACCEPT"); err != nil {
		log.Errorf("Failed to set iptables FORWARD policy to ACCEPT: %v", err)
		return err
	}
	return nil
}

// stopFirewall 停止防火墙
func stopFirewall() {
	if err := global.Command.ExecuteCommand("sudo", "systemctl", "stop", "firewalld"); err != nil {
		logFirewallAction("Stop firewalld", err)
	}
	if err := global.Command.ExecuteCommand("sudo", "systemctl", "disable", "firewalld"); err != nil {
		logFirewallAction("Disable firewalld", err)
	}
	if err := global.Command.ExecuteCommand("sudo", "ufw", "disable"); err != nil {
		logFirewallAction("Disable ufw", err)
	}
}

// logFirewallAction demotes expected absence (service/unit/binary missing) to DEBUG.
func logFirewallAction(action string, err error) {
	if isExpectedFirewallAbsence(err) {
		log.Debugf("%s skipped: %v", action, err)
		return
	}
	log.Warnf("%s failed: %v", action, err)
}

func isExpectedFirewallAbsence(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "not loaded") ||
		strings.Contains(msg, "does not exist") ||
		strings.Contains(msg, "not found") ||
		strings.Contains(msg, "command not found")
}

// verifyAndInstallIptables 验证并安装 iptables
func verifyAndInstallIptables(platform string) (string, error) {
	output, err := global.Command.ExecuteCommandWithOutput("iptables", "-V")
	if err == nil {
		log.Infof("iptables -V output: %s", output)
		return output, nil
	}

	log.Warnf("iptables -V error: %v", err)

	if !strings.Contains(err.Error(), "not found") {
		return "", err
	}

	return installIptables(platform)
}

// installIptables 根据平台安装 iptables
func installIptables(platform string) (string, error) {
	log.Info("install iptables...")

	switch strings.ToLower(platform) {
	case "centos", "kylin", "openeuler", "uos":
		return installIptablesYum()
	case "ubuntu", "debian", "fedora":
		return installIptablesApt()
	default:
		log.Error("Unsupported platform, only support centos, ubuntu, debian, fedora")
		return "", nil
	}
}

// verifyIptablesInstallation 验证 iptables 安装
func verifyIptablesInstallation() (string, error) {
	output, err := global.Command.ExecuteCommandWithOutput("iptables", "-V")
	if err != nil {
		log.Errorf("Failed to exec iptables -V, %v", err)
		return "", nil
	}
	log.Infof("iptables -V output: %s", output)
	return output, nil
}

// installIptablesYum 使用 yum 安装 iptables
func installIptablesYum() (string, error) {
	if err := global.Command.ExecuteCommand("yum", "-y", "install", "iptables"); err != nil {
		log.Errorf("Failed to install iptables, %v", err)
		return "", nil
	}
	return verifyIptablesInstallation()
}

// installIptablesApt 使用 apt 安装 iptables
func installIptablesApt() (string, error) {
	if err := global.Command.ExecuteCommand("apt", "-y", "clean"); err != nil {
		log.Warnf("apt clean failed before install iptables: %v", err)
	}
	if err := global.Command.ExecuteCommand("apt", "-y", "update"); err != nil {
		log.Warnf("apt update failed before install iptables: %v", err)
	}
	if err := global.Command.ExecuteCommand("apt", "-y", "install", "iptables"); err != nil {
		log.Errorf("Failed to install iptables, %v", err)
		return "", nil
	}
	return verifyIptablesInstallation()
}

// switchToLegacyIptables 切换到传统 iptables
func switchToLegacyIptables(platform string) error {
	switch strings.ToLower(platform) {
	case "centos", "kylin", "openeuler", "uos":
		return reinstallIptablesYum()
	case "ubuntu", "debian":
		return updateAlternativesDebian()
	case "fedora":
		return updateAlternativesFedora()
	default:
		return errors.New(fmt.Sprintf("%s is not supported", platform))
	}
}

// reinstallIptablesYum 使用 yum 重新安装 iptables
func reinstallIptablesYum() error {
	if err := global.Command.ExecuteCommand("yum", "-y", "remove", "iptables"); err != nil {
		log.Errorf("Failed to remove iptables, %v", err)
		return err
	}
	if err := global.Command.ExecuteCommand("yum", "-y", "install", "iptables"); err != nil {
		log.Errorf("Failed to install iptables, %v", err)
		return err
	}
	return nil
}

// updateAlternativesDebian 更新 Debian/Ubuntu 的 iptables 替代方案
func updateAlternativesDebian() error {
	alternatives := [][]string{
		{"iptables", "/usr/sbin/iptables-legacy"},
		{"ip6tables", "/usr/sbin/ip6tables-legacy"},
		{"arptables", "/usr/sbin/arptables-legacy"},
		{"ebtables", "/usr/sbin/ebtables-legacy"},
	}
	for _, alt := range alternatives {
		if err := global.Command.ExecuteCommand("update-alternatives", "--set", alt[0], alt[1]); err != nil {
			log.Warnf("Failed to update %s, %v", alt[0], err)
			return nil
		}
	}
	return nil
}

// updateAlternativesFedora 更新 Fedora 的 iptables 替代方案
func updateAlternativesFedora() error {
	if err := global.Command.ExecuteCommand("update-alternatives",
		"--set", "iptables", "/usr/sbin/iptables-legacy"); err != nil {
		log.Warnf("Failed to update iptables, %v", err)
		return nil
	}
	return nil
}

func RepoUpdate() error {
	// Repo metadata refresh can be slow on unstable networks.
	// Keep a bounded timeout to avoid blocking init/reset for too long.
	const repoUpdateTimeout = 120 * time.Second
	packageManager := ""
	h, _, _, err := host.PlatformInformation()
	if err != nil {
		log.Errorf("get host platform information failed, err: %v", err)
	}
	switch h {
	case "ubuntu", "debian":
		packageManager = "apt"
	case "centos", "kylin", "redhat", "fedora", "openeuler", "hopeos", "uos":
		packageManager = "yum"
	default:
		packageManager = "unknown"
	}
	switch packageManager {
	case "apt":
		output, err := global.Command.ExecuteCommandWithTimeout(
			repoUpdateTimeout, "/bin/sh", "-c", fmt.Sprintf("%s -y clean", packageManager))
		if err != nil {
			log.Errorf("update packages failed, err: %v, out: %s", err, output)
			return nil
		}

		// Reduce retry/wait behavior to avoid long stalls during update in restricted networks.
		aptCmd := fmt.Sprintf("%s -y -o Acquire::Retries=1 -o Acquire::http::Timeout=%d update",
			packageManager, utils.DefaultTimeoutSeconds)
		output, err = global.Command.ExecuteCommandWithTimeout(repoUpdateTimeout, "/bin/sh", "-c", aptCmd)
		if err != nil {
			log.Errorf("update packages failed, err: %v, out: %s", err, output)
			return nil
		}
	case "yum":
		// Skip "clean all" to avoid forcing full metadata rebuild every time.
		// Use "--timer" and setopts to reduce network stalls and speed up refresh.
		yumCmd := fmt.Sprintf("%s -y makecache --timer --setopt=skip_if_unavailable=true --setopt=fastestmirror=true",
			packageManager)
		output, err := global.Command.ExecuteCommandWithTimeout(repoUpdateTimeout, "/bin/sh", "-c", yumCmd)
		if err != nil {
			log.Errorf("update packages failed, err: %v, out: %s", err, output)
			return nil
		}
	default:
		log.Errorf("package manager %q not supported", packageManager)
		return nil
	}
	return nil
}
