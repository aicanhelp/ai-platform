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
	"os"
	"os/exec"
	"strings"

	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var bkeManagedSysctlSettings = []string{
	"fs.file-max = 9000000",
	"fs.inotify.max_user_watches = 1000000",
	"fs.inotify.max_user_instances = 1000000",
	"net.ipv4.ip_forward = 1",
}

func SetSysctl() {
	sysconf, err := os.ReadFile("/etc/sysctl.conf")
	if err != nil {
		if !os.IsNotExist(err) {
			log.Warnf("Unable to read sysctl.conf: %v", err)
			return
		}
		sysconf = nil
	}

	content := removeManagedBlockFromContent(string(sysconf), BKEManagedSysctlBegin, BKEManagedSysctlEnd)
	var appendSettings []string
	for _, setting := range bkeManagedSysctlSettings {
		key := strings.Split(setting, "=")[0]
		if !strings.Contains(content, strings.TrimSpace(key)) {
			appendSettings = append(appendSettings, setting)
		}
	}

	if len(appendSettings) == 0 && content == string(sysconf) {
		reloadSysctl()
		return
	}

	if len(appendSettings) > 0 {
		content = strings.TrimRight(content, "\n")
		if content != "" {
			content += "\n"
		}
		content += BKEManagedSysctlBegin + "\n"
		content += strings.Join(appendSettings, "\n") + "\n"
		content += BKEManagedSysctlEnd + "\n"
	}

	file, err := os.OpenFile("/etc/sysctl.conf", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, utils.DefaultReadWritePermission)
	if err != nil {
		log.Warnf("Unable to open file %s %v", "/etc/sysctl.conf", err)
		return
	}
	defer file.Close()

	if _, err := file.WriteString(content); err != nil {
		log.Warnf("Unable to write sysctl.conf: %v", err)
		return
	}

	reloadSysctl()
}

func reloadSysctl() {
	err := exec.Command("sudo", "sysctl", "-p").Run()
	if err != nil {
		log.Warnf("Unable to set sysctl %v", err)
		return
	}
}

func RemoveBKEManagedConfig() {
	removeManagedBlockFromFile("/etc/hosts", BKEManagedHostsBegin, BKEManagedHostsEnd)
	removeManagedBlockFromFile("/etc/sysctl.conf", BKEManagedSysctlBegin, BKEManagedSysctlEnd)
	reloadSysctl()
}

func removeManagedBlockFromFile(filePath, begin, end string) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Warnf("Unable to read %s: %v", filePath, err)
		}
		return
	}

	content := string(data)
	cleanContent := removeManagedBlockFromContent(content, begin, end)
	if cleanContent == content {
		return
	}

	if err = os.WriteFile(filePath, []byte(cleanContent), utils.DefaultReadWritePermission); err != nil {
		log.Warnf("Unable to write %s: %v", filePath, err)
	}
}
