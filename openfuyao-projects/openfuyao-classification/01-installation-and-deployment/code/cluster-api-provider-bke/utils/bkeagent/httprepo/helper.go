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

package httprepo

import (
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// repoUpdateTimeout bounds metadata refresh so init/reset are not blocked too long.
	repoUpdateTimeout = 120 * time.Second
	// aptAcquireTimeoutSeconds limits apt HTTP wait on restricted networks.
	aptAcquireTimeoutSeconds = 15
)

var (
	packageManager string
	executor       = &exec.CommandExecutor{}
)

func init() {
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
}

// RepoUpdate refreshes package metadata. Aligned with bkeadm syscompat.RepoUpdate:
// bounded timeout, skip yum clean all, best-effort (log and continue on failure).
func RepoUpdate() error {
	switch packageManager {
	case "apt":
		output, err := executor.ExecuteCommandWithTimeout(
			repoUpdateTimeout, "/bin/sh", "-c", fmt.Sprintf("%s -y clean", packageManager))
		if err != nil {
			log.Errorf("clean packages failed, err: %v, out: %s", err, output)
			return nil
		}

		aptCmd := fmt.Sprintf("%s -y -o Acquire::Retries=1 -o Acquire::http::Timeout=%d update",
			packageManager, aptAcquireTimeoutSeconds)
		output, err = executor.ExecuteCommandWithTimeout(repoUpdateTimeout, "/bin/sh", "-c", aptCmd)
		if err != nil {
			log.Errorf("update packages failed, err: %v, out: %s", err, output)
			return nil
		}
	case "yum":
		// Skip "clean all" to avoid forcing full metadata rebuild every time.
		// Use "--timer" and setopts to reduce network stalls and speed up refresh.
		yumCmd := fmt.Sprintf("%s -y makecache --timer --setopt=skip_if_unavailable=true --setopt=fastestmirror=true",
			packageManager)
		output, err := executor.ExecuteCommandWithTimeout(repoUpdateTimeout, "/bin/sh", "-c", yumCmd)
		if err != nil {
			log.Errorf("update packages failed, err: %v, out: %s", err, output)
			return nil
		}
	default:
		log.Errorf("skip RepoUpdate (no-op): package manager %q not supported", packageManager)
		return nil
	}
	return nil
}

// RepoSearch search package in repo
func RepoSearch(pkg string) error {
	output, err := executor.ExecuteCommandWithOutput("/bin/sh", "-c", fmt.Sprintf("%s search %s  2>/dev/null | grep -w %s", packageManager, pkg, pkg))
	if err != nil {
		return fmt.Errorf("search package %q failed, err: %w, out: %s", pkg, err, output)
	}
	if strings.Contains(output, pkg) {
		return nil
	}
	return errors.Errorf("package %q not found", pkg)
}

// RepoInstall install packages
func RepoInstall(packages ...string) error {
	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s install -y %s", packageManager, strings.Join(packages, " ")))
	if err != nil {
		return errors.Errorf("install packages %q failed, err: %v, out: %s", strings.Join(packages, " "), err, output)
	}
	return nil
}

// RepoRemove remove packages
func RepoRemove(packages ...string) error {
	output, err := executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s remove -y %s", packageManager, strings.Join(packages, " ")))
	if err != nil {
		return errors.Errorf("remove packages %q failed, err: %v, out: %s", strings.Join(packages, " "), err, output)
	}

	if packageManager == "apt" {
		//apt purge
		output, err = executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s purge -y %s", packageManager, strings.Join(packages, " ")))
		if err != nil {
			return errors.Errorf("purge packages %q failed, err: %v, out: %s", strings.Join(packages, " "), err, output)
		}
	}
	return nil
}
