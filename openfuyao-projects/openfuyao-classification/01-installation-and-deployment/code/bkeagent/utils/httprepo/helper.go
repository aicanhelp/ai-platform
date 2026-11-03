package httprepo

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
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
	case "centos", "kylin", "redhat", "fedora", "openeuler", "hopeos":
		packageManager = "yum"
	default:
		packageManager = "unknown"
	}
}

func RepoUpdate() error {
	switch packageManager {
	case "apt":
		output, err := executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s update", packageManager))
		if err != nil {
			return errors.Errorf("update packages failed, err: %v, out: %s", err, output)
		}
	case "yum":
		// yum clean all
		output, err := executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s clean all", packageManager))
		if err != nil {
			return errors.Errorf("update packages failed, err: %v, out: %s", err, output)
		}
		// yum makecache
		output, err = executor.ExecuteCommandWithCombinedOutput("/bin/sh", "-c", fmt.Sprintf("%s makecache", packageManager))
		if err != nil {
			return errors.Errorf("update packages failed, err: %v, out: %s", err, output)
		}
	default:
		return errors.Errorf("package manager %q not supported", packageManager)
	}
	return nil
}

// RepoSearch search package in repo
func RepoSearch(pkg string) error {
	output, err := executor.ExecuteCommandWithOutput("/bin/sh", "-c", fmt.Sprintf("%s search %s  2>/dev/null | grep -w %s", packageManager, pkg, pkg))
	if err != nil {
		return errors.Errorf("search package %q failed, err: %v, out: %s", pkg, err, output)
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
