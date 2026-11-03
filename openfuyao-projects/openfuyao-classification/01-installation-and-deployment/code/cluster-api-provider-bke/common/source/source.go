/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package source

import (
	_ "embed"
	"fmt"
	"net/url"
	"os"
	"runtime"
	"strings"

	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/warehouse"
)

// Constants for version parsing
const (
	minimumVersionParts = 2
)

var (
	// yum: 直接在 /etc/yum.repos.d/ 下增加 bke.repo，不移动系统原有源
	yumRepos    = "/etc/yum.repos.d"
	bkeRepoFile = "bke.repo"

	// apt: 写入 sources.list.d 目录和 preferences.d，不覆盖系统 sources.list
	aptSourcesDir = "/etc/apt/sources.list.d"
	aptPrefsDir   = "/etc/apt/preferences.d"
	bkeListFile   = "bke.list"
	bkePrefsFile  = "bke"

	centos    = "CentOS"
	kylin     = "Kylin"
	ubuntu    = "Ubuntu"
	openeuler = "OpenEuler"
	hopeos    = "HopeOS"
	euleros   = "EulerOS"

	// priority=1 使 bke 源优先级高于默认源（需 yum-plugin-priorities 或 dnf 原生支持）
	yumSource = `[bke]
name = OpenFuyao Repository
baseurl = {{.baseurl}}
enabled = 1
gpgcheck = 0
priority = 1
`

	// 写入 sources.list.d，不覆盖系统源
	aptSource = `deb [trusted=yes] {{.baseurl}} ./
`

	// Pin-Priority: 1001 高于 apt 默认优先级 500，使 bke 源包版本优先被选用
	aptPrefs = `Package: *
Pin: origin {{.origin}}
Pin-Priority: 1001
`
)

func GetCustomDownloadPath(url string) string {
	if strings.HasSuffix(url, "/") {
		return url + "files"
	}
	return url + "/files"
}

func GetRPMDownloadPath(url string) (string, error) {
	baseurl := url
	h, err := host.Info()
	if err != nil {
		return baseurl, err
	}
	if !strings.HasSuffix(baseurl, "/") {
		baseurl += "/"
	}
	switch strings.ToLower(h.Platform) {
	case "centos":
		baseurl += centos
	case "kylin":
		baseurl += kylin
	case "ubuntu":
		baseurl += ubuntu
	case "openeuler":
		baseurl += openeuler
	case "hopeos":
		baseurl += hopeos
	case "euleros":
		baseurl += euleros
	default:
		return baseurl, errors.New(fmt.Sprintf("The operating system is not supported %s", h.Platform))
	}
	ver := ""
	if !strings.Contains(h.PlatformVersion, ".") {
		ver = strings.ToUpper(h.PlatformVersion)
	} else {
		version := strings.Split(h.PlatformVersion, ".")
		if len(version) < minimumVersionParts {
			return baseurl, errors.New(fmt.Sprintf("Error getting system version %s", h.PlatformVersion))
		}
		ver = version[0]
	}
	baseurl += "/" + ver
	baseurl += "/" + runtime.GOARCH
	return baseurl, nil
}

// SetSource 在系统中增加一个高优先级的 BKE 软件源。
// 对 yum 系统：在 /etc/yum.repos.d/ 下新增 bke.repo（priority=1），不移动现有源文件。
// 对 apt 系统：在 /etc/apt/sources.list.d/ 下新增 bke.list，并在 /etc/apt/preferences.d/
// 下新增高优先级 pin 配置（Pin-Priority: 1001），不修改系统 sources.list。
func SetSource(rawURL string) error {
	baseurl, err := GetRPMDownloadPath(rawURL)
	if err != nil {
		return err
	}

	if strings.Contains(baseurl, "/"+ubuntu+"/") {
		return writeAptSource(baseurl)
	}
	return writeYumSource(baseurl)
}

// writeYumSource 在 /etc/yum.repos.d/bke.repo 写入高优先级 BKE 源，不影响系统已有 repo 文件。
func writeYumSource(baseurl string) error {
	newSource := strings.Replace(yumSource, "{{.baseurl}}", baseurl, -1)
	return os.WriteFile(yumRepos+"/"+bkeRepoFile, []byte(newSource), warehouse.FilePerm)
}

// writeAptSource 在 sources.list.d 下写入 bke.list，同时在 preferences.d 下写入
// 高优先级 pin 配置，不覆盖系统 /etc/apt/sources.list。
func writeAptSource(baseurl string) error {
	newSource := strings.Replace(aptSource, "{{.baseurl}}", baseurl, -1)
	listPath := aptSourcesDir + "/" + bkeListFile
	if err := os.WriteFile(listPath, []byte(newSource), warehouse.FilePerm); err != nil {
		return err
	}

	origin := originFromURL(baseurl)
	newPrefs := strings.Replace(aptPrefs, "{{.origin}}", origin, -1)
	prefsPath := aptPrefsDir + "/" + bkePrefsFile
	return os.WriteFile(prefsPath, []byte(newPrefs), warehouse.FilePerm)
}

// originFromURL 从 URL 中提取 hostname，用于 apt Pin: origin 指令。
func originFromURL(rawURL string) string {
	u, err := url.Parse(rawURL)
	if err != nil || u.Hostname() == "" {
		return rawURL
	}
	return u.Hostname()
}

// ResetSource 移除由 SetSource 添加的 BKE 源配置文件，恢复到注入前状态。
// 系统原有源文件在整个过程中均未被修改，无需还原。
func ResetSource() error {
	h, err := host.Info()
	if err != nil {
		return err
	}
	switch strings.ToLower(h.Platform) {
	case "centos", "kylin", "openeuler", "hopeos", "euleros":
		return resetYumSource()
	case "ubuntu":
		return resetAptSource()
	default:
		return errors.New(fmt.Sprintf("The operating system is not supported %s", h.Platform))
	}
}

// resetYumSource 仅删除 bke.repo，系统原有 repo 文件未被动过，无需恢复。
func resetYumSource() error {
	repoPath := yumRepos + "/" + bkeRepoFile
	if !utils.Exists(repoPath) {
		return nil
	}
	return os.Remove(repoPath)
}

// resetAptSource 删除 bke.list 和 preferences.d/bke，系统 sources.list 未被修改，无需恢复。
func resetAptSource() error {
	listPath := aptSourcesDir + "/" + bkeListFile
	if utils.Exists(listPath) {
		if err := os.Remove(listPath); err != nil {
			return err
		}
	}
	prefsPath := aptPrefsDir + "/" + bkePrefsFile
	if utils.Exists(prefsPath) {
		if err := os.Remove(prefsPath); err != nil {
			return err
		}
	}
	return nil
}
