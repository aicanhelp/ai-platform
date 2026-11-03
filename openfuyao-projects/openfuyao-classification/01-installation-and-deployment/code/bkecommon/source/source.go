/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
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
	"os"
	"runtime"
	"strings"

	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/host"

	"gopkg.openfuyao.cn/bkecommon/utils"
	"gopkg.openfuyao.cn/bkecommon/warehouse"
)

// Constants for version parsing
const (
	minimumVersionParts = 2
)

var (
	yumRepos = "/etc/yum.repos.d"
	aptRepos = "/etc/apt/sources.list"

	centos    = "CentOS"
	kylin     = "Kylin"
	ubuntu    = "Ubuntu"
	openeuler = "OpenEuler"
	hopeos    = "HopeOS"
	euleros   = "EulerOS"

	yumSource = `
[bke]
baseurl = {{.baseurl}}
enabled = 1
gpgcheck = 0
name = repo
`

	aptSource = `
deb [trusted=yes] {{.baseurl}} ./
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

func SetSource(url string) error {
	baseurl, err := GetRPMDownloadPath(url)
	if err != nil {
		return err
	}

	if strings.Contains(baseurl, "Ubuntu") {
		err = backupAptSource()
		if err != nil {
			return err
		}
		err = writeAptSource(baseurl)
		if err != nil {
			return err
		}
		return nil
	}

	err = backupYumSource()
	if err != nil {
		return err
	}
	err = writeYumSource(baseurl)
	if err != nil {
		return err
	}
	return nil
}

func backupYumSource() error {
	files, err := os.ReadDir(yumRepos)
	if err != nil {
		return err
	}
	bak := yumRepos + "/bak"
	if !utils.Exists(bak) {
		err = os.Mkdir(bak, warehouse.DirPerm)
		if err != nil {
			return err
		}
	}
	for _, f := range files {
		if f.Name() == "bak" {
			continue
		}
		err = os.Rename(yumRepos+"/"+f.Name(), bak+"/"+f.Name())
		if err != nil {
			return err
		}
	}
	return nil
}

func backupAptSource() error {
	return os.Rename(aptRepos, aptRepos+".bak")
}

func writeYumSource(baseurl string) error {
	newSource := strings.Replace(yumSource, "{{.baseurl}}", baseurl, -1)
	return os.WriteFile(yumRepos+"/bke.repo", []byte(newSource), warehouse.FilePerm)
}

func writeAptSource(baseurl string) error {
	newSource := strings.Replace(aptSource, "{{.baseurl}}", baseurl, -1)
	return os.WriteFile(aptRepos, []byte(newSource), warehouse.FilePerm)
}

func ResetSource() error {
	h, err := host.Info()
	if err != nil {
		return err
	}
	switch strings.ToLower(h.Platform) {
	case "centos":
		return resetYumSource()
	case "kylin":
		return resetYumSource()
	case "ubuntu":
		return resetAptSource()
	case "openeuler":
		return resetYumSource()
	default:
		return errors.New(fmt.Sprintf("The operating system is not supported %s", h.Platform))
	}
}

func resetYumSource() error {
	bak := yumRepos + "/bak"
	if !utils.Exists(bak) {
		return nil
	}
	if !utils.Exists(yumRepos + "/bke.repo") {
		return nil
	}
	err := os.Remove(yumRepos + "/bke.repo")
	if err != nil {
		return err
	}
	files, err := os.ReadDir(bak)
	if err != nil {
		return err
	}
	for _, f := range files {
		if f.Name() == "bke.repo" {
			continue
		}
		if f.IsDir() {
			continue
		}
		err = os.Rename(bak+"/"+f.Name(), yumRepos+"/"+f.Name())
		if err != nil {
			return err
		}
	}
	err = os.RemoveAll(bak)
	if err != nil {
		return err
	}
	return nil
}

func resetAptSource() error {
	if !utils.Exists(aptRepos + ".bak") {
		return nil
	}
	err := os.Remove(aptRepos)
	if err != nil {
		return err
	}
	err = os.Rename(aptRepos+".bak", aptRepos)
	if err != nil {
		return err
	}
	err = os.Remove(aptRepos + ".bak")
	if err != nil {
		return err
	}
	return nil
}
