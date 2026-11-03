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

package timezone

import (
	_ "embed"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/zh-five/xdaemon"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	localtimePath = "/etc/localtime"
	timezone      = "Asia/Shanghai"
	timezonePath  = "/etc/timezone"
	zoneDirs      = []string{
		"/usr/share/zoneinfo",
		"/usr/lib/zoneinfo",
		"/usr/local/share/zoneinfo",
		"/system/usr/share/zoneinfo",
	}
)

// 新增时区状态目录和备份文件名：timezone.marker、localtime.bkeadm.bak、localtime.bkeadm.symlink、timezone.bkeadm.bak。
const (
	bkeadmStateDirName     = "bkeadm"
	timezoneMarkerFileName = "timezone.marker"
	localtimeBackupName    = "localtime.bkeadm.bak"
	localtimeSymlinkName   = "localtime.bkeadm.symlink"
	timezoneBackupName     = "timezone.bkeadm.bak"
)

func bkeadmTimezoneStateDir() string {
	return filepath.Join(global.Workspace, bkeadmStateDirName)
}

func timezoneMarkerPath() string {
	return filepath.Join(bkeadmTimezoneStateDir(), timezoneMarkerFileName)
}

func localtimeBackupPath() string {
	return filepath.Join(bkeadmTimezoneStateDir(), localtimeBackupName)
}

func localtimeSymlinkPath() string {
	return filepath.Join(bkeadmTimezoneStateDir(), localtimeSymlinkName)
}

func timezoneBackupPath() string {
	return filepath.Join(bkeadmTimezoneStateDir(), timezoneBackupName)
}

func findTimezoneFile(zone string) (string, error) {
	for _, dir := range zoneDirs {
		path := filepath.Join(dir, zone)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("timezone file not found for %s", zone)
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, utils.DefaultFilePermission)
}

func backupTimezoneFile(src, dst string) error {
	if !utils.Exists(src) || utils.Exists(dst) {
		return nil
	}
	return copyFile(src, dst)
}

// bke init 设置时区前备份原始 /etc/localtime：原来是普通文件就复制备份；原来是软链接就记录 symlink target，后续恢复时重建软链接。
func backupLocaltime() error {
	if !utils.Exists(localtimePath) || utils.Exists(localtimeBackupPath()) || utils.Exists(localtimeSymlinkPath()) {
		return nil
	}

	info, err := os.Lstat(localtimePath)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return copyFile(localtimePath, localtimeBackupPath())
	}

	target, err := os.Readlink(localtimePath)
	if err != nil {
		return err
	}
	return os.WriteFile(localtimeSymlinkPath(), []byte(target), utils.DefaultFilePermission)
}

// 备份 /etc/timezone，并写入 marker。没有原始 /etc/timezone 时不会创建备份，reset 时会删除 init 写入的 /etc/timezone。
func backupTimezoneState() error {
	stateDir := bkeadmTimezoneStateDir()
	if err := os.MkdirAll(stateDir, utils.DefaultDirPermission); err != nil {
		return fmt.Errorf("create timezone backup dir failed: %w", err)
	}

	if err := backupLocaltime(); err != nil {
		return fmt.Errorf("backup localtime failed: %w", err)
	}
	if err := backupTimezoneFile(timezonePath, timezoneBackupPath()); err != nil {
		return fmt.Errorf("backup timezone failed: %w", err)
	}

	marker := fmt.Sprintf("timezone=%s\nlocaltime_backup=%s\nlocaltime_symlink=%s\ntimezone_backup=%s\n",
		timezone, localtimeBackupPath(), localtimeSymlinkPath(), timezoneBackupPath())
	if err := os.WriteFile(timezoneMarkerPath(), []byte(marker), utils.DefaultFilePermission); err != nil {
		return fmt.Errorf("write timezone marker failed: %w", err)
	}
	return nil
}

func removeExistingTimezoneFile(path string) error {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// bke init 设置时区前先写入 /bke/bkeadm 下的 marker 和备份，供 bke reset --all --mount 恢复。
func SetTimeZone() error {
	if utils.Exists(timezonePath) {
		tz, err := os.ReadFile(timezonePath)
		if err != nil {
			return err
		}
		if string(tz) == timezone {
			return nil
		}
	}

	zoneInfoPath, err := findTimezoneFile(timezone)
	if err != nil {
		return err
	}

	if err = backupTimezoneState(); err != nil {
		return err
	}

	if err = removeExistingTimezoneFile(localtimePath); err != nil {
		return fmt.Errorf("remove localtime failed: %w", err)
	}

	if err = os.Symlink(zoneInfoPath, localtimePath); err == nil {
		return os.WriteFile(timezonePath, []byte(timezone), utils.DefaultFilePermission)
	}

	if err = copyFile(zoneInfoPath, localtimePath); err != nil {
		return fmt.Errorf("sym link and copy file both fail: %v", err)
	}

	return os.WriteFile(timezonePath, []byte(timezone), utils.DefaultFilePermission)
}

func restoreLocaltime() {
	if utils.Exists(localtimeSymlinkPath()) {
		target, err := os.ReadFile(localtimeSymlinkPath())
		if err != nil {
			log.Warnf("read localtime symlink backup failed: %v", err)
			return
		}
		if err = removeExistingTimezoneFile(localtimePath); err != nil {
			log.Warnf("remove current localtime failed: %v", err)
			return
		}
		if err = os.Symlink(string(target), localtimePath); err != nil {
			log.Warnf("restore localtime symlink failed: %v", err)
		}
		return
	}
	if utils.Exists(localtimeBackupPath()) {
		if err := removeExistingTimezoneFile(localtimePath); err != nil {
			log.Warnf("remove current localtime failed: %v", err)
			return
		}
		if err := copyFile(localtimeBackupPath(), localtimePath); err != nil {
			log.Warnf("restore localtime failed: %v", err)
		}
	}
}

// 只有 marker 存在才恢复，优先恢复 /etc/localtime 原软链接，否则恢复普通文件备份；/etc/timezone 有备份则恢复，没有备份则删除。
func RestoreTimeZone() {
	if !utils.Exists(timezoneMarkerPath()) {
		return
	}

	restoreLocaltime()

	if utils.Exists(timezoneBackupPath()) {
		if err := removeExistingTimezoneFile(timezonePath); err != nil {
			log.Warnf("remove current timezone failed: %v", err)
			return
		}
		if err := copyFile(timezoneBackupPath(), timezonePath); err != nil {
			log.Warnf("restore timezone failed: %v", err)
		}
		return
	}
	if err := removeExistingTimezoneFile(timezonePath); err != nil {
		log.Warnf("remove timezone failed: %v", err)
	}
}

func NTPServer(ntpServer, hostIP string, online bool) (string, error) {
	log.Infof("Starting NTP sync with %s", ntpServer)
	start := time.Now()

	if ntpServer != utils.LocalNTPName {
		server, err := tryConnectExternalNTP(ntpServer, online)
		if err == nil {
			log.Infof("NTP sync completed with %s, system time adjusted, elapsed=%s",
				server, time.Since(start).Round(time.Second))
			return server, nil
		}
		if online {
			return "", err
		}
	}

	if ntpServer != utils.LocalNTPName && ntpServer != configinit.DefaultNTPServer {
		return "", errors.New(fmt.Sprintf("ntp server %s is unreachable", ntpServer))
	}

	localServer, err := startLocalNTPServer(hostIP)
	if err != nil {
		return "", err
	}
	log.Infof("NTP sync completed with local server %s, elapsed=%s",
		localServer, time.Since(start).Round(time.Second))
	return localServer, nil
}

// tryConnectExternalNTP 尝试连接外部 NTP 服务器
func tryConnectExternalNTP(ntpServer string, online bool) (string, error) {
	if err := ntp.Date(ntpServer); err == nil {
		return ntpServer, nil
	}

	if !online {
		return "", errors.New(fmt.Sprintf("ntp server %s is unreachable", ntpServer))
	}

	return retryNTPConnection(ntpServer)
}

// retryNTPConnection 重试 NTP 连接
func retryNTPConnection(ntpServer string) (string, error) {
	const maxRetries = 2
	for i := 0; i < maxRetries; i++ {
		time.Sleep(utils.DefaultSleepSeconds * time.Second)
		if err := ntp.Date(ntpServer); err == nil {
			return ntpServer, nil
		}
	}
	return "", errors.New(fmt.Sprintf("Failed to connect to the ntp server %s", ntpServer))
}

// startLocalNTPServer 启动本地 NTP 服务器
func startLocalNTPServer(hostIP string) (string, error) {
	bin, err := utils.ExecPath()
	if err != nil {
		return "", errors.New(fmt.Sprintf("Failed to obtain the execution path %v", err))
	}

	env := os.Environ()
	env = append(env, "bke="+bin)
	cmd := &exec.Cmd{
		Path:        bin,
		Args:        []string{bin, "start", "ntpserver", "--systemd"},
		Env:         env,
		SysProcAttr: xdaemon.NewSysProcAttr(),
	}
	if err = cmd.Start(); err != nil {
		return "", err
	}

	localAddr := fmt.Sprintf("127.0.0.1:%d", utils.DefaultNTPServerPort)
	if !server.TryConnectNTPServer(localAddr) {
		return "", errors.New(fmt.Sprintf("Failed to connect to the ntp server %s", localAddr))
	}
	return fmt.Sprintf("%s:%d", hostIP, utils.DefaultNTPServerPort), nil
}
