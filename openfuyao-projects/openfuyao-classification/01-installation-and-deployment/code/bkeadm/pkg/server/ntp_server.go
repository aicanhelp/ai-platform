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

package server

import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v3/process"
	"github.com/zh-five/xdaemon"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var logFile = "/tmp/ntpserver.log"

func DaemonNTPServer() {
	de := xdaemon.NewDaemon(logFile)
	de.MaxCount = 2
	de.Run()
	ntp.Serve(utils.DefaultNTPServerPort)
}

func TryConnectNTPServer(ntpServer string) bool {
	for i := 0; i < 5; i++ {
		time.Sleep(1 * time.Second)
		if err := ntp.Date(ntpServer); err == nil {
			log.Infof("Connection the ntp server %s succeeds.", ntpServer)
			return true
		}
	}
	log.Warnf("Connection the ntp server %s refused.", ntpServer)
	return false
}

func RemoveNTPServer() error {
	var err error
	if utils.Exists("/etc/systemd/system/ntpserver.service") {
		// Remove the ntp server
		err = global.Command.ExecuteCommand("sh", "-c", "sudo systemctl disable ntpserver.service")
		if err != nil {
			log.Warnf("Failed to disable ntp server: %v", err)
		}
		err = global.Command.ExecuteCommand("sh", "-c", "sudo systemctl stop ntpserver.service")
		if err != nil {
			log.Warnf("Failed to stop ntp server: %v", err)
		}
		err = os.Remove("/etc/systemd/system/ntpserver.service")
		if err != nil {
			log.Warnf("Failed to remove systemd service file: %v", err)
		}
		err = global.Command.ExecuteCommand("sh", "-c", "sudo systemctl daemon-reload")
		if err != nil {
			log.Warnf("Failed to daemon-reload ntp server: %v", err)
		}
	}

	pids, err := process.Pids()
	if err != nil {
		return err
	}
	for _, pid := range pids {
		pn, err := process.NewProcess(pid)
		if err != nil {
			continue
		}
		cmd, err := pn.Cmdline()
		if err != nil {
			continue
		}
		if strings.Contains(cmd, "bke start ntpserver") {
			err = syscall.Kill(int(pid), syscall.SIGKILL)
			if err != nil {
				return err
			}
		}
	}
	// Remove the ntp server
	if err := os.Remove(logFile); err != nil && !os.IsNotExist(err) {
		log.Warnf("Failed to remove log file: %v", err)
	}

	log.Info("Remove the ntp server")
	return nil
}

func SystemdNTPServer() {
	var err error
	bin := os.Getenv("bke")
	if len(bin) == 0 {
		bin, err = utils.ExecPath()
		if err != nil {
			log.Errorf("Failed to obtain the execution path %v", err)
			return
		}
	}

	service := `
[Unit]
Description=Network Time Protocol

[Service]
ExecStart=%s start ntpserver --foreground
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
`
	err = os.WriteFile(
		"/etc/systemd/system/ntpserver.service", []byte(fmt.Sprintf(service, bin)), utils.DefaultFilePermission)
	if err != nil {
		log.Errorf("Failed to write systemd service file: %v", err)
		return
	}
	// Enable the ntp server
	err = global.Command.ExecuteCommand("sh", "-c", "sudo systemctl enable ntpserver.service")
	if err != nil {
		log.Errorf("Failed to enable ntp server: %v", err)
		return
	}
	// Reload the ntp server
	err = global.Command.ExecuteCommand("sh", "-c", "sudo systemctl daemon-reload")
	if err != nil {
		log.Errorf("Failed to daemon-reload ntp server: %v", err)
		return
	}
	// Start the ntp server
	err = global.Command.ExecuteCommand("sh", "-c", "sudo systemctl start ntpserver.service")
	if err != nil {
		log.Errorf("Failed to start ntp server: %v", err)
		return
	}
	log.Info("Ntpserver is hosted on systemd")
}

func SystemdDaemonNTPServer() {
	ntp.Serve(utils.DefaultNTPServerPort)
}
