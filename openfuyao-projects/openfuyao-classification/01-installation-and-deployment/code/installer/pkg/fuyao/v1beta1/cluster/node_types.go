/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cluster

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh/terminal"

	"installer/pkg/clients"
	"installer/pkg/constants"
	"installer/pkg/zlog"
)

const (
	clientTimeout    = 40
	retryTimes       = 3
	openEulerVersion = "22.03"
)

var mutex sync.Mutex

// Node is node struct
type Node struct {
	Hostname       string   `yaml:"hostname"`
	User           string   `yaml:"user"`
	Password       []byte   `yaml:"password,omitempty"`
	IP             string   `yaml:"ip"`
	Port           int      `yaml:"port"`
	PrivateKeyPath string   `yaml:"privateKeyPath,omitempty"`
	RemoveTaint    bool     `yaml:"removeTaint,omitempty"`
	Labels         []Labels `yaml:"labels,omitempty"`
	OsName         string   `yaml:"osName,omitempty"`
	OsVersion      string   `yaml:"osVersion,omitempty"`
	CpuArch        string   `yaml:"cpuArch,omitempty"`
}

// Client is node client
func (n *Node) Client() clients.Client {
	if n.PrivateKeyPath == "" && len(n.Password) == 0 {
		n.inputPasswd()
	}
	return n.client()
}

func (n *Node) client() clients.Client {
	client := clients.Client{
		User:           n.User,
		Password:       n.Password,
		PrivateKeyPath: n.PrivateKeyPath,
		Port:           strconv.Itoa(n.Port),
		Host:           n.IP,
		Timeout:        clientTimeout * time.Second,
	}
	return client
}

// ClearPasswd 清除密码
func (n *Node) ClearPasswd() {
	n.Password = []byte("")
}

func (n *Node) InputPasswd() {
	n.inputPasswd()
}

func (n *Node) inputPasswd() {
	// 加锁防止同时输入多个节点密码
	mutex.Lock()
	defer mutex.Unlock()
	fmt.Printf("\nPlease enter the password for %s:", n.IP)

	times := 0
	var err error
	for times < retryTimes {
		n.Password, err = terminal.ReadPassword(int(os.Stdin.Fd()))
		if err != nil {
			fmt.Println("Failed to obtain password:", err)
			fmt.Printf("Please enter the password for %s again:", n.IP)
			times++
			continue
		}

		client := n.client()
		_, err = client.SudoExecCmdNoLog("echo hello")
		if err != nil {
			fmt.Println("\nAccess denied!")
			fmt.Printf("Please enter the password for %s again:", n.IP)
			times++
			continue
		}
		return
	}

	zlog.Fatalf("Failed to obtain password for the %s", n.IP)
}

// OsInfo is os info, cover name version arch
func (n *Node) OsInfo() (string, string, string) {
	if len(n.OsName) != 0 && len(n.OsVersion) != 0 && len(n.CpuArch) != 0 {
		return n.OsName, n.OsVersion, n.CpuArch
	}

	client := n.Client()
	output, err := client.ExecCmd("cat /etc/os-release && uname -m")
	if err != nil {
		zlog.Fatalf("get os info fail ", err)
	}

	namePattern := regexp.MustCompile(`^NAME="(.+?)"`)
	var osName string

	for _, line := range strings.Split(output, "\n") {
		matches := namePattern.FindStringSubmatch(line)

		if len(matches) == 2 {
			osName = matches[1]
			break
		}
	}
	// centos系统名字特殊处理
	if strings.Contains(osName, constants.CentOS) {
		osName = constants.CentOS
	}

	versionIDRegex := regexp.MustCompile(`VERSION_ID="([^"]+)"`)
	osVersion := versionIDRegex.FindStringSubmatch(string(output))[1]

	// 支持EulerOS\CentOS\CULinux\KylinOS系统
	switch osName {
	case constants.EulerOS:
		osVersion = openEulerVersion
		osName = constants.OpenEuler
	case constants.CentOS:
		osVersion = openEulerVersion
		osName = constants.OpenEuler
	case constants.CULinux:
		osVersion = openEulerVersion
		osName = constants.OpenEuler
	case constants.KylinOS:
		osVersion = openEulerVersion
		osName = constants.OpenEuler
	default:
		zlog.Infof("os name %s os version %s", osName, osVersion)
	}

	cpuArch := ""
	if strings.Contains(string(output), constants.X8664) {
		cpuArch = constants.AMD64
	} else if strings.Contains(string(output), constants.AARCH64) {
		cpuArch = constants.ARM64
	} else {
		zlog.Fatalf("CPU architecture information not obtained")
	}

	n.OsName, n.CpuArch, n.OsVersion = osName, cpuArch, osVersion
	return osName, osVersion, cpuArch
}

// Labels 标签
type Labels struct {
	Name  string `yaml:"name"`
	Value string `yaml:"value"`
}

// Nodes 节点规划部分
type Nodes struct {
	Masters  []*Node `yaml:"masters"`
	Workers  []*Node `yaml:"workers,omitempty"`
	Registry []*Node `yaml:"registry,omitempty"`
}

// GetAllNodes 获取所有节点
func (n *Nodes) GetAllNodes() []*Node {
	allNodes := make([]*Node, 0)
	allNodes = append(allNodes, n.Masters...)
	allNodes = append(allNodes, n.Workers...)
	allNodes = append(allNodes, n.Registry...)
	return allNodes
}

// ClearNodesPasswd 清除节点密码
func (n *Nodes) ClearNodesPasswd() {
	for _, node := range n.GetAllNodes() {
		node.ClearPasswd()
	}
}
