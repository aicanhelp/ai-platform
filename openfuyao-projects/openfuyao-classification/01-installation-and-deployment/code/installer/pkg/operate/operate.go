/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package operate

import (
	"fmt"
	"net"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/utils/cache"
	"installer/pkg/zlog"
)

// NodesOperate 节点操作器
type NodesOperate struct {
	Cmd   []string
	Tools []string
	Nodes cluster.Nodes

	// 缓存，用于记录操作结果
	Cache          *cache.Cache
	RoutineMaximum *utils.GoRoutineLimit
}

// NewNodesOperate is new NodesOperate
func NewNodesOperate(nodes cluster.Nodes) *NodesOperate {
	return &NodesOperate{
		Nodes:          nodes,
		Cache:          cache.NewCache(),
		RoutineMaximum: utils.NewGoRoutineLimits(),
	}
}

// AllNodeExecOperate 所有节点执行操作
func (n *NodesOperate) AllNodeExecOperate(function func(*cluster.Node, *sync.WaitGroup)) {
	var wg sync.WaitGroup

	funcName := runtime.FuncForPC(reflect.ValueOf(function).Pointer()).Name()
	nodes := n.Nodes.GetAllNodes()
	for _, node := range nodes {
		wg.Add(1)
		n.RoutineMaximum.Add()
		zlog.Debugf("exec %s on %s", funcName, node.IP)
		go function(node, &wg)
		zlog.Debugf("finish %s on %s", funcName, node.IP)
	}

	wg.Wait()
}

// MasterNodeExecOperate master节点执行操作
func (n *NodesOperate) MasterNodeExecOperate(function func(*cluster.Node, *sync.WaitGroup)) {
	var wg sync.WaitGroup
	for _, master := range n.Nodes.Masters {
		wg.Add(1)
		n.RoutineMaximum.Add()
		go function(master, &wg)
	}
	wg.Wait()
}

// WorkerNodeExecOperate worker节点执行操作
func (n *NodesOperate) WorkerNodeExecOperate(function func(*cluster.Node, *sync.WaitGroup)) {
	var wg sync.WaitGroup
	for _, worker := range n.Nodes.Workers {
		wg.Add(1)
		n.RoutineMaximum.Add()
		go function(worker, &wg)
	}
	wg.Wait()
}

// RegistryNodeExecOperate 镜像节点执行操作
func (n *NodesOperate) RegistryNodeExecOperate(function func(*cluster.Node, *sync.WaitGroup)) {
	var wg sync.WaitGroup
	for _, registry := range n.Nodes.Registry {
		wg.Add(1)
		n.RoutineMaximum.Add()
		go function(registry, &wg)
	}
	wg.Wait()
}

// GetMemorySize 获取节点内存大小
func (n *NodesOperate) GetMemorySize(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	client := node.Client()
	sizeStr, err := client.ExecCmd("cat /proc/meminfo | awk 'NR==1 {print $2}'")
	if err != nil {
		zlog.Errorf("failed to obtain memory size:%v", err)
	}

	size, err := strconv.ParseFloat(strings.TrimSuffix(sizeStr, "\r\n"), constants.FloatBitSizeMax)
	if err != nil {
		zlog.Errorf("Error parsing float: %v", err)
		size = 0
	}

	n.Cache.Set(node.IP, size)
}

// GetOSName 获取操作系统名字
func (n *NodesOperate) GetOSName(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	name, _, _ := node.OsInfo()
	n.Cache.Set(node.IP, name)
}

// GetTime 获取系统时间
func (n *NodesOperate) GetTime(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	client := node.Client()
	output, err := client.ExecCmd("date +%s")
	if err != nil {
		zlog.Errorf("Failed to obtain os time on %s error is %v", node.IP, err)
		output = ""
	}
	n.Cache.Set(node.IP, output)
}

// GetCPUCount 获取cpu数量并记录
func (n *NodesOperate) GetCPUCount(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	client := node.Client()
	cpuCountStr, err := client.SudoExecCmd("nproc")
	if err != nil {
		zlog.Error(fmt.Sprintf("Failed to obtain cpu member:%v", err))
	}

	cpuCount, err := strconv.Atoi(strings.TrimSuffix(cpuCountStr, "\r\n"))
	if err != nil {
		zlog.Error(fmt.Sprintf("Integer conversion failure: %v and cpuCountStr is %s", err, cpuCountStr))
		cpuCount = 0
	}

	n.Cache.Set(node.IP, cpuCount)
}

// InstallTools 安装对应工具，并记录安装结果
func (n *NodesOperate) InstallTools(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	client := node.Client()

	baseCmd := constants.AptGet
	_, err := client.ExecCmd(fmt.Sprintf("command -v %s", baseCmd))
	if err != nil {
		zlog.Infof("exec command 'command -v %s', error: %v", baseCmd, err)
		baseCmd = constants.Yum
	}

	installStatus := make(map[string]bool)
	for _, tool := range n.Tools {
		_, err = client.SudoExecCmd(fmt.Sprintf(`[ -x "$(which %s)" ] || %s install %s -y`, tool, baseCmd,
			tool))
		if err != nil {
			zlog.Errorf("failed to installed %s on %s", tool, node.IP)
			installStatus[tool] = false
			continue
		}
		installStatus[tool] = true
	}
	n.Cache.Set(node.IP, installStatus)
}

// ExecCmd 执行cmd命令
func (n *NodesOperate) ExecCmd(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	client := node.Client()

	execResult := make(map[string]map[string]any)
	for _, cmd := range n.Cmd {
		outputStr, err := client.SudoExecCmd(cmd)
		var res = make(map[string]any)
		res[output] = outputStr
		if err != nil {
			zlog.Error(fmt.Sprintf("Failed to exec %s on %s", cmd, node.IP))
			res[success] = false
			execResult[cmd] = res
			continue
		}
		res[success] = true
		execResult[cmd] = res
	}
	n.Cache.Set(node.IP, execResult)
}

// Ping is ping node
func (n *NodesOperate) Ping(node *cluster.Node, wg *sync.WaitGroup) {
	defer wg.Done()
	defer n.RoutineMaximum.Done()

	timeout := time.Second * pingTimeOut
	conn, err := net.DialTimeout("ip4:icmp", node.IP, timeout)
	defer conn.Close()

	if err != nil {
		zlog.Info(fmt.Sprintf("Failed to ping %s : %v", node.IP, err))
		n.Cache.Set(node.IP, false)
	} else {
		n.Cache.Set(node.IP, true)
	}
}
