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

package precheck

import (
	"fmt"
	"strings"
	"time"

	"installer/pkg/constants"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/operate"
	"installer/pkg/zlog"
)

// FlightCheck 飞行检查
type FlightCheck struct {
	Nodes        fuyaov1beta1.Nodes
	NodesOperate *operate.NodesOperate
}

// NewFlightCheck is new FlightCheck
func NewFlightCheck(nodes fuyaov1beta1.Nodes) *FlightCheck {
	return &FlightCheck{
		Nodes:        nodes,
		NodesOperate: operate.NewNodesOperate(nodes),
	}
}

func (f *FlightCheck) cpu(nodes []*fuyaov1beta1.Node, cpuStandard int) bool {
	isSatisfy := true
	for _, node := range nodes {
		ip := node.IP
		GetCPUCount, ok := f.NodesOperate.Cache.Get(ip)
		if !ok {
			zlog.Error(fmt.Sprintf("Failed to obtain the number of cpus: %s", ip))
			GetCPUCount = 0
		}

		if GetCPUCount.(int) < cpuStandard {
			isSatisfy = false
			zlog.Error(fmt.Sprintf("A minimum of %d cpus are required. The number of cpus for %s nodes is %d",
				cpuStandard, ip, GetCPUCount))
		}
	}
	return isSatisfy
}

// allNodeCupCheck 检查CPU资源是否满足
func (f *FlightCheck) allNodeCupCheck(cpuStandard int) error {
	f.NodesOperate.AllNodeExecOperate(f.NodesOperate.GetCPUCount)
	isMasterSatisfy := f.cpu(f.Nodes.Masters, cpuStandard)
	isWorkerSatisfy := f.cpu(f.Nodes.Workers, cpuStandard)
	isRegistrySatisfy := f.cpu(f.Nodes.Registry, cpuStandard)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy || !isWorkerSatisfy || !isRegistrySatisfy {
		return fmt.Errorf("the number of cpus is insufficient")
	}
	return nil
}

// masterNodesCupCheck 检查CPU资源是否满足
func (f *FlightCheck) masterNodesCupCheck(cpuStandard int) error {
	f.NodesOperate.MasterNodeExecOperate(f.NodesOperate.GetCPUCount)
	isMasterSatisfy := f.cpu(f.Nodes.Masters, cpuStandard)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy {
		return fmt.Errorf("the number of cpus is insufficient")
	}
	return nil
}

// workerNodesCupCheck 检查CPU资源是否满足
func (f *FlightCheck) workerNodesCupCheck(cpuStandard int) error {
	f.NodesOperate.WorkerNodeExecOperate(f.NodesOperate.GetCPUCount)
	isWorkerSatisfy := f.cpu(f.Nodes.Workers, cpuStandard)

	f.NodesOperate.Cache.Clear()
	if !isWorkerSatisfy {
		return fmt.Errorf("the number of cpus is insufficient")
	}
	return nil
}

// registryNodesCupCheck 检查CPU资源是否满足
func (f *FlightCheck) registryNodesCupCheck(cpuStandard int) error {
	f.NodesOperate.RegistryNodeExecOperate(f.NodesOperate.GetCPUCount)
	isRegistrySatisfy := f.cpu(f.Nodes.Registry, cpuStandard)

	f.NodesOperate.Cache.Clear()
	if !isRegistrySatisfy {
		return fmt.Errorf("the number of cpus is insufficient")
	}
	return nil
}

func (f *FlightCheck) memory(nodes []*fuyaov1beta1.Node, memoryStandard int) bool {
	isSatisfy := true
	// 标准内存 * 0.9 防止出现部分组件占用内存较多的情况
	memoryFloat := float64(memoryStandard) * constants.MemoryRatio

	for _, node := range nodes {
		ip := node.IP
		memorySize, ok := f.NodesOperate.Cache.Get(ip)
		if !ok {
			zlog.Errorf("failed to obtain memory size: %s", ip)
			memorySize = 0
		}

		memoryGB := memorySize.(float64) / (constants.MemoryConversionNum * constants.MemoryConversionNum)
		if memoryGB < memoryFloat {
			isSatisfy = false
			zlog.Errorf("The minimum memory size is %dGB, and the memory size of the %s node is %dGB",
				memoryStandard, ip, memoryGB)
		}
	}
	return isSatisfy
}

// memoryCheck 检查内存资源是否满足
func (f *FlightCheck) allNodeMemoryCheck(memoryStandard int) error {
	f.NodesOperate.AllNodeExecOperate(f.NodesOperate.GetMemorySize)
	isMasterSatisfy := f.memory(f.Nodes.Masters, memoryStandard)
	isWorkerSatisfy := f.memory(f.Nodes.Workers, memoryStandard)
	isRegistrySatisfy := f.memory(f.Nodes.Registry, memoryStandard)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy || !isWorkerSatisfy || !isRegistrySatisfy {
		return fmt.Errorf("the memory size is not sufficient")
	}
	return nil
}

// masterNodesMemoryCheck 检查内存资源是否满足
func (f *FlightCheck) masterNodesMemoryCheck(memoryStandard int) error {
	f.NodesOperate.MasterNodeExecOperate(f.NodesOperate.GetMemorySize)
	isMasterSatisfy := f.memory(f.Nodes.Masters, memoryStandard)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy {
		return fmt.Errorf("the memory size is not sufficient")
	}
	return nil
}

// workerNodesMemoryCheck 检查内存资源是否满足
func (f *FlightCheck) workerNodesMemoryCheck(memoryStandard int) error {
	f.NodesOperate.WorkerNodeExecOperate(f.NodesOperate.GetMemorySize)
	isWorkerSatisfy := f.memory(f.Nodes.Workers, memoryStandard)

	f.NodesOperate.Cache.Clear()
	if !isWorkerSatisfy {
		return fmt.Errorf("the memory size is not sufficient")
	}
	return nil
}

// memoryCheck 检查内存资源是否满足
func (f *FlightCheck) registryNodesMemoryCheck(memoryStandard int) error {
	f.NodesOperate.RegistryNodeExecOperate(f.NodesOperate.GetMemorySize)
	isRegistrySatisfy := f.memory(f.Nodes.Registry, memoryStandard)

	f.NodesOperate.Cache.Clear()
	if !isRegistrySatisfy {
		return fmt.Errorf("the memory size is not sufficient")
	}
	return nil
}

// tools 检查工具是否安装
func (f *FlightCheck) tools(nodes []*fuyaov1beta1.Node) bool {
	for _, node := range nodes {
		ip := node.IP
		recordMap, ok := f.NodesOperate.Cache.Get(ip)
		if !ok {
			zlog.Error(fmt.Sprintf("Failed to obtain install record: %s", ip))
			return false
		}
		for toolName, isInstall := range recordMap.(map[string]bool) {
			if !isInstall {
				fmt.Printf("The %s tool is not installed on %s, but we depend on it\n", toolName, ip)
				return false
			}
		}
	}
	return true
}

// allNodeToolCheck 检查基础工具是否全部安装
func (f *FlightCheck) allNodeToolCheck() error {
	f.NodesOperate.AllNodeExecOperate(f.NodesOperate.InstallTools)
	isMasterSatisfy := f.tools(f.Nodes.Masters)
	isWorkerSatisfy := f.tools(f.Nodes.Workers)
	isRegistrySatisfy := f.tools(f.Nodes.Registry)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy || !isWorkerSatisfy || !isRegistrySatisfy {
		return fmt.Errorf("the required foundational tools are missing. Please install them manually")
	}
	return nil
}

// masterNodesToolCheck 检查基础工具是否全部安装
func (f *FlightCheck) masterNodesToolCheck() error {
	f.NodesOperate.MasterNodeExecOperate(f.NodesOperate.InstallTools)
	isMasterSatisfy := f.tools(f.Nodes.Masters)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy {
		return fmt.Errorf("the required foundational tools are missing. Please install them manually")
	}
	return nil
}

// workerNodesToolCheck 检查基础工具是否全部安装
func (f *FlightCheck) workerNodesToolCheck() error {
	f.NodesOperate.WorkerNodeExecOperate(f.NodesOperate.InstallTools)
	isWorkerSatisfy := f.tools(f.Nodes.Workers)

	f.NodesOperate.Cache.Clear()
	if !isWorkerSatisfy {
		return fmt.Errorf("the required foundational tools are missing. Please install them manually")
	}
	return nil
}

// registryNodesToolCheck 检查基础工具是否全部安装
func (f *FlightCheck) registryNodesToolCheck() error {
	f.NodesOperate.RegistryNodeExecOperate(f.NodesOperate.InstallTools)
	isRegistrySatisfy := f.tools(f.Nodes.Registry)

	f.NodesOperate.Cache.Clear()
	if !isRegistrySatisfy {
		return fmt.Errorf("the required foundational tools are missing. Please install them manually")
	}
	return nil
}

func (f *FlightCheck) isNodeOsNameSatisfy(osName interface{}, nodes []*fuyaov1beta1.Node) bool {
	isSatisfy := true
	for _, node := range nodes {
		zlog.Debugf("check %s os name", node.IP)
		srcName, ok := f.NodesOperate.Cache.Get(node.IP)
		if !ok {
			zlog.Errorf("failed to obtained OS name of node %s", node.IP)
			isSatisfy = false
			continue
		}
		if srcName != osName {
			isSatisfy = false
			zlog.Errorf("%s Node OS name is %s", node.IP, srcName)
		}
	}
	return isSatisfy
}

// checkAllNodesOsName 检查所有节点操作系统名是否一致
func (f *FlightCheck) checkAllNodesOsName() error {
	f.NodesOperate.AllNodeExecOperate(f.NodesOperate.GetOSName)

	osName, ok := f.NodesOperate.Cache.Get(f.Nodes.Masters[0].IP)
	if !ok {
		zlog.Errorf("the OS name of node %s is not obtained", f.Nodes.Masters[0].IP)
		f.NodesOperate.Cache.Clear()
		return fmt.Errorf("failed to get node %s OS name", f.Nodes.Masters[0].IP)
	}

	isMasterSatisfy := f.isNodeOsNameSatisfy(osName, f.Nodes.Masters[1:])
	isWorkerSatisfy := f.isNodeOsNameSatisfy(osName, f.Nodes.Workers)
	isRegistrySatisfy := f.isNodeOsNameSatisfy(osName, f.Nodes.Registry)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy || !isWorkerSatisfy || !isRegistrySatisfy {
		return fmt.Errorf("cluster memory Different types of operating systems")
	}
	return nil
}

// checkALLNodesTimeDiff 检查所有节点时间差
func (f *FlightCheck) checkALLNodesTimeDiff() error {
	f.NodesOperate.AllNodeExecOperate(f.NodesOperate.GetTime)

	nodes := append(f.Nodes.Masters, f.Nodes.Workers...)
	nodes = append(nodes, f.Nodes.Registry...)

	var times []time.Time
	for _, node := range nodes {
		remoteTimeStr, ok := f.NodesOperate.Cache.Get(node.IP)
		if !ok {
			return fmt.Errorf("failed to obtain time of node %s", node.IP)
		}

		var remoteTime int64
		if remoteTime, ok = remoteTimeStr.(int64); !ok {
			remoteTime = 0
		}
		times = append(times, time.Unix(remoteTime, 0))
	}

	// Allow a time difference of 3 seconds
	const allowedDifference = 3 * time.Second

	isSatisfy := true
	if len(times) > 1 {
		for i := 1; i < len(times); i++ {
			diff := times[0].Sub(times[i])
			if diff < 0 {
				diff = -diff
			}
			if diff > allowedDifference {
				zlog.Warnf("Times are different: %v vs %v (difference: %v)", times[0], times[i], diff)
				isSatisfy = false
			}
		}
	}

	if !isSatisfy {
		fmt.Println("[warning] If the time difference between different machines is greater than 3 seconds," +
			"it may cause some services to be unavailable, such as monitoring and alarms")
	}
	return nil
}

func (f *FlightCheck) checkWorkerCPUMemory() error {
	cpuStandard, memoryStandard := 4, 8
	err := f.workerNodesCupCheck(cpuStandard)
	if err != nil {
		return err
	}

	err = f.workerNodesMemoryCheck(memoryStandard)
	if err != nil {
		return err
	}
	return nil
}

func (f *FlightCheck) pingCheck(nodes []*fuyaov1beta1.Node) bool {
	isSatisfy := true
	for _, node := range nodes {
		ip := node.IP
		res, ok := f.NodesOperate.Cache.Get(ip)
		if !ok {
			zlog.Error(fmt.Sprintf("Failed to obtain ping result: %s", ip))
		}

		if !res.(bool) {
			isSatisfy = false
			zlog.Error(fmt.Sprintf("Failed to ping: %s", ip))
		}
	}
	return isSatisfy
}

// allNodePingCheck ping all node
func (f *FlightCheck) allNodePingCheck() error {
	f.NodesOperate.AllNodeExecOperate(f.NodesOperate.Ping)
	isMasterSatisfy := f.pingCheck(f.Nodes.Masters)
	isWorkerSatisfy := f.pingCheck(f.Nodes.Workers)
	isRegistrySatisfy := f.pingCheck(f.Nodes.Registry)

	f.NodesOperate.Cache.Clear()
	if !isMasterSatisfy || !isWorkerSatisfy || !isRegistrySatisfy {
		return fmt.Errorf("some nodes have abnormal network links")
	}
	return nil
}

func (f *FlightCheck) checkOrSetHostName(nodes []*fuyaov1beta1.Node, hostnameMap map[string]string) error {
	if hostnameMap == nil {
		hostnameMap = make(map[string]string)
	}

	for _, node := range nodes {
		if node.Hostname == "" {
			client := node.Client()
			output, err := client.ExecCmd("hostname")
			if err != nil {
				return err
			}
			node.Hostname = strings.ToLower(strings.TrimSuffix(output, "\r\n"))
		}
		node.Hostname = strings.ToLower(node.Hostname)

		ip, ok := hostnameMap[node.Hostname]
		if ok {
			return fmt.Errorf("node %s has the same hostname as node %s", ip, node.IP)
		}
		hostnameMap[node.Hostname] = node.IP
	}
	return nil
}

func (f *FlightCheck) isHasSameIP(nodes []*fuyaov1beta1.Node, ipMap map[string]bool) error {
	if ipMap == nil {
		ipMap = make(map[string]bool)
	}

	for _, node := range nodes {
		_, ok := ipMap[node.IP]
		if ok {
			return fmt.Errorf("have the same ip address. The same ip address is %s", node.IP)
		}
		ipMap[node.IP] = true
	}
	return nil
}

// computeMasterNodeResource 根据worker节点数量计算master节点CPU与内存资源
func computeMasterNodeResource(workerCount int) (int, int) {
	// The memory unit is BG
	cpuStandard, memoryStandard := 4, 8
	switch {
	case workerCount <= OneLevelClusterWorkerCount:
		cpuStandard, memoryStandard = 4, 8
	case workerCount > OneLevelClusterWorkerCount && workerCount <= TwoLevelClusterWorkerCount:
		cpuStandard, memoryStandard = 4, 16
	case workerCount > TwoLevelClusterWorkerCount && workerCount <= ThreeLevelClusterWorkerCount:
		cpuStandard, memoryStandard = 8, 32
	case workerCount > ThreeLevelClusterWorkerCount && workerCount <= FourLevelClusterWorkerCount:
		cpuStandard, memoryStandard = 16, 64
	case workerCount > FourLevelClusterWorkerCount && workerCount <= FiveLevelClusterWorkerCount:
		cpuStandard, memoryStandard = 24, 64
	default:
		cpuStandard, memoryStandard = 36, 128
	}
	return cpuStandard, memoryStandard
}
