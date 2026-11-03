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

package executor

import (
	"fmt"
	"installer/pkg/fuyao/v1beta1/manifests"
	"path"
	"path/filepath"
	"sync"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor/auth"
	"installer/pkg/executor/core"
	"installer/pkg/executor/join"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

var mutex sync.Mutex

// Executor is an executor
type Executor struct {
	Operate string
	Graph   *core.DirectedGraph
	Nodes   fuyaov1beta1.Nodes
	// InitCfgFile is init k8s config file
	InitCfgFile string
	// SkipDestroyHub is skip remove hub
	SkipDestroyHub bool
	// HubComponentMap is the component required for installing the hub
	HubComponentMap map[string]string

	installComponentManifest *manifests.InstallComponentManifest
	routineMaximum           *utils.GoRoutineLimit
}

// NewExecutor is new an Executor
func NewExecutor(operate string, nodes fuyaov1beta1.Nodes, graph *core.DirectedGraph) Executor {
	return Executor{
		Operate:                  operate,
		Nodes:                    nodes,
		Graph:                    graph,
		routineMaximum:           utils.NewGoRoutineLimits(),
		installComponentManifest: manifests.NewInstallComponentManifest(),
	}
}

// componentOperateGroup 组件操作分组
func (e *Executor) componentOperateGroup(vertexList []*core.Vertex) map[string][]fuyaov1beta1.Component {
	group := make(map[string][]fuyaov1beta1.Component)

	for _, vertex := range vertexList {
		zlog.Debugf("The grouping component is %s", vertex.Data.Name)
		switch vertex.Data.BootstrapNode {
		case constants.Master:
			tmp := group[constants.Master]
			tmp = append(tmp, vertex.Data)
			group[constants.Master] = tmp
		case constants.All:
			tmp := group[constants.All]
			tmp = append(tmp, vertex.Data)
			group[constants.All] = tmp
		case constants.Masters:
			tmp := group[constants.Masters]
			tmp = append(tmp, vertex.Data)
			group[constants.Masters] = tmp
		case constants.Registry:
			tmp := group[constants.Registry]
			tmp = append(tmp, vertex.Data)
			group[constants.Registry] = tmp
		default:
			zlog.Error("the bootstrapNode is invalid")

		}
	}

	return group
}

func (e *Executor) groupAndExecOperate(zeroVertexList []*core.Vertex) {
	componentsMap := e.componentOperateGroup(zeroVertexList)

	for bootstrapNode, components := range componentsMap {
		switch bootstrapNode {
		case constants.All:
			e.allNodesOperateComponent(components)
		case constants.Master:
			e.masterNodeOperateComponent(components)
		case constants.Masters:
			e.mastersNodeOperateComponent(components)
		case constants.Registry:
			e.registryNodeOperateComponent(components)
		default:
			zlog.Error("the bootstrapNode is invalid")
		}
	}
}

// Install 安装组件
func (e *Executor) Install() error {
	fmt.Println("[executor] Start installing components")
	fmt.Println("[executor] This may take a few minutes")
	zlog.Info("Start installing components")
	zeroVertexList := e.Graph.ZeroInDegreeVertexes()

	for {
		if len(zeroVertexList) == 0 {
			break
		}
		e.groupAndExecOperate(zeroVertexList)

		ok, component := isContainsComponent(zeroVertexList, constants.Kubernetes)
		if ok {
			if err := e.initKubernetes(component); err != nil {
				return err
			}
			if err := join.NodeJoinCluster(e.Nodes, component, e.InitCfgFile, constants.InstallCluster); err != nil {
				return err
			}
		}

		ok, _ = isContainsComponent(zeroVertexList, constants.FuyaoSystem)
		if ok && len(e.Nodes.Masters) > constants.MasterCount {
			if err := auth.OauthOperate(e.Nodes.Masters); err != nil {
				return err
			}
		}

		var vertexList []*core.Vertex
		for _, vertex := range zeroVertexList {
			vertexes := e.Graph.NextZeroInDegreeVertexByVertex(vertex)
			vertexList = append(vertexList, vertexes...)
		}
		zeroVertexList = vertexList
	}

	assetAbsPath, err := common.GetAssetAbsPath()
	if err != nil {
		return err
	}
	// e.installComponentManifest
	savePath := filepath.Join(assetAbsPath, constants.InstallComponentManifestFile)
	return common.MarshalYaml[*manifests.InstallComponentManifest](savePath, e.installComponentManifest)
}

// RegistryInstall 安装镜像仓
func (e *Executor) RegistryInstall() {
	fmt.Println("[executor] Start installing components")
	fmt.Println("[executor] This may take a few minutes")
	zlog.Info("Start installing registry")
	e.exec()
}

func (e *Executor) initKubernetes(component *fuyaov1beta1.Component) error {
	master := e.Nodes.Masters[0]
	client := master.Client()

	var cmd string
	if e.InitCfgFile != "" {
		targetPath := filepath.Join(constants.OptPath, constants.OpenFuyao)
		if err := client.CopyFileToRemote(e.InitCfgFile, targetPath); err != nil {
			return fmt.Errorf("failed copy initCfg: %v", err)
		}
		cmd = fmt.Sprintf("cd targetPath && bash %s", filepath.Base(e.InitCfgFile))
	} else {
		packageName, err := common.GetPackageName(master)
		if err != nil {
			return fmt.Errorf("failed to get package name: %v", err)
		}
		PkgSavePath := path.Join(constants.OptPath, constants.OpenFuyao)
		targetDir := filepath.Join(PkgSavePath, packageName, component.Group, component.Name)
		cmd = fmt.Sprintf("cd %s && bash %s", targetDir, constants.InitKubernetesFile)
	}

	_, err := client.SudoExecCmd(cmd)
	if err != nil {
		return fmt.Errorf("failed to init kubernetes: %v", err)
	}
	zlog.Info("Initialized successfully")
	return nil
}

// AddNodes 结点加入集群
func (e *Executor) AddNodes() error {
	fmt.Println("[executor] Start installing components")
	fmt.Println("[executor] This may take a few minutes")
	zlog.Info("Start installing components")
	zeroVertexList := e.Graph.ZeroInDegreeVertexes()

	rawMasters := e.Nodes.Masters
	e.Nodes.Masters = e.Nodes.Masters[1:]

	for {
		if len(zeroVertexList) == 0 {
			break
		}
		e.groupAndExecOperate(zeroVertexList)

		ok, component := isContainsComponent(zeroVertexList, constants.Kubernetes)
		if ok {
			e.Nodes.Masters = rawMasters
			if err := join.NodeJoinCluster(e.Nodes, component, e.InitCfgFile, constants.AddNodes); err != nil {
				return err
			}
			break
		}

		var vertexList []*core.Vertex
		for _, vertex := range zeroVertexList {
			vertexes := e.Graph.NextZeroInDegreeVertexByVertex(vertex)
			vertexList = append(vertexList, vertexes...)
		}
		zeroVertexList = vertexList
	}

	if len(e.Nodes.Masters) > constants.MasterCount {
		ok, err := common.IsDeployComponentByNs(e.Nodes.Masters[0], constants.FuyaoSystem)
		if err != nil {
			return err
		}
		if ok {
			if err := auth.OauthOperate(e.Nodes.Masters); err != nil {
				return err
			}
		}
	}

	return nil
}

// registryNodeOperateComponent 在镜像节点执行操作
func (e *Executor) registryNodeOperateComponent(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, registry := range e.Nodes.Registry {
		e.routineMaximum.Add()
		wg.Add(1)
		go e.execAllNodesOperateComponent(components, registry, &wg)
	}

	wg.Wait()
}

// mastersNodeOperateComponent 在所有的master节点操作组件
func (e *Executor) mastersNodeOperateComponent(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, master := range e.Nodes.Masters {
		e.routineMaximum.Add()
		wg.Add(1)
		go e.execAllNodesOperateComponent(components, master, &wg)
	}

	wg.Wait()
}

// masterNodeOperateComponent 依赖k8s的组件操作，只在一个master节点执行
func (e *Executor) masterNodeOperateComponent(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	for _, component := range components {
		e.execOperate(component, e.Nodes.Masters[0])
	}
}

// execAllNodeOperateComponent 以组件为维度开启多线程
func (e *Executor) execAllNodeOperateComponent(components []fuyaov1beta1.Component) {
	var wg sync.WaitGroup

	for _, component := range components {
		wg.Add(1)
		e.routineMaximum.Add()
		go e.execOperateComponent(component, &wg)
	}

	wg.Wait()
}

func (e *Executor) execOperateComponent(component fuyaov1beta1.Component, wg *sync.WaitGroup) {
	defer wg.Done()
	defer e.routineMaximum.Done()

	e.execOperate(component, e.Nodes.Masters[0])
}

// allNodesOperateComponent 基础组件操作，所有节点均执行
// 以机器为维度开启多线程
func (e *Executor) allNodesOperateComponent(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	// 单台机器安装已组件为维度开启多线程
	if len(e.Nodes.Masters) == 1 && len(e.Nodes.Workers) == 0 && len(e.Nodes.Registry) == 0 {
		e.execAllNodeOperateComponent(components)
		return
	}

	var wg sync.WaitGroup
	for _, worker := range e.Nodes.Workers {
		wg.Add(1)
		e.routineMaximum.Add()
		go e.execAllNodesOperateComponent(components, worker, &wg)
	}

	for _, master := range e.Nodes.Masters {
		wg.Add(1)
		e.routineMaximum.Add()
		go e.execAllNodesOperateComponent(components, master, &wg)
	}

	for _, registry := range e.Nodes.Registry {
		wg.Add(1)
		e.routineMaximum.Add()
		go e.execAllNodesOperateComponent(components, registry, &wg)
	}

	wg.Wait()
}

// execAllNodesOperateComponent 执行基础组件操作
func (e *Executor) execAllNodesOperateComponent(components []fuyaov1beta1.Component, node *fuyaov1beta1.Node,
	wg *sync.WaitGroup) {
	defer wg.Done()
	defer e.routineMaximum.Done()

	for _, component := range components {
		e.execOperate(component, node)
	}
}

// execOperate 执行具体操作
// 执行报错包括 install uninstall preCheck
func (e *Executor) execOperate(component fuyaov1beta1.Component, node *fuyaov1beta1.Node) {
	goon := e.beforeExecOperate(component, node)
	if !goon {
		return
	}

	client := node.Client()

	pkgName, err := common.GetPackageName(node)
	if err != nil {
		zlog.Fatalf(fmt.Sprintf("Failed to obtain package name: %v", err))
	}

	PkgSavePath := path.Join(constants.OptPath, constants.OpenFuyao)
	basePath := filepath.Join(PkgSavePath, pkgName, component.Group, component.Name)
	cmd := fmt.Sprintf("cd %s && bash %s.sh", basePath, e.Operate)
	_, err = client.SudoExecCmd(cmd)
	if err != nil {
		if e.Operate == constants.Uninstall {
			zlog.Errorf("failed to %s %s on %s: %v", e.Operate, component.Name, node.IP, err)
			return
		}
		zlog.Fatalf("failed to %s %s on %s: %v. Please check the status or version of the component", e.Operate,
			component.Name, node.IP, err)
	}
	zlog.Info(fmt.Sprintf("Successfully %s %s on %s", e.Operate, component.Name, node.IP))
	e.afterExecOperate(component, node)
}

// beforeExecOperate 执行对应操作的前置操作
func (e *Executor) beforeExecOperate(component fuyaov1beta1.Component, node *fuyaov1beta1.Node) bool {
	switch e.Operate {
	case constants.Uninstall:
		return e.isUnInstallComponent(component, node)
	default:
		zlog.Info("not extra operate")
	}
	return true
}

// afterExecOperate 执行对应操作的后置操作
func (e *Executor) afterExecOperate(component fuyaov1beta1.Component, node *fuyaov1beta1.Node) {
	switch e.Operate {
	case constants.Install:
		e.recordInstallComponent(component, node)
	default:
		zlog.Info("not extra operate")
	}
}

// isUnInstallComponent 判断是否卸载对应组件
func (e *Executor) isUnInstallComponent(component fuyaov1beta1.Component, node *fuyaov1beta1.Node) bool {
	// 不卸载hub
	if e.SkipDestroyHub {
		for _, registry := range e.Nodes.Registry {
			_, ok := e.HubComponentMap[component.Name]
			if registry.IP == node.IP && ok {
				return false
			}
		}
	}

	// 如果卸载hub，则直接卸载
	if component.Name == constants.Harbor || component.Name == constants.Nerdctl {
		return true
	}

	// 没有清单默认卸载
	installManifests := e.installComponentManifest.Spec.Manifests
	if len(installManifests) == 0 {
		zlog.Info("not install manifests")
		return true
	}

	// 根据清单判断是否卸载
	for _, installManifest := range installManifests {
		if installManifest.Node.Ip != node.IP {
			continue
		}

		for _, uninstallComponent := range installManifest.Components {
			if component.Name == uninstallComponent.Name {
				zlog.Infof("will to uninstall %s", component.Name)
				return true
			}
		}
	}
	return false
}

// recordInstallComponent is record install component info
func (e *Executor) recordInstallComponent(component fuyaov1beta1.Component, node *fuyaov1beta1.Node) {
	// 加锁防止多个线程同时写入信息
	mutex.Lock()
	defer mutex.Unlock()

	isNewNode := true
	installManifests := e.installComponentManifest.Spec.Manifests
	for _, installManifest := range installManifests {
		if installManifest.Node.Ip != node.IP {
			continue
		}
		// add component
		installComponent := manifests.NewInstallComponent(component.Name, component.Version, component.BootstrapNode)
		installManifest.Components = append(installManifest.Components, installComponent)
		isNewNode = false
		break
	}

	// new node
	if isNewNode {
		newNode := manifests.NewNode(node.IP)

		components := make([]*manifests.InstallComponent, 0)
		installComponent := manifests.NewInstallComponent(component.Name, component.Version, component.BootstrapNode)
		components = append(components, installComponent)

		installManifest := manifests.NewInstallManifest(newNode, components)
		e.installComponentManifest.Spec.Manifests = append(e.installComponentManifest.Spec.Manifests, installManifest)
	}
}

func (e *Executor) exec() {
	zeroVertexList := e.Graph.ZeroInDegreeVertexes()

	for {
		if len(zeroVertexList) == 0 {
			break
		}
		e.groupAndExecOperate(zeroVertexList)

		var vertexList []*core.Vertex
		for _, vertex := range zeroVertexList {
			vertexes := e.Graph.NextZeroInDegreeVertexByVertex(vertex)
			vertexList = append(vertexList, vertexes...)
		}
		zeroVertexList = vertexList
	}
}

// Uninstall 卸载组件
func (e *Executor) Uninstall() error {
	fmt.Println("[executor] Start uninstalling components")
	fmt.Println("[executor] This may take a few minutes")
	zlog.Info("Start uninstalling components")

	assetAbsPath, err := common.GetAssetAbsPath()
	if err != nil {
		return err
	}
	installComponentManifestPath := filepath.Join(assetAbsPath, constants.InstallComponentManifestFile)
	err = common.UnmarshalYaml[*manifests.InstallComponentManifest](e.installComponentManifest,
		installComponentManifestPath)
	if err != nil {
		zlog.Error("not found install component manifest")
	}
	e.exec()
	return nil
}

// FlightCheck is used to pre-check components during create cluster or cluster expansion
func (e *Executor) FlightCheck() {
	fmt.Println("[preflight] Checking the installation environment")
	zlog.Info("Start flight check.")
	e.groupAndExecOperate(e.Graph.GetVertex())
}

// isContainsComponent 顶点是否包含组件
func isContainsComponent(vertexList []*core.Vertex, name string) (bool, *fuyaov1beta1.Component) {
	for _, vertex := range vertexList {
		if vertex.Data.Name == name {
			return true, &vertex.Data
		}
	}
	return false, nil
}
