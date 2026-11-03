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

package binary

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sync"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor/core"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/prepare/asset"
	"installer/pkg/utils"
	"installer/pkg/utils/pathutil"
	"installer/pkg/zlog"
)

// PrepareArtifact 准备制品
type PrepareArtifact struct {
	ArtifactSavePath       string
	UnCompressArtifactPath string
	EnableCopyImage        bool
	EnableCopyChart        bool
	EnableCopyBinary       bool

	Graph              *core.DirectedGraph
	Nodes              fuyaov1beta1.Nodes
	AdvancedParams     []base.Param
	componentManifests *manifests.ComponentManifests

	routineMaximum *utils.GoRoutineLimit
}

// NewPrepareArtifact new prepare artifact struct
func NewPrepareArtifact(graph *core.DirectedGraph, nodes fuyaov1beta1.Nodes, artifactSavePath string) *PrepareArtifact {
	unCompressArtifactPath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp,
		constants.OfflineArtifactPath)
	return &PrepareArtifact{
		UnCompressArtifactPath: unCompressArtifactPath,
		Graph:                  graph,
		Nodes:                  nodes,
		componentManifests:     &manifests.ComponentManifests{},
		routineMaximum:         utils.NewGoRoutineLimits(),
		AdvancedParams:         []base.Param{},
		ArtifactSavePath:       artifactSavePath,
	}
}

func (c *PrepareArtifact) bootstrapGroup(vertexList []*core.Vertex) map[string][]fuyaov1beta1.Component {
	group := make(map[string][]fuyaov1beta1.Component)

	for _, vertex := range vertexList {
		zlog.Debugf("The grouping component is %s", vertex.Data.Name)
		if vertex.Data.BootstrapNode == constants.Master {
			tmp := group[constants.Master]
			tmp = append(tmp, vertex.Data)
			group[constants.Master] = tmp
		} else if vertex.Data.BootstrapNode == constants.All {
			tmp := group[constants.All]
			tmp = append(tmp, vertex.Data)
			group[constants.All] = tmp
		} else if vertex.Data.BootstrapNode == constants.Masters {
			tmp := group[constants.Masters]
			tmp = append(tmp, vertex.Data)
			group[constants.Masters] = tmp
		} else if vertex.Data.BootstrapNode == constants.Registry {
			tmp := group[constants.Registry]
			tmp = append(tmp, vertex.Data)
			group[constants.Registry] = tmp
		} else {
			zlog.Error("not exec node")
		}
	}

	return group
}

// execCopyToAllNodes 一个线程内遍历组件拷贝
func (c *PrepareArtifact) execCopyToAllNodes(components []fuyaov1beta1.Component, node *fuyaov1beta1.Node,
	wg *sync.WaitGroup) {
	defer wg.Done()
	defer c.routineMaximum.Done()

	for _, component := range components {
		c.execCopy(component, node)
	}
}

// copyToAllNodes 拷贝二进制文件到到所有节点
func (c *PrepareArtifact) copyToAllNodes(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, worker := range c.Nodes.Workers {
		wg.Add(1)
		c.routineMaximum.Add()
		go c.execCopyToAllNodes(components, worker, &wg)
	}

	for _, master := range c.Nodes.Masters {
		wg.Add(1)
		c.routineMaximum.Add()
		go c.execCopyToAllNodes(components, master, &wg)
	}

	for _, registry := range c.Nodes.Registry {
		wg.Add(1)
		c.routineMaximum.Add()
		go c.execCopyToAllNodes(components, registry, &wg)
	}

	wg.Wait()
}

// copyToMasterNode 拷贝组件二进制文件到所有master节点
func (c *PrepareArtifact) copyToMastersNode(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, master := range c.Nodes.Masters {
		wg.Add(1)
		c.routineMaximum.Add()
		go c.execCopyToAllNodes(components, master, &wg)
	}

	wg.Wait()
}

// copyToOneMasterNode 拷贝二进制文件到一个master节点
func (c *PrepareArtifact) copyToOneMasterNode(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, component := range components {
		wg.Add(1)
		c.routineMaximum.Add()
		go c.execCopyToOneMasterNode(component, c.Nodes.Masters[0], &wg)
	}
	wg.Wait()
}

func (c *PrepareArtifact) execCopyToOneMasterNode(component fuyaov1beta1.Component, node *fuyaov1beta1.Node,
	wg *sync.WaitGroup) {
	defer wg.Done()
	defer c.routineMaximum.Done()
	c.execCopy(component, node)
}

func (c *PrepareArtifact) copyToRegistryNode(components []fuyaov1beta1.Component) {
	if len(components) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, registry := range c.Nodes.Registry {
		wg.Add(1)
		c.routineMaximum.Add()
		go c.execCopyToAllNodes(components, registry, &wg)
	}
	wg.Wait()
}

// execCopy 执行具体拷贝
func (c *PrepareArtifact) execCopy(component fuyaov1beta1.Component, node *fuyaov1beta1.Node) {
	pkgName, err := common.GetPackageName(node)
	if err != nil {
		zlog.Fatalf("failed to obtain package name: %v", err)
	}

	dstBasePath := filepath.Join(constants.OptPath, constants.OpenFuyao, pkgName, component.Group, component.Name)
	// 二进制文件存储路径示例：fuyao-offline-artifact/fuyao-openEuler-22.03/cri/containerd/arm64-bin
	artifactFileName := fmt.Sprintf("%s-%s-%s", constants.Fuyao, node.OsName, node.OsVersion)
	srcBasePath := filepath.Join(c.UnCompressArtifactPath, artifactFileName, component.Group, component.Name)

	pathMap := make(map[string]string)
	if c.EnableCopyImage {
		pathMap[filepath.Join(srcBasePath, constants.Image)] = filepath.Join(dstBasePath, constants.Image)
	}
	if c.EnableCopyChart {
		pathMap[filepath.Join(srcBasePath, constants.Chart)] = filepath.Join(dstBasePath, constants.Chart)
	}
	if c.EnableCopyBinary {
		pathMap[filepath.Join(srcBasePath, fmt.Sprintf("%s-bin", node.CpuArch))] = filepath.Join(dstBasePath,
			fmt.Sprintf("%s-bin", node.CpuArch))
	}
	// 安装卸载时均需要拷贝yaml到目标节点
	pathMap[filepath.Join(srcBasePath, constants.Yaml)] = filepath.Join(dstBasePath, constants.Yaml)

	for srcPath, dstPath := range pathMap {
		c.copy(srcPath, dstPath, component.Name, node)
	}
}

func (c *PrepareArtifact) copy(srcPath, dstPath, componentName string, node *fuyaov1beta1.Node) {
	client := node.Client()

	if _, err := os.Stat(srcPath); os.IsNotExist(err) {
		zlog.Infof("%s not have %s path", componentName, srcPath)
		return
	}
	if err := client.CopyFilesToRemote(srcPath, dstPath); err != nil {
		zlog.Fatal(fmt.Sprintf("failed copy %s from %s to remote node %s : %v", componentName, srcPath, node.IP,
			err))
	}
}

func (c *PrepareArtifact) copyBinary() {
	execGroupMap := c.bootstrapGroup(c.Graph.GetVertex())
	for key, components := range execGroupMap {
		if key == constants.All {
			c.copyToAllNodes(components)
		} else if key == constants.Master {
			c.copyToOneMasterNode(components)
		} else if key == constants.Masters {
			c.copyToMastersNode(components)
		} else if key == constants.Registry {
			c.copyToRegistryNode(components)
		} else {
			zlog.Errorf("invalid execution node group: %s", key)
		}
	}
}

// OfflineCopyBinary is offline install copy binary
func (c *PrepareArtifact) OfflineCopyBinary() error {
	fmt.Println("[prepare] Copy the binary file to the node")
	if err := c.unCompressArtifact(); err != nil {
		return err
	}
	c.EnableCopyChart = true

	// 执行拷贝
	c.copyBinary()

	err := os.RemoveAll(c.UnCompressArtifactPath)
	if err != nil {
		zlog.Errorf("failed to delete tmp dir: %s", c.UnCompressArtifactPath)
	}
	return nil
}

// OnlineCopyBinary is online install copy binary
func (c *PrepareArtifact) OnlineCopyBinary() error {
	// 直接制定制解压路径为制品存放路径
	c.UnCompressArtifactPath = c.ArtifactSavePath

	// 执行拷贝
	c.copyBinary()

	if err := os.RemoveAll(c.UnCompressArtifactPath); err != nil {
		zlog.Errorf("failed to delete tmp dir: %v", err)
	}
	return nil
}

func (c *PrepareArtifact) unCompressArtifact() error {
	tmpPath := filepath.Join(pathutil.InstallExecPath(), constants.Tmp)
	if err := utils.MkdirAll(tmpPath, constants.RwxRxRx); err != nil {
		return err
	}
	if err := common.UnCompressTarGZ(c.ArtifactSavePath, tmpPath); err != nil {
		return err
	}
	return nil
}

// CreateRegistryCopyBinary is create registry copy binary
func (c *PrepareArtifact) CreateRegistryCopyBinary() error {
	fmt.Println("[prepare] Copy the binary file to the node")
	if err := c.unCompressArtifact(); err != nil {
		return err
	}
	// 执行拷贝
	c.copyBinary()

	return nil
}

// UploadImages 上传镜像到镜像仓
func (c *PrepareArtifact) UploadImages(domain string) error {
	componentManifests, err := common.GetComponentManifests()
	if err != nil {
		return err
	}

	node := &fuyaov1beta1.Node{}
	if len(c.Nodes.Registry) > 0 {
		node = c.Nodes.Registry[0]
	} else if len(c.Nodes.Masters) > 0 {
		node = c.Nodes.Masters[0]
	} else {
		return fmt.Errorf("there are no valid nodes")
	}
	artifactBasePath := path.Join(c.UnCompressArtifactPath,
		fmt.Sprintf("%s-%s-%s", constants.Fuyao, node.OsName, node.OsVersion))
	pkgPath, err := common.GetOptPkgPathByNode(node)
	if err != nil {
		return err
	}
	for group, components := range componentManifests.Spec.Components {
		for _, component := range components {
			srcShellPath := filepath.Join(pkgPath, group, component.Name)
			imagesPath := filepath.Join(artifactBasePath, group, component.Name, constants.Image)
			registry := domain + constants.Slash + constants.DefaultProject
			command := fmt.Sprintf("cd %s && bash %s -r %s -p %s", srcShellPath, constants.UploadFile, registry,
				imagesPath)
			_, err = utils.ExecLocalCmd(command)
			if err != nil {
				return err
			}
		}
	}
	if err := os.RemoveAll(c.UnCompressArtifactPath); err != nil {
		zlog.Errorf("failed to delete tmp dir: %v", err)
	}
	return nil
}

func (c *PrepareArtifact) getPkgNameMap() (map[string]string, error) {
	pkgNameMap := make(map[string]string)
	nodes := append(c.Nodes.Masters, c.Nodes.Workers...)

	for _, master := range nodes {
		osName, osVersion, _ := master.OsInfo()
		key := fmt.Sprintf("%s-%s-%s", constants.Fuyao, osName, osVersion)
		packageName, err := common.GetPackageName(master)
		if err != nil {
			return nil, err
		}
		pkgNameMap[key] = packageName
	}

	return pkgNameMap, nil
}

// execDownload 执行下载
func (c *PrepareArtifact) execDownload(pkgPath string, component fuyaov1beta1.Component, saveFolder string, wg *sync.WaitGroup) {
	defer wg.Done()
	defer c.routineMaximum.Done()
	zlog.Debug("start download ", component.Name)

	// 渲染组件参数
	basePath := filepath.Join(pkgPath, component.Group, component.Name)
	filePath := filepath.Join(basePath, constants.DownloadFile)
	var param = make(map[string]interface{})
	for _, p := range c.AdvancedParams {
		param[p.Name] = p.Value
	}
	if err := asset.RenderComponentParam(filePath, param); err != nil {
		zlog.Fatalf("Failed to render params: %v", err)
	}

	// 组件目录，该目录下放二进制和镜像
	// fuyao-offline-artifact/fuyao-openEuler-22.03/cri/containerd
	savePath := filepath.Join(saveFolder, component.Group, component.Name)
	if err := utils.MkdirAll(savePath, constants.RwxRxRx); err != nil {
		zlog.Fatalf("failed to create dir: %v", err)
	}

	// 执行下载
	cmd := fmt.Sprintf("cd %s && bash %s -p %s -y true", basePath, constants.DownloadFile, savePath)
	if c.EnableCopyBinary {
		cmd = fmt.Sprintf("cd %s && bash %s -p %s -y true -b true", basePath, constants.DownloadFile, savePath)
	}
	_, err := utils.ExecLocalCmd(cmd)
	if err != nil {
		zlog.Fatalf("the component %s failed to exec %s: %v", component.Name, constants.DownloadFile, err)
	}

	zlog.Debug("success download ", component.Name)
}

func (c *PrepareArtifact) downLoadArtifact() error {
	// 精准获取代码包名
	pkgNameMap, err := c.getPkgNameMap()
	if err != nil {
		return err
	}

	// 获取文件路径
	absPath, err := common.GetAssetAbsPathReal()
	if err != nil {
		return err
	}
	tmpPath := filepath.Join(absPath, constants.Tmp)
	if err := utils.Mkdir(tmpPath, constants.Rwx); err != nil {
		return err
	}

	var wg sync.WaitGroup
	for artifactPkgName, installPkgName := range pkgNameMap {
		if err := common.UnCompressTarGZ(filepath.Join(absPath, installPkgName+constants.TarGz), tmpPath); err != nil {
			return err
		}

		// 创建制品文件夹
		artifactPath := filepath.Join(c.ArtifactSavePath, artifactPkgName)
		if err := utils.MkdirAll(artifactPath, constants.Rwx); err != nil {
			return err
		}

		for _, vertex := range c.Graph.GetVertex() {
			wg.Add(1)
			c.routineMaximum.Add()
			go c.execDownload(filepath.Join(tmpPath, installPkgName), vertex.Data, artifactPath, &wg)
		}
	}
	wg.Wait()

	if err := os.RemoveAll(tmpPath); err != nil {
		zlog.Error("failed to remove ", tmpPath)
	}
	return nil
}

// CreateClusterPrepareArtifact 安装集群准备制品
func (c *PrepareArtifact) CreateClusterPrepareArtifact(offline bool) error {
	fmt.Println("[prepare] Start preparing the artifact")
	if offline {
		return c.OfflineCopyBinary()
	}

	if c.ArtifactSavePath == "" {
		c.ArtifactSavePath = filepath.Join(pathutil.InstallExecPath(), constants.OfflineArtifactPath)
	} else {
		return c.OfflineCopyBinary()
	}

	fmt.Println("[prepare] This may take a few minutes because need to download the artifact")
	if err := c.downLoadArtifact(); err != nil {
		return err
	}
	return c.OnlineCopyBinary()
}
