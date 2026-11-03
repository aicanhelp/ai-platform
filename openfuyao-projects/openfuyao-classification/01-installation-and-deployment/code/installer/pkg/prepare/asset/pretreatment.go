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

package asset

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"installer/pkg/common"
	"installer/pkg/constants"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// Pretreatment 安装包预处理结构体
type Pretreatment struct {
	Cluster *fuyaov1beta1.Cluster
	// 安装包原始存放路径
	PkgRawSavePath string
	// 预处理后的安装包存储路径
	PkgSavePath string

	routineMaximum *utils.GoRoutineLimit
}

func NewPretreatment(cluster *fuyaov1beta1.Cluster) *Pretreatment {
	pkgSavePath := filepath.Join(constants.OptPath, constants.OpenFuyao)
	return &Pretreatment{
		Cluster:        cluster,
		PkgSavePath:    pkgSavePath,
		routineMaximum: utils.NewGoRoutineLimits(),
	}
}

// PretreatmentPackage 多线程预处理代码包
func (p *Pretreatment) PretreatmentPackage() error {
	fmt.Println("[prepare] Preprocessing shell script")
	assetPath, err := common.GetAssetAbsPathReal()
	if err != nil {
		return err
	}
	p.PkgRawSavePath = assetPath

	var wg sync.WaitGroup
	// 开启多线程的数量限制

	for _, master := range p.Cluster.Spec.Nodes.Masters {
		wg.Add(1)
		p.routineMaximum.Add()
		go p.ExecPretreatment(master, constants.Master, &wg)
	}

	for _, worker := range p.Cluster.Spec.Nodes.Workers {
		wg.Add(1)
		p.routineMaximum.Add()
		go p.ExecPretreatment(worker, constants.Worker, &wg)
	}

	registryNode := getRegistryNodeNotInCluster(p.Cluster.Spec.Nodes)
	for _, registry := range registryNode {
		wg.Add(1)
		p.routineMaximum.Add()
		go p.ExecPretreatment(registry, constants.Registry, &wg)
	}

	wg.Wait()
	zlog.Info("Complete installation package preprocessing!")

	return nil
}

func (p *Pretreatment) renderFile(pkgPath string, component fuyaov1beta1.Component, filenames []string,
	node *fuyaov1beta1.Node) {
	data := ParamsMap(component, pkgPath, p.Cluster.Spec.AdvancedParams)

	switch component.Name {
	case KubeVip:
		data = AddKubeVipExtraParam(data, node.IP, p.Cluster.Spec.Nodes)
	case constants.AscendDeployer:
		data = handleAscendDeployerParams(data, p.Cluster.Spec.Nodes)
	case Kubernetes:
		filePath := filepath.Join(pkgPath, component.Group, component.Name, constants.InitKubernetesFile)
		data = AddK8sInitExtraParam(data, node, p.Cluster)
		RenderComponentParams(component, filePath, data)
	default:
		zlog.Info("no additional processing required")
	}

	for _, filename := range filenames {
		filePath := filepath.Join(pkgPath, component.Group, component.Name, filename)
		RenderComponentParams(component, filePath, data)
	}
}

// ExecPretreatment 执行预处理
func (p *Pretreatment) ExecPretreatment(node *fuyaov1beta1.Node, nodeRole string, wg *sync.WaitGroup) {
	defer wg.Done()
	defer p.routineMaximum.Done()

	// 创建目录
	fileDir := node.IP + constants.Minus + nodeRole
	targetDir := filepath.Join(p.PkgRawSavePath, fileDir)
	// 删除原目录，防止出现解压偶发异常
	if err := os.RemoveAll(targetDir); err != nil {
		zlog.Fatalf("failed to remove %s on %s", targetDir, fileDir)
	}
	if err := utils.MkdirAll(targetDir, constants.RwxRxRx); err != nil {
		zlog.Fatalf(fmt.Sprintf("Failed to create target folder: %v", err))
	}

	// 解压代码
	pkgName, err := common.GetPackageName(node)
	if err != nil {
		zlog.Fatalf("error：", err)
	}

	err = common.UnCompressTarGZ(filepath.Join(p.PkgRawSavePath, pkgName+constants.TarGz), targetDir)
	if err != nil {
		zlog.Fatalf(fmt.Sprintf("Failed to decompress the installation package: %v . Package name is %s", err, pkgName))
	}

	// 渲染 and 参数处理
	for _, component := range p.Cluster.Spec.Components {
		zlog.Info("Start rendering ", component.Name, " file")
		pkgPath := filepath.Join(targetDir, pkgName)
		filenames := []string{constants.InstallFile, constants.UnInstallFile, constants.PreInstallFile}
		p.renderFile(pkgPath, component, filenames, node)
		zlog.Info("Successfully rendered ", component.Name, " file")
	}

	// 压缩文件
	err = common.CompressTarGZ(filepath.Join(targetDir, pkgName), filepath.Join(targetDir, pkgName+constants.TarGz))
	if err != nil {
		zlog.Fatalf(fmt.Sprintf("Compression failed: %v", err))
	}

	// 拷贝文件
	client := node.Client()
	_, err = client.SudoExecCmd(fmt.Sprintf("mkdir -p -m %o %s && chmod %o %s", constants.RwxRwxRwx,
		p.PkgSavePath, constants.RwxRwxRwx, p.PkgSavePath))
	if err != nil {
		zlog.Fatalf("failed to mkdir on %s: %v", node.IP, err)
	}

	zipPkgName := fmt.Sprintf("%s.tar.gz", pkgName)
	err = client.CopyFileToRemote(filepath.Join(targetDir, zipPkgName), filepath.Join(p.PkgSavePath, zipPkgName))
	if err != nil {
		zlog.Fatalf("failed copy file to %s: %v", node.IP, err)
	}

	// 解压
	_, err = client.SudoExecCmd(fmt.Sprintf("cd %s && tar -zxvf %s", p.PkgSavePath, zipPkgName))
	if err != nil {
		zlog.Fatalf("failed to unzip file one %s: %v", node.IP, err)
	}
	zlog.Info("unzip package on ", node.IP)

	// 删除目录
	if os.RemoveAll(targetDir) != nil {
		zlog.Error(fmt.Sprintf("remove file path fail %v", err))
	}
}

// AddKubeVipExtraParam 增加kube-vip的额外参数
func AddKubeVipExtraParam(data map[string]interface{}, ip string, nodes fuyaov1beta1.Nodes) map[string]interface{} {
	if data == nil {
		data = make(map[string]interface{})
	}
	data["bgp_router_id"] = ip

	defaultString := ":65000::false,"
	bgpPeers := ""
	for _, node := range nodes.Masters {
		bgpPeers += node.IP + defaultString
	}
	bgpPeers = strings.TrimRight(bgpPeers, ",")
	data["bgppeers"] = bgpPeers

	return data
}

// AddK8sInitExtraParam 增加k8s初始化的额外参数
func AddK8sInitExtraParam(data map[string]interface{}, node *fuyaov1beta1.Node,
	cluster *fuyaov1beta1.Cluster) map[string]interface{} {
	if data == nil {
		data = make(map[string]interface{})
	}

	for _, component := range cluster.Spec.Components {
		if component.Name == Kubernetes {
			data["kubernetes_version"] = component.Version
		}
		continue
	}
	data["host"] = node.IP
	data["hostname"] = node.Hostname

	isHa := "false"
	if len(cluster.Spec.Nodes.Masters) > 1 {
		isHa = "true"
		for _, component := range cluster.Spec.Components {
			if component.Name == KubeVip {
				data = addHaParam(data, component)
				break
			}
		}
	}
	data["is_ha"] = isHa

	endpoint, ok := data["controlPlaneEndpoint"]
	if ok && endpoint == "" {
		data["controlPlaneEndpoint"] = node.IP
	}

	return data
}

func addHaParam(data map[string]interface{}, component fuyaov1beta1.Component) map[string]interface{} {
	if data == nil {
		data = make(map[string]interface{})
	}
	for _, param := range component.Param {
		if param.Name == VIP {
			data[VIP] = param.Value
			break
		}
	}
	return data
}

func getRegistryNodeNotInCluster(nodes fuyaov1beta1.Nodes) []*fuyaov1beta1.Node {
	clusterNodes := nodes.Masters
	clusterNodes = append(clusterNodes, nodes.Workers...)

	registryNode := make([]*fuyaov1beta1.Node, 0)

	for _, node := range nodes.Registry {
		isIn := false

		for _, clusterNode := range clusterNodes {
			if node.IP == clusterNode.IP {
				isIn = true
				break
			}
		}

		if isIn {
			continue
		}

		registryNode = append(registryNode, node)
	}
	return registryNode
}

// getNodeUserPasswd 获取节点的用户名和密码
func getNodeUserPasswd(nodes []*fuyaov1beta1.Node) string {
	nodeInfoList := make([]string, 0)
	for _, node := range nodes {
		info := fmt.Sprintf("%s:%s", node.User, string(node.Password))
		nodeInfoList = append(nodeInfoList, info)
	}
	return strings.Join(nodeInfoList, ",")
}

// handleAscendDeployerParams 处理ascend deployer的参数
func handleAscendDeployerParams(data map[string]interface{}, nodes fuyaov1beta1.Nodes) map[string]interface{} {
	if data == nil {
		data = make(map[string]interface{})
	}

	data["masterInfo"] = getNodeUserPasswd(nodes.Masters)
	data["workerInfo"] = getNodeUserPasswd(nodes.Workers)
	return data
}
