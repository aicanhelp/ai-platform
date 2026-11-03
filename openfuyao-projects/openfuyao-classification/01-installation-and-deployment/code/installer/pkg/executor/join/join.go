/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package join

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/apis/meta/v1"

	"installer/pkg/clients/k8s"
	"installer/pkg/common"
	"installer/pkg/constants"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/values"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// NodeJoinClusterOperate 加入集群需要执行的操作
type NodeJoinClusterOperate struct {
	joinCommand string

	caTmpPath     string
	etcdCaTmpPath string
	certDir       string
	etcdCertDir   string

	kubeConfigPath string

	initCfgFile string

	component *fuyaov1beta1.Component

	k8sClient k8s.Clients
}

// NewNodeJoinClusterOperate is new NodeJoinClusterOperate
func NewNodeJoinClusterOperate(initCfgFile, kubeConfigPath string, component *fuyaov1beta1.Component,
	k8sClient k8s.Clients) *NodeJoinClusterOperate {
	return &NodeJoinClusterOperate{
		initCfgFile:    initCfgFile,
		kubeConfigPath: kubeConfigPath,
		component:      component,
		k8sClient:      k8sClient,
	}
}

// copyCertToRemote 拷贝证书到远端
func (n *NodeJoinClusterOperate) copyCertToRemote(node *fuyaov1beta1.Node) error {
	// 创建证书存放路径
	client := node.Client()
	cmd := fmt.Sprintf("mkdir -p -m %o %s && mkdir -p -m %o %s", constants.RwxRwxRwx, n.certDir,
		constants.RwxRwxRwx, n.etcdCertDir)
	_, err := client.SudoExecCmd(cmd)
	if err != nil {
		return fmt.Errorf(fmt.Sprintf("Failed to create kubernetes cert path: %v", err))
	}

	certPathMap := map[string]string{
		filepath.Join(n.caTmpPath, caCrt):           filepath.Join(n.certDir, caCrt),
		filepath.Join(n.caTmpPath, caKey):           filepath.Join(n.certDir, caKey),
		filepath.Join(n.caTmpPath, saKey):           filepath.Join(n.certDir, saKey),
		filepath.Join(n.caTmpPath, saPub):           filepath.Join(n.certDir, saPub),
		filepath.Join(n.caTmpPath, frontProxyCaCrt): filepath.Join(n.certDir, frontProxyCaCrt),
		filepath.Join(n.caTmpPath, frontProxyCaKey): filepath.Join(n.certDir, frontProxyCaKey),
		filepath.Join(n.etcdCaTmpPath, etcdCaCrt):   filepath.Join(n.etcdCertDir, etcdCaCrt),
		filepath.Join(n.etcdCaTmpPath, etcdCaKey):   filepath.Join(n.etcdCertDir, etcdCaKey),
	}
	for localPath, remotePath := range certPathMap {
		err := client.CopyFileToRemote(localPath, remotePath)
		if err != nil {
			return err
		}
	}

	// 给证书可执行权限
	cmd = fmt.Sprintf("chmod %o %s && chmod %o %s && chmod %o %s && chmod %o %s",
		constants.Rw, filepath.Join(n.certDir, constants.Asterisk),
		constants.Rw, filepath.Join(n.etcdCertDir, constants.Asterisk),
		constants.RwxRxRx, n.certDir,
		constants.RwxRxRx, n.etcdCertDir)
	if _, err = client.SudoExecCmd(cmd); err != nil {
		return fmt.Errorf(fmt.Sprintf("Failed to create kubernetes cert path: %v", err))
	}
	return nil
}

// downloadCertFromRemote 从远端下载证书到本地
func (n *NodeJoinClusterOperate) downloadCertFromRemote(node *fuyaov1beta1.Node) error {
	if err := n.setK8sCertSavePath(node); err != nil {
		return err
	}

	certPath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp, k8sCerts)
	err := utils.MkdirAll(certPath, constants.RwxRxRx)
	if err != nil {
		return fmt.Errorf("failed to create cert save path: %v", err)
	}
	etcdCertPath := filepath.Join(certPath, etcd)
	if err := utils.MkdirAll(etcdCertPath, constants.RwxRxRx); err != nil {
		return fmt.Errorf("failed to create etcd cert save path: %v", err)
	}

	certPathMap := map[string]string{
		filepath.Join(n.certDir, caCrt):           filepath.Join(certPath, caCrt),
		filepath.Join(n.certDir, caKey):           filepath.Join(certPath, caKey),
		filepath.Join(n.certDir, saKey):           filepath.Join(certPath, saKey),
		filepath.Join(n.certDir, saPub):           filepath.Join(certPath, saPub),
		filepath.Join(n.certDir, frontProxyCaCrt): filepath.Join(certPath, frontProxyCaCrt),
		filepath.Join(n.certDir, frontProxyCaKey): filepath.Join(certPath, frontProxyCaKey),
		filepath.Join(n.etcdCertDir, etcdCaCrt):   filepath.Join(etcdCertPath, etcdCaCrt),
		filepath.Join(n.etcdCertDir, etcdCaKey):   filepath.Join(etcdCertPath, etcdCaKey),
	}
	client := node.Client()

	for cert, savePath := range certPathMap {
		if _, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.RwRR, cert)); err != nil {
			return err
		}
		if err := client.DownloadFile(savePath, cert); err != nil {
			return err
		}
		if _, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.Rw, cert)); err != nil {
			return err
		}
	}
	n.caTmpPath = certPath
	n.etcdCaTmpPath = etcdCertPath
	return nil
}

func (n *NodeJoinClusterOperate) removeLocalCert() error {
	return os.RemoveAll(n.caTmpPath)
}

func (n *NodeJoinClusterOperate) copyKubeConfigToWorker(node *fuyaov1beta1.Node) error {
	client := node.Client()

	homeDir, err := client.ExecCmd("mkdir -p $(cd && pwd)/.kube/ && cd && pwd")
	if err != nil {
		return err
	}

	err = client.CopyFileToRemote(n.kubeConfigPath, filepath.Join(homeDir, constants.KubeConfigSavePath))
	if err != nil {
		return err
	}

	return nil
}

// masterJoinCluster master节点加入集群
func (n *NodeJoinClusterOperate) masterJoinCluster(node *fuyaov1beta1.Node) error {
	if n.isNodeInCluster(node.IP) {
		zlog.Infof("%s node has been added to the cluster", node.IP)
		return nil
	}

	client := node.Client()
	command := fmt.Sprintf("%s --control-plane --node-name %s ", n.joinCommand, node.Hostname)
	output, err := client.SudoExecCmd(command)
	zlog.Info(output)
	if err != nil {
		zlog.Errorf("the master node fails to be added to the cluster: %v", err)
		return err
	}
	zlog.Infof("%s node is successfully added to the cluster", node.IP)
	return nil
}

func (n *NodeJoinClusterOperate) isNodeInCluster(ip string) bool {
	nodes, err := n.k8sClient.Kubernetes().CoreV1().Nodes().List(context.TODO(), v1.ListOptions{})
	if err != nil {
		zlog.Fatalf("failed to get nodes info: %v", err)
	}
	for _, node := range nodes.Items {
		for _, address := range node.Status.Addresses {
			if address.Type == constants.InternalIP && address.Address == ip {
				return true
			}
		}
	}

	return false
}

// generateKubeConfig 生成kubeconfig文件
func (n *NodeJoinClusterOperate) generateKubeConfig(node *fuyaov1beta1.Node) error {
	client := node.Client()
	mkdirCmd := "mkdir -p $HOME/.kube"
	copyConfigCmd := "sudo cp -f /etc/kubernetes/admin.conf $HOME/.kube/config"
	chownCmd := "sudo chown $(id -u):$(id -g) $HOME/.kube/config"
	cmd := fmt.Sprintf("%s && %s && %s", mkdirCmd, copyConfigCmd, chownCmd)
	_, err := client.ExecCmd(cmd)
	if err != nil {
		zlog.Errorf("failed generate kubeconfig: %v", err)
	}
	return nil
}

// workerJoinCluster worker加入集群
func (n *NodeJoinClusterOperate) workerJoinCluster(node *fuyaov1beta1.Node) error {
	if n.isNodeInCluster(node.IP) {
		zlog.Infof("%s node has been added to the cluster", node.IP)
		return nil
	}
	client := node.Client()
	command := fmt.Sprintf("%s --node-name %s ", n.joinCommand, node.Hostname)
	_, err := client.SudoExecCmd(command)
	if err != nil {
		zlog.Error("worker join cluster fail:", err)
		return err
	}
	zlog.Info(node.IP, " finish join cluster")
	return nil
}

// generateJoinCommand 生成join命令
func (n *NodeJoinClusterOperate) generateJoinCommand(node *fuyaov1beta1.Node) error {
	client := node.Client()
	joinCommand, err := client.SudoExecCmd("kubeadm token create --print-join-command")
	if err != nil {
		return fmt.Errorf("failed to obtain join command: %v", err)
	}

	// 返回的joinCommand结尾有\r\n字符，需手动去掉，不然影响后续join增加参数
	joinCommand = strings.TrimSuffix(joinCommand, "\n")
	joinCommand = strings.TrimSuffix(joinCommand, "\r")

	n.joinCommand = joinCommand
	return nil
}

// getK8sCertSavePath 获取k8s证书存放路径
func (n *NodeJoinClusterOperate) setK8sCertSavePath(node *fuyaov1beta1.Node) error {
	// 从配置文件获取
	if n.initCfgFile != "" {
		data, err := os.ReadFile(n.initCfgFile)
		if err != nil {
			return err
		}

		initFIleContent := string(data)

		// 使用正则表达式提取certificatesDir后面的字符串
		re := regexp.MustCompile(`certificatesDir:\s+(.+)`)
		match := re.FindStringSubmatch(initFIleContent)
		if len(match) > 1 {
			certificatesDir := strings.TrimSpace(match[1])
			n.certDir = certificatesDir
			n.etcdCertDir = filepath.Join(certificatesDir, etcd)
			return nil
		} else {
			return fmt.Errorf("certificatesDir is an empty character string in initCfgFile")
		}
	}

	// 从组件参数获取
	for _, param := range n.component.Param {
		if param.Name == certDir {
			n.certDir = param.Value
			n.etcdCertDir = filepath.Join(param.Value, etcd)
			return nil
		}
	}

	// 从默认组件参数获取
	pkgPath, err := common.GetOptPkgPathByNode(node)
	if err != nil {
		return err
	}
	defaultValues := filepath.Join(pkgPath, n.component.Group, n.component.Name, constants.DefaultValues)
	valuesData := &values.Values{}
	if err := common.UnmarshalYaml[*values.Values](valuesData, defaultValues); err != nil {
		return err
	}
	for _, param := range valuesData.Spec.Params {
		if param.Name == certDir {
			n.certDir = param.Value
			n.etcdCertDir = filepath.Join(param.Value, etcd)
			return nil
		}
	}

	return fmt.Errorf("can not obtain cert dir")
}

// NodeJoinCluster is node join cluster
func NodeJoinCluster(nodes fuyaov1beta1.Nodes, component *fuyaov1beta1.Component, initCfgFile, execOperate string) error {
	kubeConfigPath, err := k8s.DownLoadKubeConfig(nodes.Masters[0])
	if err != nil {
		return fmt.Errorf("failed download kube config %v", err)
	}

	k8sCfg := k8s.NewKubernetesCfg(kubeConfigPath)
	k8sClient, err := k8s.NewKubernetesClient(k8sCfg)
	if err != nil {
		return fmt.Errorf("failed to get k8s client: %v", err)
	}

	operate := NewNodeJoinClusterOperate(initCfgFile, kubeConfigPath, component, k8sClient)
	exec := NewNodeJoinExecutor(nodes, operate)

	if err := exec.operate.generateJoinCommand(nodes.Masters[0]); err != nil {
		return fmt.Errorf("failed generate join command: %v", err)
	}
	if err := exec.masterJoinCluster(execOperate); err != nil {
		return err
	}

	var wg sync.WaitGroup
	// 内部多线程执行，可认为同时执行join的一系列操作
	exec.workerJoinCluster(&wg)
	wg.Wait()

	if err := exec.operate.removeLocalCert(); err != nil {
		zlog.Errorf("fail to remove k8s cert: %v", err)
	}
	return nil
}
