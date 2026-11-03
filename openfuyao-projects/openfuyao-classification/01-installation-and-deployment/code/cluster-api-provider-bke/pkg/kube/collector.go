/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
)

const (
	bocloudEtcdCertVolumeName1 = "ssl"
	bocloudEtcdCertVolumeName2 = "etcd-certs"
	bocloudPkiCertVolumeName   = "pki"

	bocloudEtcdCertDefaultPath = "/etc/etcd/ssl"
	bocloudPkiCertDefaultPath  = "/etc/kubernetes/pki"

	kubeadmK8sCertsVolumeName  = "k8s-certs"
	kubeadmEtcdCertsVolumeName = "etcd-certs"

	base    = 10
	bitsize = 32
)

// Collect is a func for collecting information from a exit cluster
func (c *Client) Collect() (*CollectResult, []error, []error) {
	// todo maybe need to check the cluster status
	return NewCollector(c).Collect()
}

type Collector struct {
	client  *Client
	keyWord CommandKeyWords

	availableCollectNode *confv1beta1.Node

	result *CollectResult

	errs  []error
	warns []error
}

type CollectResult struct {
	ControlPlaneEndpoint   confv1beta1.APIEndpoint
	Networking             confv1beta1.Networking
	Nodes                  bkenode.Nodes
	KubernetesVersion      string
	EtcdCertificatesDir    string
	K8sCertificatesDir     string
	AvailableCollectedNode *confv1beta1.Node
	ContainerRuntime       confv1beta1.ContainerRuntime
}

func NewCollector(client *Client) *Collector {
	return &Collector{
		client:  client,
		keyWord: NewCertCommandKeyWords(),
		result:  newResult(),
	}
}

func newResult() *CollectResult {
	return &CollectResult{
		Nodes:                bkenode.Nodes{},
		Networking:           confv1beta1.Networking{},
		ControlPlaneEndpoint: confv1beta1.APIEndpoint{},
	}
}

func (c *Collector) Collect() (*CollectResult, []error, []error) {

	c.collectNodeInfo()

	c.collectClusterInfo()

	return c.result, c.warns, c.errs

}

// collectNodeInfo is a func for collecting node information from a exit cluster
// collect node hostname, role, ip, k8s version
func (c *Collector) collectNodeInfo() {
	nodes, err := c.client.ListNodes(nil)
	if err != nil {
		c.errs = append(c.errs, apierrors.NewBadRequest(fmt.Sprintf("failed to list nodes: %v", err)))
		return
	}
	if len(nodes.Items) == 0 {
		c.errs = append(c.errs, errors.New("no nodes found"))
		return
	}

	bkeNodes := make(bkenode.Nodes, len(nodes.Items))
	containerRuntimeMap := initializeContainerRuntimeMap()

	for i, n := range nodes.Items {
		bkeNode := c.processNode(&n)
		bkeNodes[i] = bkeNode

		useRuntime := n.Status.NodeInfo.ContainerRuntimeVersion
		updateContainerRuntimeMap(containerRuntimeMap, useRuntime)

		// 检查etcd角色
		updatedBkeNode := c.checkEtcdRole(bkeNode)
		bkeNodes[i] = updatedBkeNode

		// 选择一个主节点来收集集群信息
		c.selectAvailableCollectNode(&n, updatedBkeNode)

		c.client.BKELog.Info(constant.ClusterManagingReason, "collected node %q information\n ***** role: %v , ip: %q , hostname: %q, runtime: %q", phaseutil.NodeInfo(updatedBkeNode), updatedBkeNode.Role, updatedBkeNode.IP, updatedBkeNode.Hostname, useRuntime)
	}

	containerRuntime := c.determineContainerRuntime(containerRuntimeMap)
	c.result.ContainerRuntime = containerRuntime
	c.result.Nodes = bkeNodes
}

// initializeContainerRuntimeMap 初始化容器运行时计数映射
func initializeContainerRuntimeMap() map[string]int {
	return map[string]int{
		"docker":     0,
		"containerd": 0,
	}
}

// processNode 处理单个节点信息
func (c *Collector) processNode(n *corev1.Node) confv1beta1.Node {
	return confv1beta1.Node(bkenode.Node{
		Hostname: getNodeObjHostname(n),
		Role:     getNodeRole(n),
		IP:       GetNodeIP(n),
	})
}

// updateContainerRuntimeMap 更新容器运行时计数映射
func updateContainerRuntimeMap(containerRuntimeMap map[string]int, useRuntime string) {
	if useRuntime != "" {
		if strings.HasPrefix(useRuntime, "docker") {
			containerRuntimeMap["docker"]++
		}
		if strings.HasPrefix(useRuntime, "containerd") {
			containerRuntimeMap["containerd"]++
		}
	}
}

// checkEtcdRole 检查并更新节点的etcd角色
func (c *Collector) checkEtcdRole(bkeNode confv1beta1.Node) confv1beta1.Node {
	_, err := c.client.GetPod(metav1.NamespaceSystem, StaticPodName(mfutil.Etcd, bkeNode.Hostname))
	if err == nil {
		bkeNode.Role = append(bkeNode.Role, bkenode.EtcdNodeRole)
	}
	return bkeNode
}

// selectAvailableCollectNode 选择可用的收集节点
func (c *Collector) selectAvailableCollectNode(n *corev1.Node, bkeNode confv1beta1.Node) {
	node := bkenode.Node(bkeNode)
	if (node.IsMasterWorker() || node.IsMaster()) && c.availableCollectNode == nil {
		if err := c.client.CheckComponentHealth(n); err == nil {
			c.availableCollectNode = &bkeNode
			c.result.AvailableCollectedNode = &bkeNode
			c.client.BKELog.Info(constant.ClusterManagingReason, "Node %s can be used as basic information collection node", phaseutil.NodeInfo(bkeNode))
		} else {
			c.client.BKELog.Warn(constant.ClusterManagingReason, "Node %s failed health check and cannot be used as a basic information collection node", phaseutil.NodeInfo(bkeNode))
		}
	}
}

// determineContainerRuntime 确定容器运行时配置
func (c *Collector) determineContainerRuntime(containerRuntimeMap map[string]int) confv1beta1.ContainerRuntime {
	containerRuntime := confv1beta1.ContainerRuntime{
		CRI:     bkeinit.CRIContainerd,
		Runtime: bkeinit.DefaultRuntime,
		Param: map[string]string{
			"cgroupDriver": bkeinit.DefaultCgroupDriver,
			"data-root":    bkeinit.DefaultCRIContainerdDataRootDir,
		},
	}
	if containerRuntimeMap["docker"] > containerRuntimeMap["containerd"] {
		containerRuntime.CRI = bkeinit.CRIDocker
		containerRuntime.Param["data-root"] = bkeinit.DefaultCRIDockerDataRootDir
	}
	return containerRuntime
}

func (c *Collector) collectClusterInfo() {
	c.collectKubernetesVersion()

	if c.availableCollectNode == nil {
		c.errs = append(c.errs, errors.New("no basic information collection node, could not collect cluster base info"))
		return
	}

	hostName := c.availableCollectNode.Hostname
	c.collectControllerManagerInfo(hostName)
	c.collectAPIServerInfo(hostName)
	c.collectEtcdInfo(hostName)
	c.collectControlPlaneEndpoint()
	c.setDefaults()
}

func (c *Collector) collectKubernetesVersion() {
	version, err := c.client.ClientSet.Discovery().ServerVersion()
	if err != nil {
		c.errs = append(c.errs, apierrors.NewBadRequest(fmt.Sprintf("failed to get kubernetes cluster version err: %v", err)))
	} else {
		c.result.KubernetesVersion = version.String()
	}
}

func (c *Collector) collectControllerManagerInfo(hostName string) {
	controllerManagerPodName := StaticPodName(mfutil.KubeControllerManager, hostName)
	controllerManagerPod, err := c.client.GetPod(metav1.NamespaceSystem, controllerManagerPodName)
	if err != nil || controllerManagerPod == nil {
		c.errs = append(c.errs, errors.Errorf("failed to get pod %s, "+
			"so that can't get cluster pod CIDR and service CIDR, err: %v", controllerManagerPodName, err))
		return
	}

	// pod cidr
	if v, ok := commandExit("--cluster-cidr", controllerManagerPod.Spec.Containers[0].Command); ok {
		c.result.Networking.PodSubnet = v
	} else {
		c.result.Networking.PodSubnet = bkeinit.DefaultPodSubnet
		c.warns = append(c.warns, errors.Errorf("can't get cluster pod CIDR form pod %s, "+
			"start command '--cluster-cidr' not found", controllerManagerPodName))
	}
}

func (c *Collector) collectAPIServerInfo(hostName string) {
	apiServerPodName := StaticPodName(mfutil.KubeAPIServer, hostName)
	apiServerPod, err := c.client.GetPod(metav1.NamespaceSystem, apiServerPodName)
	if err != nil || apiServerPod == nil {
		c.errs = append(c.errs, errors.Errorf("failed to get pod %s, "+
			"so that can't get cluster dns domain and service CIDR, err: %v", apiServerPodName, err))
		return
	}

	// dns domain
	if v, ok := commandExit("--service-account-issuer", apiServerPod.Spec.Containers[0].Command); ok {
		v = strings.TrimPrefix(v, "https://kubernetes.default.svc.")
		v = strings.TrimPrefix(v, "http://kubernetes.default.svc")
		c.result.Networking.DNSDomain = v
	} else {
		c.result.Networking.DNSDomain = bkeinit.DefaultServiceDNSDomain
		c.warns = append(c.warns, errors.Errorf("can't get cluster dns domain form pod %s, "+
			"start command '--service-account-issuer' not found, use default dns domain %s",
			apiServerPodName, bkeinit.DefaultServiceDNSDomain))
	}

	// service cidr
	if v, ok := commandExit("--service-cluster-ip-range", apiServerPod.Spec.Containers[0].Command); ok {
		c.result.Networking.ServiceSubnet = v
	} else {
		c.result.Networking.ServiceSubnet = bkeinit.DefaultServicesSubnet
		c.warns = append(c.warns, errors.Errorf("can't get cluster service CIDR form pod %s, "+
			"start command '--service-cluster-ip-range' not found, use default service CIDR %s",
			apiServerPodName, bkeinit.DefaultServicesSubnet))
	}

	// k8s pki store dir
	for _, v := range apiServerPod.Spec.Volumes {
		if v.Name == "pki" || v.Name == "k8s-certs" {
			pkiPath := v.HostPath.Path
			if !strings.HasSuffix(pkiPath, "pki") {
				pkiPath = filepath.Join(pkiPath, "pki")
			}
			c.result.K8sCertificatesDir = pkiPath
			break
		}
	}
}

func (c *Collector) collectEtcdInfo(hostName string) {
	etcdPodName := StaticPodName(mfutil.Etcd, hostName)
	etcdPod, err := c.client.GetPod(metav1.NamespaceSystem, etcdPodName)
	if err != nil || etcdPod == nil {
		c.errs = append(c.errs, errors.Errorf("failed to get pod %s, "+
			"so that can't get etcd pki store dir, err: %v", etcdPodName, err))
		return
	}

	for _, v := range etcdPod.Spec.Volumes {
		if v.Name == "etcd-certs" {
			c.result.EtcdCertificatesDir = strings.TrimSuffix(v.HostPath.Path, "etcd")
			break
		}
	}
}

func (c *Collector) collectControlPlaneEndpoint() {
	// control plane endpoint
	endpoint, err := url.Parse(c.client.RestConfig.Host)
	if err != nil {
		c.errs = append(c.errs, fmt.Errorf("failed to parse control plane endpoint form kubeconfig: %w", err))
		return
	}
	// 使用十进制解析端口号
	portInt, err := strconv.ParseInt(endpoint.Port(), base, bitsize)
	if err != nil {
		c.errs = append(c.errs, fmt.Errorf("failed to parse control plane endpoint port form kubeconfig: %w", err))
		return
	}
	c.result.ControlPlaneEndpoint = confv1beta1.APIEndpoint{
		Host: endpoint.Hostname(),
		Port: int32(portInt),
	}
}

func (c *Collector) setDefaults() {
	if c.result.EtcdCertificatesDir == "" {
		c.result.EtcdCertificatesDir = bocloudEtcdCertDefaultPath
		c.warns = append(c.warns, errors.Errorf("etcd pki store dir not found, "+
			"use default value: %s", bocloudEtcdCertDefaultPath))
	}
	if c.result.K8sCertificatesDir == "" {
		c.result.K8sCertificatesDir = bocloudPkiCertDefaultPath
		c.warns = append(c.warns, errors.Errorf("k8s pki store dir not found, "+
			"use default value: %s", bocloudPkiCertDefaultPath))
	}
}

func getNodeObjHostname(node *corev1.Node) string {
	labels := node.GetLabels()
	if labels == nil {
		return node.Name
	}
	hostname, ok := labels[corev1.LabelHostname]
	if !ok {
		return node.Name
	}
	if hostname == "" {
		hostname = node.Name
	}
	return hostname
}

func getNodeRole(node *corev1.Node) []string {
	if labelhelper.IsMasterNode(node) && labelhelper.IsWorkerNode(node) {
		return []string{bkenode.MasterWorkerNodeRole}
	}
	if labelhelper.IsMasterNode(node) && !labelhelper.IsWorkerNode(node) {
		return []string{bkenode.MasterNodeRole}
	}
	return []string{bkenode.WorkerNodeRole}
}

func GetNodeIP(node *corev1.Node) string {
	for _, address := range node.Status.Addresses {
		if address.Type == corev1.NodeInternalIP {
			return address.Address
		}
	}
	return ""
}

func getNodeK8sVersion(node *corev1.Node) string {
	return node.Status.NodeInfo.KubeletVersion
}

func getCertificatesDirFromCertPaths(certPaths []string) string {
	pathsDirs := make([]string, len(certPaths))
	for i, path := range certPaths {
		pathsDirs[i] = filepath.Dir(path)
	}
	return utils.CommonPrefix(pathsDirs)
}

type CommandKeyWords map[string][]string

func NewCertCommandKeyWords() CommandKeyWords {
	return map[string][]string{
		mfutil.KubeAPIServer: {
			"--kubelet-client-certificate",
			"--kubelet-client-key",
			"--service-account-key-file",
			"--proxy-client-cert-file",
			"--proxy-client-key-file",
			"--requestheader-client-ca-file",
			"--etcd-cafile",
			"--etcd-certfile",
			"--etcd-keyfile",
			"--client-ca-file",
			"--tls-cert-file",
			"--tls-private-key-file",
		},
		mfutil.KubeControllerManager: {
			"--cluster-signing-cert-file",
			"--cluster-signing-key-file",
			"--root-ca-file",
			"--service-account-private-key-file",
			"--tls-cert-file",
			"--tls-private-key-file",
		},
		mfutil.KubeScheduler: {},
		mfutil.Etcd: {
			"--cert-file",
			"--key-file",
			"--trusted-ca-file",
			"--peer-cert-file",
			"--peer-key-file",
			"--peer-trusted-ca-file",
		},
		"etcdComponentVolumeCertName": {
			kubeadmEtcdCertsVolumeName,
			"ssl",
		},
		"k8sComponentVolumeCertName": {
			"pki",
			kubeadmK8sCertsVolumeName,
		},
		"network": {
			"--cluster-cidr",
			"--service-cluster-ip-range",
		},
	}
}

func (c CommandKeyWords) certCommandExit(component string, command string) (string, bool) {
	prefixs := c[component]
	for _, prefix := range prefixs {
		if strings.HasPrefix(command, prefix) {
			str := strings.TrimPrefix(command, prefix)
			str = strings.TrimPrefix(str, "=")
			return str, true
		}
	}
	return "", false
}

func commandExit(except string, commands []string) (string, bool) {
	for _, command := range commands {
		if strings.HasPrefix(command, except) {
			str := strings.TrimPrefix(command, except)
			str = strings.TrimPrefix(str, "=")
			return str, true
		}
	}
	return "", false
}

func (c CommandKeyWords) certVolumeExit(volumeName string, componentType string) bool {
	switch componentType {
	case mfutil.Etcd:
		return utils.ContainsString(c["etcdComponentVolumeCertName"], volumeName)
	case "":
		return utils.ContainsString(c["etcdComponentVolumeCertName"], volumeName) ||
			utils.ContainsString(c["k8sComponentVolumeCertName"], volumeName)
	default:
		return utils.ContainsString(c["k8sComponentVolumeCertName"], volumeName)
	}

}
