/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package node

import (
	"fmt"
	"net"
	"reflect"
	"strings"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
)

const (
	WorkerNodeRole       = "node"
	MasterNodeRole       = "master"
	EtcdNodeRole         = "etcd"
	MasterWorkerNodeRole = MasterNodeRole + "/" + WorkerNodeRole

	// DefaultNodeSSHPort is the default SSH port for nodes
	DefaultNodeSSHPort = "22"
	// DefaultNodeUserRoot is the default SSH username for nodes
	DefaultNodeUserRoot = "root"
)

type Nodes []v1beta1.Node
type Node v1beta1.Node

type NodesInterface interface {
	Filter(options FilterOptions) Nodes
	Exclude(options FilterOptions) Nodes
	CurrentNode() (v1beta1.Node, error)
	Length() int
	Master() Nodes
	Etcd() Nodes
	Worker() Nodes
	MasterWorker() Nodes
}

// ConvertBKENodeListToNodes converts a BKENodeList to Nodes type
func ConvertBKENodeListToNodes(bkeNodeList *v1beta1.BKENodeList) Nodes {
	if bkeNodeList == nil || len(bkeNodeList.Items) == 0 {
		return Nodes{}
	}
	nodes := make(Nodes, 0, len(bkeNodeList.Items))
	for _, bkeNode := range bkeNodeList.Items {
		nodes = append(nodes, bkeNode.ToNode())
	}
	return nodes
}

// ConvertBKENodesToNodes converts a slice of BKENode to Nodes type
func ConvertBKENodesToNodes(bkeNodes []v1beta1.BKENode) Nodes {
	if len(bkeNodes) == 0 {
		return Nodes{}
	}
	nodes := make(Nodes, 0, len(bkeNodes))
	for _, bkeNode := range bkeNodes {
		nodes = append(nodes, bkeNode.ToNode())
	}
	return nodes
}

// ConvertNodesToBKENodes converts Nodes to a slice of BKENode (Spec only)
func ConvertNodesToBKENodes(nodes Nodes, namespace, clusterName string) []v1beta1.BKENode {
	if len(nodes) == 0 {
		return []v1beta1.BKENode{}
	}
	bkeNodes := make([]v1beta1.BKENode, 0, len(nodes))
	for _, node := range nodes {
		bkeNode := v1beta1.BKENode{
			Spec: v1beta1.FromNode(node),
		}
		// Set name based on cluster name and node IP
		bkeNode.Name = GenerateBKENodeName(clusterName, node.IP)
		bkeNode.Namespace = namespace
		// Set labels for association with BKECluster
		bkeNode.Labels = map[string]string{
			"cluster.x-k8s.io/cluster-name": clusterName,
		}
		bkeNodes = append(bkeNodes, bkeNode)
	}
	return bkeNodes
}

// GenerateBKENodeName generates a unique name for BKENode based on cluster name and node IP
func GenerateBKENodeName(clusterName, nodeIP string) string {
	// Replace dots with dashes for valid k8s resource name
	safeIP := strings.ReplaceAll(nodeIP, ".", "-")
	return fmt.Sprintf("%s-%s", clusterName, safeIP)
}

// SetDefaultsForNodes sets default values (Port, Username) for nodes.
// This is a pure function that returns a new slice with defaults applied.
func SetDefaultsForNodes(nodes Nodes) Nodes {
	if len(nodes) == 0 {
		return nodes
	}
	result := make(Nodes, len(nodes))
	for i, n := range nodes {
		if n.Port == "" {
			n.Port = DefaultNodeSSHPort
		}
		if n.Username == "" {
			n.Username = DefaultNodeUserRoot
		}
		result[i] = n
	}
	return result
}

func (n Nodes) Master() Nodes {
	master := n.Filter(FilterOptions{"Role": MasterNodeRole})
	masterWorker := n.Filter(FilterOptions{"Role": MasterWorkerNodeRole})
	return append(master, masterWorker...)
}

func (n Nodes) Etcd() Nodes {
	return n.Filter(FilterOptions{"Role": EtcdNodeRole})
}

func (n Nodes) Worker() Nodes {
	return n.Filter(FilterOptions{"Role": WorkerNodeRole})
}

func (n Nodes) MasterWorker() Nodes {
	return n.Filter(FilterOptions{"Role": MasterWorkerNodeRole})
}

func NodeInfo(node v1beta1.Node) string {
	if node.Hostname == "" {
		return node.IP
	}
	if node.IP == "" {
		return node.Hostname
	}
	return fmt.Sprintf("%s/%s", node.Hostname, node.IP)
}

type NodeInterface interface {
	IsMaster() bool
	IsWorker() bool
	IsEtcd() bool
	IsMasterWorker() bool
}

func (node Node) IsMaster() bool {
	return utils.SliceContainsString(node.Role, MasterNodeRole)
}

func (node Node) IsWorker() bool {
	return utils.SliceContainsString(node.Role, WorkerNodeRole)
}

func (node Node) IsEtcd() bool {
	return utils.SliceContainsString(node.Role, EtcdNodeRole)
}

func (node Node) IsMasterWorker() bool {
	return utils.SliceContainsString(node.Role, MasterWorkerNodeRole)
}

// CurrentNode return the node which the ip is the same as the local ip
func (n Nodes) CurrentNode() (Node, error) {
	ips, err := bkenet.GetAllInterfaceIP()
	if err != nil {
		return Node{}, errors.Wrapf(err, "get ips from interface failed")
	}
	// 将配置中的节点 IP 转为 net.IP 便于比较
	var nodeIPs []struct {
		ip   net.IP
		node Node
	}
	for _, node := range n {
		parsed := net.ParseIP(node.IP)
		if parsed == nil {
			// 可选：跳过无效 IP，或报错
			continue
		}
		nodeIPs = append(nodeIPs, struct {
			ip   net.IP
			node Node
		}{ip: parsed, node: Node(node)})
	}

	// 遍历本机所有接口 IP
	for _, ipWithCIDR := range ips {
		// 去掉 CIDR 部分，如 "192.168.1.10/24" → "192.168.1.10"
		ipStr := strings.SplitN(ipWithCIDR, "/", 2)[0]
		localIP := net.ParseIP(ipStr)
		if localIP == nil {
			continue // 跳过无法解析的地址
		}

		// 精确比较 IP（支持 IPv4/IPv6）
		for _, ni := range nodeIPs {
			if localIP.Equal(ni.ip) {
				return ni.node, nil
			}
		}
	}
	return Node{}, errors.New("can not find the current node from bkeConfig nodes")
}

func (n Nodes) Decrypt() Nodes {
	newNodes := Nodes{}
	for _, node := range n {
		password, err := security.AesDecrypt(node.Password)
		// 如果解密失败，直接返回原始数据
		if err != nil {
			newNodes = append(newNodes, node)
			continue
		}
		node.Password = password
		newNodes = append(newNodes, node)
	}
	return newNodes
}

type FilterOptions map[string]string

// Filter used to filter nodes match the optional
// if the node match all options,it will be returned
func (n Nodes) Filter(options FilterOptions) Nodes {
	nodes := Nodes{}
	for _, node := range n {
		if matchesOptions(node, options) {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// matchesOptions checks if a node matches all the given filter options
func matchesOptions(node v1beta1.Node, options FilterOptions) bool {
	refVal := reflect.ValueOf(node)
	for key, value := range options {
		field := refVal.FieldByName(key)
		if !field.IsValid() {
			return false
		}
		// 对Role单独处理
		if key == "Role" {
			roles, ok := field.Interface().([]string)
			if !ok {
				return false
			}
			optionRoles := strings.Split(value, ",")
			if !utils.SliceContainsSlice(roles, optionRoles) {
				return false
			}
			continue
		}
		if fmt.Sprint(field.Interface()) != value {
			return false
		}
	}
	return true
}

// nodeKey is used as map key to identify a node by IP and Hostname
type nodeKey struct {
	IP       string
	Hostname string
}

// Exclude used to exclude nodes not match the optional
// if the node not match all options,it will be returned
func (n Nodes) Exclude(options FilterOptions) Nodes {
	filtered := n.Filter(options)
	excludedMap := make(map[nodeKey]bool)
	for _, f := range filtered {
		excludedMap[nodeKey{IP: f.IP, Hostname: f.Hostname}] = true
	}
	nodes := Nodes{}
	for _, node := range n {
		key := nodeKey{IP: node.IP, Hostname: node.Hostname}
		if !excludedMap[key] {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

func (n Nodes) Length() int {
	return len(n)
}
