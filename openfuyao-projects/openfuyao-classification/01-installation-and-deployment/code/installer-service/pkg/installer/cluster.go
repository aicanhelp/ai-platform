/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-ping/ping"
	"github.com/gorilla/websocket"
	version "github.com/hashicorp/go-version"
	"github.com/pkg/errors"
	"golang.org/x/crypto/ssh"
	"gopkg.in/yaml.v3"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	bkecommon "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	pathstore "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgradepath"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"installer-service/pkg/constant"
	"installer-service/pkg/utils"
	"installer-service/pkg/utils/httputil"
	"installer-service/pkg/utils/k8sutil"
	"installer-service/pkg/zlog"
)

const (
	// BKEEventAnnotationKey is the annotation key for BKE event
	BKEEventAnnotationKey = "bke.bocloud.com/event"

	// BKEFinishEventAnnotationKey is the annotation Key for BKE complete event
	BKEFinishEventAnnotationKey = "bke.bocloud.com/complete"

	// SSHConnectTimeout /* ssh登录超时时间 */
	SSHConnectTimeout = 30 * time.Second

	// MaxTimeDiff define max diff seconds
	MaxTimeDiff = 10 * time.Second

	// BaseDecimal number base (base 10)
	BaseDecimal = 10
	// BitSize64 Parse to 64-bit integer
	BitSize64 = 64
)

const decoderBufferSize = 4096

const (
	pathPermission = os.FileMode(0755)
	filePermission = os.FileMode(0644)
)

func (c *installerClient) GetClientSet() kubernetes.Interface {
	return c.clientset
}

func (c *installerClient) GetDynamicClient() dynamic.Interface {
	return c.dynamicClient
}

// decodeToUnstructured 将 rawObj 解码为 *unstructured.Unstructured，并返回其 RESTMapping
func (c *installerClient) decodeToUnstructured(rawObj runtime.RawExtension, restMapper meta.RESTMapper) (
	*unstructured.Unstructured, *meta.RESTMapping, error) {
	obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
	if err != nil {
		return nil, nil, err
	}

	mapping, err := restMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return nil, nil, err
	}

	unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		zlog.Error("runtime.Object convert to unstructured failed")
		return nil, nil, err
	}

	unstruct := &unstructured.Unstructured{Object: unstructuredObj}
	return unstruct, mapping, nil
}

// CreateCluster creates a BKECluster and BKENode CRs from structured request.
func (c *installerClient) CreateCluster(object string) error {
	buffer := bytes.NewBuffer([]byte(object))
	decoder := yamlutil.NewYAMLOrJSONDecoder(buffer, decoderBufferSize)
	dc := c.clientset.Discovery()
	restMapperRes, err := restmapper.GetAPIGroupResources(dc)
	if err != nil {
		return nil
	}
	restMapper := restmapper.NewDiscoveryRESTMapper(restMapperRes)
	for {
		var rawObj runtime.RawExtension
		if err = decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			} else {
				zlog.Error("Decode error1")
				return err
			}
		}
		unstruct, mapping, err := c.decodeToUnstructured(rawObj, restMapper)
		if err != nil {
			return err
		}
		if unstruct.GetKind() == "BKECluster" { // 修改：仅对集群CR应用补丁逻辑
			err = patchVersion(c, unstruct)
			if err != nil {
				zlog.Errorf("patchVersion error")
				return err
			}
			err2 := patchAddonsInfo(c, unstruct)
			if err2 != nil {
				zlog.Errorf("patchAddonsInfo error")
				return err2
			}
			err = patchCoreDNSAntiAffinity(unstruct)
			if err != nil {
				return err
			}
		}
		err = c.createResourcesOnCrateCluster(unstruct, mapping)
		if err != nil {
			return err
		}
	}
	return nil
}

var bkeNodeGVR = schema.GroupVersionResource{
	Group:    configv1beta1.GVK.Group,
	Version:  configv1beta1.GVK.Version,
	Resource: "bkenodes",
}

// createBKENodes creates node CRs in the given namespace
func (c *installerClient) createBKENodes(namespace string, nodes []ClusterNode) error {
	// use global bkeNodeGVR so group/version are centralized
	nodeGVR := bkeNodeGVR
	for _, n := range nodes {
		name := n.Hostname
		if name == "" {
			// fallback name from ip
			name = strings.ReplaceAll(n.Ip, ".", "-")
		}
		un := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "bke.bocloud.com/v1beta1",
				"kind":       "BKENode",
				"metadata": map[string]interface{}{
					"name":      name,
					"namespace": namespace,
					"labels": map[string]interface{}{
						"cluster.x-k8s.io/cluster-name": namespace,
					},
				},
				"spec": map[string]interface{}{
					"hostname": n.Hostname,
					"ip":       n.Ip,
					"port":     n.Port,
					"username": n.Username,
					"password": n.Password,
					// convert role slice to []interface{} to avoid deep-copy panics when
					// client-go dynamic fake attempts to deep-copy unstructured data which
					// does not support []string directly.
					"role": func() []interface{} {
						if len(n.Role) == 0 {
							return nil
						}
						out := make([]interface{}, 0, len(n.Role))
						for _, r := range n.Role {
							out = append(out, r)
						}
						return out
					}(),
				},
			},
		}

		_, err := c.dynamicClient.Resource(nodeGVR).Namespace(namespace).Create(context.Background(), un, metav1.CreateOptions{})
		if err != nil {
			if apierrors.IsAlreadyExists(err) {
				zlog.Infof("BKENode %s already exists in %s", name, namespace)
				continue
			}
			return err
		}
		zlog.Infof("created BKENode %s/%s", namespace, name)
	}
	return nil
}

func (c *installerClient) createResourcesOnCrateCluster(unstruct *unstructured.Unstructured,
	mapping *meta.RESTMapping) error {
	var obj *unstructured.Unstructured
	if unstruct.GetNamespace() == "" {
		obj, err := c.dynamicClient.Resource(mapping.Resource).
			Create(context.Background(), unstruct, metav1.CreateOptions{})
		if err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return err
			}
			zlog.Errorf("%s/%s already exists", unstruct.GetKind(), unstruct.GetName())
			return nil
		}
		zlog.Infof("%s/%s created", obj.GetKind(), obj.GetName())
	} else {
		newNs := unstruct.GetNamespace()
		zlog.Info(newNs)
		nsResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"}
		// 检查命名空间是否存在
		_, err := c.dynamicClient.Resource(nsResource).Get(context.TODO(), newNs, metav1.GetOptions{})
		if err != nil {
			// 命名空间不存在，创建新命名空间
			nsObj := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Namespace",
					"metadata": map[string]interface{}{
						"name": newNs,
					},
				},
			}
			_, err := c.dynamicClient.Resource(nsResource).Create(context.TODO(), nsObj, metav1.CreateOptions{})
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("Namespace %s created successfully\n", newNs)
		} else {
			fmt.Printf("Namespace %s already exists\n", newNs)
		}
		obj, err = c.dynamicClient.Resource(mapping.Resource).Namespace(newNs).Create(context.Background(), unstruct, metav1.CreateOptions{})
		if err != nil {
			if !apierrors.IsAlreadyExists(err) {
				zlog.Error("2:", err.Error())
				return err
			}
			zlog.Errorf("%s/%s already exists", unstruct.GetKind(), unstruct.GetName())
			return nil
		}
		zlog.Infof("%s/%s created", obj.GetKind(), obj.GetName())
	}
	return nil
}

func getNodeInternalIP(node *corev1.Node) (string, error) {
	for _, addr := range node.Status.Addresses {
		if addr.Type == corev1.NodeInternalIP {
			return addr.Address, nil
		}
	}
	return "", errors.New("no internal IP found")
}

func (c *installerClient) collectAllNodeIPs() (map[string]struct{}, error) {
	clusters, err := c.GetClusters()
	if err != nil {
		zlog.Errorf("[JUDGE] get bc clusters err: %v", err)
		return nil, errors.New("get bc cluster err")
	}

	allNodeIp := make(map[string]struct{})
	for _, cluster := range clusters {
		nodes, err := c.GetNodes(cluster.Name)
		if err != nil {
			zlog.Errorf("[JUDGE] get cluster's node err: %v", err)
			continue
		}

		for _, node := range nodes {
			ip, err := getNodeInternalIP(&node)
			if err != nil {
				zlog.Errorf("[JUDGE] get node internal ip err: %v", err)
				continue
			}
			allNodeIp[ip] = struct{}{}
		}
	}
	return allNodeIp, nil
}

func (c *installerClient) IsNodeIpOk(nodeInfo *ClusterNodeInfo) (bool, error) {
	allNodeIp, err := c.collectAllNodeIPs()
	if err != nil {
		return false, err
	}

	for _, node := range nodeInfo.Nodes {
		if _, exist := allNodeIp[node.Ip]; exist {
			return false, errors.New(fmt.Sprintf("%v ip is not ok", node.Ip))
		}
		allNodeIp[node.Ip] = struct{}{}
	}
	return true, nil
}

func (c *installerClient) IsNodeNameOk(nodeInfo *ClusterNodeInfo) (bool, error) {
	allNodeName := make(map[string]struct{})

	nodes, err := c.GetNodes(nodeInfo.NameSpace)
	if err != nil {
		zlog.Errorf("[JUDGE] get cluster's node err: %v", err)
	} else {
		for _, node := range nodes {
			if _, ok := allNodeName[node.Name]; !ok {
				allNodeName[node.Name] = struct{}{}
			}
		}
	}

	for _, ipNode := range nodeInfo.Nodes {
		if _, ok := allNodeName[ipNode.Hostname]; ok {
			return false, errors.New(fmt.Sprintf("%v name is not ok", ipNode.Hostname))
		}
		allNodeName[ipNode.Hostname] = struct{}{}
	}

	return true, nil
}

func getRemoteTime(sshClient *ssh.Client) (time.Time, error) {
	session, err := sshClient.NewSession()
	if err != nil {
		return time.Time{}, err
	}
	defer session.Close()

	output, err := session.Output("date +%s")
	if err != nil {
		return time.Time{}, err
	}

	timestampStr := strings.TrimSpace(string(output))
	timestamp, err := strconv.ParseInt(timestampStr, BaseDecimal, BitSize64)
	if err != nil {
		return time.Time{}, err
	}

	return time.Unix(timestamp, 0), nil
}

func (c *installerClient) IsNodeInfoOk(nodeInfo *ClusterNodeInfo) (bool, error) {
	for _, node := range nodeInfo.Nodes {
		if err := c.checkSingleNode(&node); err != nil {
			return false, err
		}
	}
	return true, nil
}

func (c *installerClient) checkSingleNode(node *ClusterNode) error {
	sshClient, err := createSSHClient(node)
	if err != nil || sshClient == nil {
		zlog.Errorf("[JUDGE] create node: %v ssh client err: %v", node, err)
		return fmt.Errorf("%v create ssh client err", node.Ip)
	}
	defer sshClient.Close()

	return c.validateNodeTime(sshClient, node)
}

func createSSHClient(node *ClusterNode) (*ssh.Client, error) {
	config := &ssh.ClientConfig{
		User:            node.Username,
		Auth:            []ssh.AuthMethod{ssh.Password(node.Password)},
		Timeout:         SSHConnectTimeout,
		HostKeyCallback: func(hostname string, remote net.Addr, key ssh.PublicKey) error { return nil },
	}
	config.SetDefaults()
	address := fmt.Sprintf("%s:%s", node.Ip, node.Port)
	return ssh.Dial("tcp", address, config)
}

func (c *installerClient) validateNodeTime(sshClient *ssh.Client, node *ClusterNode) error {
	localTime := time.Now()
	remoteTime, err := getRemoteTime(sshClient)
	if err != nil {
		zlog.Errorf("[JUDGE] get remote time from node: %v err: %v", node.Ip, err)
		return fmt.Errorf("%v get remote time err", node.Ip)
	}

	zlog.Infof("[JUDGE] get remote time %v and local time %v",
		remoteTime.Format(time.RFC3339), localTime.Format(time.RFC3339))

	timeDiff := localTime.Sub(remoteTime).Abs()
	if timeDiff > MaxTimeDiff {
		zlog.Errorf("[JUDGE] node: %v time diff too large: %v", node.Ip, timeDiff)
		return fmt.Errorf("%v time diff too large: %v", node.Ip, timeDiff)
	}

	zlog.Infof("[JUDGE] node: %v time diff: %v", node.Ip, timeDiff)
	return nil
}

func (c *installerClient) IsBalanceIpOk(balanceIp string) (bool, error) {
	pinger, err := ping.NewPinger(balanceIp)
	if err != nil {
		zlog.Errorf("[JUDGE] create balance ip: %v pinger err: %v", balanceIp, err)
		return false, errors.New(fmt.Sprintf("%v create pinger err", balanceIp))
	}
	// 设置特权模式为true
	pinger.SetPrivileged(true)

	pinger.Count = 1
	pinger.Timeout = 1 * time.Second
	err = pinger.Run()
	if err != nil {
		zlog.Errorf("[JUDGE] run balance ip: %v pinger err: %v", balanceIp, err)
		return false, errors.New(fmt.Sprintf("%v run pinger err", balanceIp))
	}

	stats := pinger.Statistics()
	if stats.PacketsRecv > 0 {
		zlog.Errorf("[JUDGE] ping balance ip: %v ok, but balance ip need not ping ok")
		return false, errors.New(fmt.Sprintf("balance ip %v not ok", balanceIp))
	}
	return true, nil
}

func (c *installerClient) JudgeClusterNode(nodeInfo *ClusterNodeInfo) error {
	zlog.Infof("[JUDGE] request node info: %v", nodeInfo)
	var (
		isOk bool
		err  error
	)

	// 1、节点IP，所有集群中不能出现重复的
	isOk, err = c.IsNodeIpOk(nodeInfo)
	if !isOk {
		zlog.Errorf("[JUDGE] node ip is not ok")
		return err
	}

	// 2、节点名称，集群内不能出现重复的
	isOk, err = c.IsNodeNameOk(nodeInfo)
	if !isOk {
		zlog.Errorf("[JUDGE] node name is not ok")
		return err
	}

	// 3、所有node节点，均是正常的，不能出现虚假ip、ip不可访问的情况
	isOk, err = c.IsNodeInfoOk(nodeInfo)
	if !isOk {
		zlog.Errorf("[JUDGE] node info is not ok")
		return err
	}

	// 4、当创建集群且master节点数大于等于3而且是奇数个，如果传入了负载均衡器IP，那么此IP不能ping通
	if nodeInfo.BalanceIp != "" {
		isOk, err = c.IsBalanceIpOk(nodeInfo.BalanceIp)
		if !isOk {
			zlog.Errorf("[JUDGE] balance ip is not ok")
			return err
		}
	}

	zlog.Infof("[JUDGE] node judge ok")
	return nil
}

func (c *installerClient) DeleteCluster(clusterName string) (*httputil.ResponseJson, int) {
	namespace := clusterName
	zlog.Info(namespace)

	//  获取 BC 资源
	bc, err := c.dynamicClient.Resource(gvr).Namespace(namespace).Get(context.TODO(), clusterName, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("Failed to get custom resource: %v", err)
		return &httputil.ResponseJson{
			Code:    http.StatusInternalServerError,
			Message: fmt.Sprintf("Failed to get cluster resource: %v", err),
		}, http.StatusInternalServerError
	}
	zlog.Info(bc.GetName())
	zlog.Info(bc)

	patchData := map[string]interface{}{
		"metadata": map[string]interface{}{
			"annotations": map[string]interface{}{
				"bke.bocloud.com/ignore-target-cluster-delete": "false",
			},
		},
		"spec": map[string]interface{}{
			"reset": true,
		},
	}
	patchBytes, err := json.Marshal(patchData)
	if err != nil {
		zlog.Errorf("Failed to marshal patch data: %v", err)
		return &httputil.ResponseJson{
			Code:    http.StatusInternalServerError,
			Message: fmt.Sprintf("Failed to marshal patch data: %v", err),
		}, http.StatusInternalServerError
	}

	_, err = c.dynamicClient.Resource(gvr).Namespace(namespace).Patch(
		context.TODO(), clusterName, types.MergePatchType, patchBytes, metav1.PatchOptions{})
	if err != nil {
		zlog.Errorf("Failed to patch custom resource: %v", err)
		return &httputil.ResponseJson{
			Code:    http.StatusInternalServerError,
			Message: fmt.Sprintf("Failed to patch custom resource: %v", err),
		}, http.StatusInternalServerError
	}
	zlog.Info("BKECluster resource patched successfully")

	// 清理状态处理器
	statusProcessors.Delete(clusterName)
	zlog.Debugf("Cleaned up status processor for cluster: %s", clusterName)

	return &httputil.ResponseJson{
			Code:    200,
			Message: "delete the cluster successfully."},
		http.StatusOK
}

// 扩容、升级
func (c *installerClient) PatchYaml(object string, isUpgrade bool) error {
	buffer := bytes.NewBuffer([]byte(object))
	decoder := yamlutil.NewYAMLOrJSONDecoder(buffer, decoderBufferSize)
	dc := c.clientset.Discovery()
	restMapperRes, err := restmapper.GetAPIGroupResources(dc)
	if err != nil {
		return errors.Wrap(err, "get resource failed in PatchYaml")
	}
	restMapper := restmapper.NewDiscoveryRESTMapper(restMapperRes)

	for {
		var rawObj runtime.RawExtension
		if err = decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			} else {
				zlog.Error("Decode rawObj error")
				return err
			}
		}
		unstruct, mapping, err := c.decodeToUnstructured(rawObj, restMapper)
		if err != nil {
			return err
		}
		if isUpgrade {
			err = patchVersion(c, unstruct)
			if err != nil {
				return err
			}
		}
		json, err := unstruct.MarshalJSON()
		if err != nil {
			return err
		}
		_, err = c.dynamicClient.Resource(mapping.Resource).Namespace(unstruct.GetNamespace()).Patch(context.Background(),
			unstruct.GetName(), types.MergePatchType, json, metav1.PatchOptions{})
		if err != nil {
			zlog.Errorf("failed patch %v", unstruct.GroupVersionKind())
			return err
		}
	}
	return nil
}

// 缩容
func (c *installerClient) ScaleDownCluster(object string, ip string) error {
	buffer := bytes.NewBuffer([]byte(object))
	decoder := yamlutil.NewYAMLOrJSONDecoder(buffer, decoderBufferSize)
	dc := c.clientset.Discovery()
	restMapperRes, err := restmapper.GetAPIGroupResources(dc)
	if err != nil {
		return err
	}
	restMapper := restmapper.NewDiscoveryRESTMapper(restMapperRes)
	for {
		var rawObj runtime.RawExtension
		if err = decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break
			} else {
				zlog.Error("Decode failed")
				return err
			}
		}
		unstruct, mapping, err := c.decodeToUnstructured(rawObj, restMapper)
		if err != nil {
			return err
		}
		annotations := unstruct.GetAnnotations()
		if annotations == nil {
			annotations = make(map[string]string)
		}
		annotations["bke.bocloud.com/ignore-target-cluster-delete"] = "false"
		nodeIP := ip //要删除的节点
		zlog.Info(nodeIP, "will delete")
		annotations["bke.bocloud.com/appointment-deleted-nodes"] = nodeIP
		unstruct.SetAnnotations(annotations)
		json, err := unstruct.MarshalJSON()
		if err != nil {
			return err
		}
		_, err = c.dynamicClient.Resource(mapping.Resource).Namespace(unstruct.GetNamespace()).Patch(
			context.Background(), unstruct.GetName(), types.MergePatchType, json, metav1.PatchOptions{})
		if err != nil {
			zlog.Errorf("failed patch %v", unstruct.GroupVersionKind())
			return err
		}
	}
	return nil
}

// DeleteBKENodes deletes BKENode CRs identified by metadata.name, spec.hostname, or spec.ip.
// The console scale-down API passes NodeData.Hostname (often spec.hostname), which may differ
// from the CR metadata.name.
func (c *installerClient) DeleteBKENodes(clusterName string, nodeNames []string) error {
	if c.dynamicClient == nil {
		return fmt.Errorf("dynamic client is nil")
	}
	list, err := c.dynamicClient.Resource(bkeNodeGVR).Namespace(clusterName).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list BKENodes in %s failed: %w", clusterName, err)
	}
	byKey := indexBKENodesForDelete(list.Items)

	var firstErr error
	deleted := 0
	for _, key := range nodeNames {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		crName, ok := byKey[key]
		if !ok {
			err := fmt.Errorf("BKENode %q not found in cluster %s (match by name/hostname/ip)", key, clusterName)
			zlog.Errorf("%v", err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		err := c.dynamicClient.Resource(bkeNodeGVR).Namespace(clusterName).Delete(context.TODO(), crName, metav1.DeleteOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				err = fmt.Errorf("BKENode %s/%s not found", clusterName, crName)
			}
			zlog.Errorf("failed to delete BKENode %s/%s: %v", clusterName, crName, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		deleted++
		zlog.Infof("deleted BKENode %s/%s (requested %q)", clusterName, crName, key)
		// Avoid double-delete if the same CR is referenced by multiple keys in one request.
		delete(byKey, crName)
		for k, v := range byKey {
			if v == crName {
				delete(byKey, k)
			}
		}
	}
	if deleted == 0 && firstErr == nil {
		return fmt.Errorf("no BKENode deleted in cluster %s", clusterName)
	}
	return firstErr
}

// indexBKENodesForDelete maps metadata.name, spec.hostname and spec.ip to CR metadata.name.
func indexBKENodesForDelete(items []unstructured.Unstructured) map[string]string {
	byKey := make(map[string]string, len(items)*3)
	for i := range items {
		u := &items[i]
		crName := u.GetName()
		if crName == "" {
			continue
		}
		byKey[crName] = crName
		if hn, found, _ := unstructured.NestedString(u.Object, "spec", "hostname"); found {
			hn = strings.TrimSpace(hn)
			if hn != "" {
				byKey[hn] = crName
			}
		}
		if ip, found, _ := unstructured.NestedString(u.Object, "spec", "ip"); found {
			ip = strings.TrimSpace(ip)
			if ip != "" {
				byKey[ip] = crName
			}
		}
	}
	return byKey
}

// CreateBKENodes is the exported wrapper that creates BKENode CRs in the given cluster namespace.
func (c *installerClient) CreateBKENodes(clusterName string, nodes []ClusterNode) error {
	return c.createBKENodes(clusterName, nodes)
}

// UpgradeOpenFuyao patches the BKECluster resource to set the desired openFuyaoVersion.
func (c *installerClient) UpgradeOpenFuyao(clusterName string, version string) error {
	if c.dynamicClient == nil {
		return fmt.Errorf("dynamic client is nil")
	}

	patch := map[string]interface{}{
		"spec": map[string]interface{}{
			"clusterConfig": map[string]interface{}{
				"cluster": map[string]interface{}{
					"openFuyaoVersion": version,
				},
			},
		},
	}

	data, err := json.Marshal(patch)
	if err != nil {
		return err
	}

	_, err = c.dynamicClient.Resource(gvr).Namespace(clusterName).Patch(context.Background(), clusterName, types.MergePatchType, data, metav1.PatchOptions{})
	if err != nil {
		zlog.Errorf("failed to patch BKECluster %s openFuyaoVersion: %v", clusterName, err)
		return err
	}
	zlog.Infof("patched BKECluster %s openFuyaoVersion=%s", clusterName, version)
	return nil
}

func (c *installerClient) UpgradeClusterVersion(clusterName string, version string) error {
	if c.dynamicClient == nil {
		return fmt.Errorf("dynamic client is nil")
	}
	if strings.TrimSpace(clusterName) == "" {
		return fmt.Errorf("cluster name is required")
	}
	if strings.TrimSpace(version) == "" {
		return fmt.Errorf("version is required")
	}

	if _, err := c.dynamicClient.Resource(gvr).Namespace(clusterName).Get(context.Background(), clusterName, metav1.GetOptions{}); err != nil {
		return fmt.Errorf("get BKECluster %s/%s failed: %w", clusterName, clusterName, err)
	}

	clusterVersion := map[string]interface{}{
		"apiVersion": upv1alpha1.GroupVersion.String(),
		"kind":       "ClusterVersion",
		"metadata": map[string]interface{}{
			"name":      clusterName,
			"namespace": clusterName,
		},
		"spec": map[string]interface{}{
			"desiredVersion": version,
		},
	}

	_, err := c.dynamicClient.Resource(clusterVersionGVR).Namespace(clusterName).
		Get(context.Background(), clusterName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, createErr := c.dynamicClient.Resource(clusterVersionGVR).Namespace(clusterName).
			Create(context.Background(), &unstructured.Unstructured{Object: clusterVersion}, metav1.CreateOptions{})
		if createErr != nil {
			zlog.Errorf("failed to create ClusterVersion %s desiredVersion: %v", clusterName, createErr)
			return createErr
		}
		zlog.Infof("created ClusterVersion %s desiredVersion=%s", clusterName, version)
		return nil
	}
	if err != nil {
		return err
	}

	patch := map[string]interface{}{
		"spec": map[string]interface{}{
			"desiredVersion": version,
		},
	}
	data, err := json.Marshal(patch)
	if err != nil {
		return err
	}

	_, err = c.dynamicClient.Resource(clusterVersionGVR).Namespace(clusterName).
		Patch(context.Background(), clusterName, types.MergePatchType, data, metav1.PatchOptions{})
	if err != nil {
		zlog.Errorf("failed to patch ClusterVersion %s desiredVersion: %v", clusterName, err)
		return err
	}
	zlog.Infof("patched ClusterVersion %s desiredVersion=%s", clusterName, version)
	return nil
}

// Listen for cluster deployment events
func (c *installerClient) GetClusterLog(namespace string, conn *websocket.Conn) (*httputil.ResponseJson, int) {
	stop := make(chan struct{})
	defer func() {
		if !utils.IsChanClosed(stop) {
			close(stop)
		}
	}()
	clientSet := c.clientset
	watchList := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			ls, err := clientSet.CoreV1().Events(namespace).List(context.Background(), options)
			return ls, err
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			w, err := clientSet.CoreV1().Events(namespace).Watch(context.Background(), options)
			return w, err
		},
	}

	// informer 首次 List 同步完成前，AddFunc 收到的对象来自 ListFunc（历史 events）。
	// 这些历史 events 不应触发 stop close；只有进入 Watch 增量阶段后才按现有逻辑 close。
	var cacheSynced atomic.Bool
	eventHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e, ok := obj.(*corev1.Event)
			if ok {
				if e.Annotations == nil {
					return
				}
				if _, ok = e.Annotations[BKEFinishEventAnnotationKey]; ok {
					message := fmt.Sprintf("Time:%s, Type: %s, Reason: %s, Message: %s \n",
						e.LastTimestamp.Format("2006-01-02 15:04:05"), e.Type, e.Reason, e.Message)
					if err := conn.WriteMessage(websocket.TextMessage, []byte(message)); err != nil {
						log.Println("Error while sending message:", err)
					}
					if cacheSynced.Load() && !utils.IsChanClosed(stop) {
						close(stop)
					}
					return
				}
				if _, ok = e.Annotations[BKEEventAnnotationKey]; ok {
					message := fmt.Sprintf("Time:%s, Type: %s, Reason: %s, Message: %s \n",
						e.LastTimestamp.Format("2006-01-02 15:04:05"), e.Type, e.Reason, e.Message)
					if err := conn.WriteMessage(websocket.TextMessage, []byte(message)); err != nil {
						log.Println("Error while sending message:", err)
					}
				}
			}
		},
		DeleteFunc: func(obj interface{}) {
			if e, ok := obj.(*corev1.Event); ok {
				zlog.Debugf("Event deleted: %s/%s", e.Namespace, e.Name)
			}
		},
		UpdateFunc: func(o, n interface{}) {
			if e, ok := n.(*corev1.Event); ok {
				zlog.Debugf("Event updated: %s/%s", e.Namespace, e.Name)
			}
		},
	}
	_, controller := cache.NewInformer(watchList, &corev1.Event{}, 0, eventHandler)

	// 等待首次 List 同步完成后再允许 finish 事件关闭 stop（即进入 Watch 增量阶段）
	go func() {
		if cache.WaitForCacheSync(stop, controller.HasSynced) {
			cacheSynced.Store(true)
		}
	}()

	controller.Run(stop)
	return &httputil.ResponseJson{Code: 200, Message: "get the log of cluster."}, http.StatusOK
}

var gvr = schema.GroupVersionResource{
	Group:    configv1beta1.GVK.Group,
	Version:  configv1beta1.GVK.Version,
	Resource: "bkeclusters",
}

var upgradePathGVR = schema.GroupVersionResource{
	Group:    upv1alpha1.GroupVersion.Group,
	Version:  upv1alpha1.GroupVersion.Version,
	Resource: "upgradepaths",
}

var clusterVersionGVR = schema.GroupVersionResource{
	Group:    upv1alpha1.GroupVersion.Group,
	Version:  upv1alpha1.GroupVersion.Version,
	Resource: "clusterversions",
}

// 获得所有bc
func (c *installerClient) GetClusters() ([]configv1beta1.BKECluster, error) {
	var workloadUnstructured *unstructured.UnstructuredList
	var err error
	dynamicClient := c.dynamicClient
	workloadUnstructured, err = dynamicClient.Resource(gvr).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		zlog.Error(err)
		return nil, err
	}
	bclusterlist := &configv1beta1.BKEClusterList{}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(workloadUnstructured.UnstructuredContent(), bclusterlist)
	if err != nil {
		zlog.Error(err)
		return nil, err
	}
	zlog.Info("get all the bc")
	zlog.Infof("cluster list: %v", bclusterlist.Items)
	return bclusterlist.Items, nil
}

func (c *installerClient) GetClustersByName(name string) (*configv1beta1.BKECluster, error) {
	var workloadUnstructured *unstructured.Unstructured
	var err error
	dynamicClient := c.dynamicClient
	workloadUnstructured, err = dynamicClient.Resource(gvr).Namespace(name).Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("not found %s bc,return the nil", name, err)
		return nil, err
	}
	bcluster := &configv1beta1.BKECluster{}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(workloadUnstructured.UnstructuredContent(), bcluster)
	if err != nil {
		zlog.Error(err)
		return nil, err
	}
	return bcluster, nil
}

// GetClustersByQuery 获取集群信息
func (c *installerClient) GetClustersByQuery(req ClusterRequest) (ClusterResponse, error) {
	res := ClusterResponse{}
	clusters, err := c.GetClusters()
	if err != nil {
		zlog.Errorf("Failed to get clusters: %v", err)
		return res, err
	}

	cleanupStatusProcessors(clusters) // 清理孤儿状态处理器
	items := make([]*ClusterData, 0, len(clusters))

	for _, cluster := range clusters {
		// 1. 名称过滤
		if req.Name != "" && cluster.Name != req.Name {
			continue
		}

		// 2. 获取状态处理器
		processor, _ := statusProcessors.LoadOrStore(cluster.Name, NewStatusProcessor())
		statusProcessor, ok := processor.(*StatusProcessor)
		if !ok {
			statusProcessor = NewStatusProcessor()
			statusProcessors.Store(cluster.Name, statusProcessor)
		}

		// 3. 计算映射状态，再做状态过滤
		mappedStatus := statusProcessor.getStatus(cluster.Status)
		if req.Status != "" && mappedStatus != req.Status {
			continue
		}

		// 4. 创建基础数据
		createTime, _ := utils.ConvertToChinaTime(cluster.CreationTimestamp)
		// determine node count: count BKENode CRs in the cluster namespace
		nodeSum := 0
		if cnt, err := c.getBKENodeCount(cluster.Name); err == nil {
			nodeSum = cnt
		}

		baseData := &ClusterData{
			Name:              cluster.Name,
			OpenFuyaoVersion:  cluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion,
			KubernetesVersion: cluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
			ContainerdVersion: cluster.Spec.ClusterConfig.Cluster.ContainerdVersion,
			Status:            mappedStatus,
			CreateTime:        createTime,
			NodeSum:           nodeSum,
			ContainerRuntime:  cluster.Spec.ClusterConfig.Cluster.ContainerRuntime.CRI,
			IsAmd:             false,
			IsArm:             false,
			OsList:            []string{},
		}

		var addons []string
		if extracted, err := utils.ExtractAddons(cluster); err != nil {
			zlog.Error(err)
		} else {
			addons = extracted
		}
		baseData.IsAdminCluster = utils.IsAdminCluster(cluster, addons)

		// 4. 处理节点数据
		if nodes, err := c.GetNodes(cluster.Name); err != nil {
			zlog.Warnf("Cluster %s creating, nodes unavailable: %v", cluster.Name, err)
			items = append(items, baseData)
		} else {
			items = append(items, c.processNodeData(baseData, nodes))
		}
	}

	res.Items = items
	return res, nil
}

// GetAllClusters 返回所有集群的 ClusterResponse，用于 GET /clusters
func (c *installerClient) GetAllClusters() (ClusterResponse, error) {
	// reuse GetClustersByQuery with empty request to avoid duplicating mapping logic
	return c.GetClustersByQuery(ClusterRequest{})
}

// getBKENodeCount counts BKENode CRs in the given namespace. Returns error if list fails.
func (c *installerClient) getBKENodeCount(namespace string) (int, error) {
	if c.dynamicClient == nil {
		return 0, fmt.Errorf("dynamic client is nil")
	}
	selector := fmt.Sprintf("cluster.x-k8s.io/cluster-name=%s", namespace)
	list, err := c.dynamicClient.Resource(bkeNodeGVR).Namespace(namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return 0, err
	}
	return len(list.Items), nil
}

// 处理节点数据（独立函数）
func (c *installerClient) processNodeData(base *ClusterData, nodes []corev1.Node) *ClusterData {
	data := *base // 复制基础数据
	data.NodeSum = len(nodes)

	osSet := make(map[string]struct{})
	for _, node := range nodes {
		switch node.Status.NodeInfo.Architecture {
		case "amd64":
			data.IsAmd = true
		case "arm64":
			data.IsArm = true
		default:
			// ignore unknown/empty architecture
		}

		osImage := node.Status.NodeInfo.OSImage
		if osImage == "" {
			continue
		}
		if _, exists := osSet[osImage]; !exists {
			osSet[osImage] = struct{}{}
			data.OsList = append(data.OsList, osImage)
		}
	}

	return &data
}

func getVersion(cluster configv1beta1.BKECluster) string {
	// 未升级完仍显示原来k8s版本
	if cluster.Status.ClusterStatus == "Upgrading" {
		res := "v1.28.8"
		return res
	}
	return cluster.Spec.ClusterConfig.Cluster.KubernetesVersion
}

// StatusProcessor 维护集群状态映射
type StatusProcessor struct{}

// NewStatusProcessor 进行初始化赋值
func NewStatusProcessor() *StatusProcessor {
	return &StatusProcessor{}
}

func (p *StatusProcessor) getStatus(status configv1beta1.BKEClusterStatus) string {
	return p.mapClusterStatus(status.ClusterStatus)
}

func (p *StatusProcessor) mapClusterStatus(cs configv1beta1.ClusterStatus) string {
	switch cs {
	// 扩缩容进行中（透传）
	case "ScalingWorkerNodesUp":
		return "ScalingWorkerNodesUp"
	case "ScalingWorkerNodesDown":
		return "ScalingWorkerNodesDown"
	case "ScalingMasterNodesUp":
		return "ScalingMasterNodesUp"
	case "ScalingMasterNodesDown":
		return "ScalingMasterNodesDown"

	// 扩缩容失败 → 统一 ScaleFailed
	case "MasterScalingUpFailed", "MasterScalingDownFailed",
		"WorkerScalingUpFailed", "WorkerScalingDownFailed":
		return "ScaleFailed"

	// 删除
	case "Deleting":
		return "Deleting"
	case "DeleteFailed":
		return "DeleteFailed"

	// 安装/初始化
	case "Initializing", "DeployingAddon":
		return "Deploying"
	case "InitializationFailed", "DeployAddonFailed":
		return "DeployFailed"

	// 升级
	case "Upgrading":
		return "Upgrading"
	case "UpgradeFailed":
		return "UpgradeFailed"

	// 管理 → 健康
	case "Managing", "ManageFailed":
		return "Healthy"

	// 健康
	case "Ready":
		return "Healthy"
	case "Unhealthy":
		return "Unhealthy"

	// 未映射状态：原样透传
	default:
		return string(cs)
	}
}

// 全局状态处理器映射（并发安全）
var statusProcessors sync.Map

// 清理不再存在的集群的状态处理器
func cleanupStatusProcessors(existingClusters []configv1beta1.BKECluster) {
	existingMap := make(map[string]bool)
	for _, cluster := range existingClusters {
		existingMap[cluster.Name] = true
	}

	statusProcessors.Range(func(key, value interface{}) bool {
		clusterName, ok := key.(string) // 安全类型断言
		if !ok {
			// 记录错误并跳过无效条目
			zlog.Errorf("Invalid key type in statusProcessors: expected string, got %T", key)
			return true // 继续遍历
		}
		if !existingMap[clusterName] {
			statusProcessors.Delete(clusterName)
			zlog.Debugf("Cleaned up orphaned status processor for cluster: %s", clusterName)
		}
		return true
	})
}

func (c *installerClient) GetNodes(clusterName string) ([]corev1.Node, error) {
	// Always list from management-cluster BKENode CRs.
	// Remote corev1.Node only contains successfully joined nodes and would hide
	// Failed/Pending BKENodes from the console (openFuyao/sig-installation#186).
	if c.dynamicClient == nil {
		return nil, fmt.Errorf("dynamic client is nil")
	}
	list, err := c.dynamicClient.Resource(bkeNodeGVR).Namespace(clusterName).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	zlog.Infof("cluster %s has BKENodes number: %v", clusterName, len(list.Items))
	out := make([]corev1.Node, 0, len(list.Items))
	for i := range list.Items {
		out = append(out, bkeNodeToCoreNode(&list.Items[i]))
	}
	// BKENode has no OS/arch/capacity; fill those from remote Nodes when reachable.
	c.enrichNodesFromRemote(clusterName, out)
	return out, nil
}

// bkeNodeToCoreNode synthesizes a corev1.Node view from a BKENode CR for API responses.
func bkeNodeToCoreNode(u *unstructured.Unstructured) corev1.Node {
	n := corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   u.GetName(),
			Labels: map[string]string{},
		},
	}
	if hn, found, _ := unstructured.NestedString(u.Object, "spec", "hostname"); found && hn != "" {
		n.ObjectMeta.Name = hn
	}
	if ip, found, _ := unstructured.NestedString(u.Object, "spec", "ip"); found && ip != "" {
		n.Status.Addresses = append(n.Status.Addresses, corev1.NodeAddress{Type: corev1.NodeInternalIP, Address: ip})
	}
	if roles, found, _ := unstructured.NestedStringSlice(u.Object, "spec", "role"); found {
		for _, role := range roles {
			lower := strings.ToLower(role)
			if strings.Contains(lower, "master") || strings.Contains(lower, "control-plane") {
				n.Labels[constant.NodeRoleMasterLabel] = "true"
			}
			if strings.Contains(lower, "node") || strings.Contains(lower, "worker") {
				n.Labels[constant.NodeRoleNodeLabel] = "true"
			}
		}
	}
	// Map BKENode status.state onto NodeReady; keep raw state in Reason for getNodeStatus().
	condStatus := corev1.ConditionUnknown
	state, _, _ := unstructured.NestedString(u.Object, "status", "state")
	switch state {
	case string(configv1beta1.NodeReady):
		condStatus = corev1.ConditionTrue
	case "":
		condStatus = corev1.ConditionUnknown
	default:
		condStatus = corev1.ConditionFalse
	}
	n.Status.Conditions = []corev1.NodeCondition{{
		Type:   corev1.NodeReady,
		Status: condStatus,
		Reason: state,
	}}
	return n
}

// enrichNodesFromRemote copies OS/arch and capacity from joined remote Nodes onto the
// BKENode-synthesized list. Failures are ignored so Failed/Pending nodes still show.
func (c *installerClient) enrichNodesFromRemote(clusterName string, nodes []corev1.Node) {
	if len(nodes) == 0 {
		return
	}
	cluster, err := c.GetClustersByName(clusterName)
	if err != nil {
		zlog.Warnf("skip remote node enrich for %s: get cluster failed: %v", clusterName, err)
		return
	}
	remoteClient, err := c.NewRemoteClusterClient(cluster)
	if err != nil {
		zlog.Warnf("skip remote node enrich for %s: remote client failed: %v", clusterName, err)
		return
	}
	nodeList, err := remoteClient.GetClientSet().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		zlog.Warnf("skip remote node enrich for %s: list remote nodes failed: %v", clusterName, err)
		return
	}
	mergeRemoteNodeInfo(nodes, nodeList.Items)
}

// mergeRemoteNodeInfo fills NodeInfo/Capacity/Allocatable by hostname or InternalIP match.
func mergeRemoteNodeInfo(nodes []corev1.Node, remoteNodes []corev1.Node) {
	byName, byIP := indexRemoteNodes(remoteNodes)
	for i := range nodes {
		rn, ok := lookupRemoteNode(nodes[i], byName, byIP)
		if !ok {
			continue
		}
		applyRemoteNodeInfo(&nodes[i], rn)
	}
}

func indexRemoteNodes(remoteNodes []corev1.Node) (map[string]corev1.Node, map[string]corev1.Node) {
	byName := make(map[string]corev1.Node, len(remoteNodes))
	byIP := make(map[string]corev1.Node, len(remoteNodes))
	for _, rn := range remoteNodes {
		byName[rn.Name] = rn
		if ip := nodeInternalIP(rn); ip != "" {
			byIP[ip] = rn
		}
	}
	return byName, byIP
}

func nodeInternalIP(node corev1.Node) string {
	for _, addr := range node.Status.Addresses {
		if addr.Type == corev1.NodeInternalIP && addr.Address != "" {
			return addr.Address
		}
	}
	return ""
}

func lookupRemoteNode(node corev1.Node, byName, byIP map[string]corev1.Node) (corev1.Node, bool) {
	if rn, ok := byName[node.Name]; ok {
		return rn, true
	}
	if ip := nodeInternalIP(node); ip != "" {
		rn, ok := byIP[ip]
		return rn, ok
	}
	return corev1.Node{}, false
}

func applyRemoteNodeInfo(dst *corev1.Node, src corev1.Node) {
	dst.Status.NodeInfo = src.Status.NodeInfo
	if len(src.Status.Capacity) > 0 {
		dst.Status.Capacity = src.Status.Capacity
	}
	if len(src.Status.Allocatable) > 0 {
		dst.Status.Allocatable = src.Status.Allocatable
	}
}

func (c *installerClient) NewRemoteClusterClient(bkeCluster *configv1beta1.BKECluster) (Operation, error) {
	if !bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		msg := fmt.Sprintf("failed use k8s token create remote cluster client, "+
			"BKECluster %q controlPlaneEndpoint is invalid",
			fmt.Sprintf("%s/%s", bkeCluster.GetNamespace(), bkeCluster.GetName()))
		zlog.Error(msg)
		return nil, errors.New(msg)
	}

	restConfig, err := c.GetRestConfigByToken(bkeCluster)
	if err != nil {
		zlog.Error(err)
		return nil, err
	}

	ret, err := NewInstallerOperation(restConfig)
	if err != nil {
		msg := fmt.Sprintf("failed to create client for Cluster %s/%s", bkeCluster.Namespace, bkeCluster.Name)
		zlog.Error(err, msg)
		return nil, errors.Wrap(err, msg)
	}
	return ret, nil
}

// getRestConfigByToken get rest config by token
func (c *installerClient) GetRestConfigByToken(bkeCluster *configv1beta1.BKECluster) (*rest.Config, error) {
	secret, err := k8sutil.GetSecret(c.clientset,
		fmt.Sprintf("%s-k8s-token", bkeCluster.Name), bkeCluster.Namespace)
	if err != nil {
		zlog.Error(err)
		return nil, err
	}
	token, ok := secret.Data["token"]
	if !ok || string(token) == "" {
		msg := fmt.Sprintf("token data in secret %q not found",
			fmt.Sprintf("%s/%s", bkeCluster.GetNamespace(), bkeCluster.GetName()))
		zlog.Error(msg)
		return nil, errors.New(msg)
	}

	config := &rest.Config{
		Host:            fmt.Sprintf("https://%s", bkeCluster.Spec.ControlPlaneEndpoint.String()),
		BearerToken:     string(token),
		BearerTokenFile: "",
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}
	return config, nil
}

func BKEConfigCmKey() client.ObjectKey {
	return client.ObjectKey{
		Namespace: "cluster-system",
		Name:      bkecommon.BKEClusterConfigFileName,
	}
}

func parseImageRepo(data map[string]string) configv1beta1.Repo {
	const len2 = 2
	repo := configv1beta1.Repo{Domain: data["domain"]}
	if data["otherRepo"] != "" {
		img := strings.Split(data["otherRepo"], "/")
		img1 := strings.Split(img[0], ":")
		port := "443"
		if len(img1) == len2 {
			port = img1[1]
		}
		repo = configv1beta1.Repo{
			Domain: img1[0],
			Ip:     data["otherRepoIp"],
			Port:   port,
			Prefix: strings.TrimRight(strings.Join(img[1:], "/"), "/"),
		}
	} else if data["onlineImage"] != "" {
		img := strings.Split(data["onlineImage"], "/")
		img1 := strings.Split(img[0], ":")
		port := "443"
		if len(img1) == len2 {
			port = img1[1]
		}
		repo = configv1beta1.Repo{
			Domain: img1[0],
			Ip:     data["otherRepoIp"],
			Port:   port,
		}
		if repo.Ip == "" {
			repo.Ip = data["host"]
		}
	} else {
		// neither otherRepo nor onlineImage is configured, keep default repo built from domain
	}
	return repo
}

func parseHttpRepo(data map[string]string) configv1beta1.Repo {
	const len2 = 2
	repo2 := configv1beta1.Repo{Domain: configinit.DefaultYumRepo}
	if data["otherSource"] != "" {
		httpRepo := strings.TrimLeft(data["otherSource"], "http://")
		httpRepoArray := strings.Split(httpRepo, ":")
		port := "80"
		if len(httpRepoArray) == len2 {
			port = httpRepoArray[1]
		}
		repo2 = configv1beta1.Repo{
			Domain: configinit.DefaultYumRepo,
			Port:   port,
		}
		if net.ParseIP(httpRepoArray[0]) == nil {
			repo2 = configv1beta1.Repo{Ip: httpRepoArray[0]}
		} else {
			repo2 = configv1beta1.Repo{Domain: httpRepoArray[0]}
		}
	}
	return repo2
}

func (c *installerClient) GetDefaultConfig() (DefaultResp, error) {
	res := DefaultResp{}
	configMap, err := k8sutil.GetConfigMap(c.clientset, BKEConfigCmKey().Name, BKEConfigCmKey().Namespace)
	if err != nil {
		zlog.Error(err)
		return res, err
	}
	data := configMap.Data
	res.ImageRepo = parseImageRepo(data)
	res.HttpRepo = parseHttpRepo(data)
	res.Ip = data["host"]
	res.ContainerRuntime = data["runtime"]
	if strings.TrimSpace(res.ContainerRuntime) == "" {
		res.ContainerRuntime = "containerd"
	}
	res.KubernetesVersion = data["kubernetesVersion"]
	res.AgentHealthPort = data["agentHealthPort"]
	zlog.Info(res.Ip)
	return res, nil
}

func (c *installerClient) GetClusterConfig(clusterName string) (ClusterConfig, error) {
	res := ClusterConfig{}
	name := clusterName
	zlog.Info(name)
	cluster, err := c.GetClustersByName(name)
	if err != nil {
		zlog.Error("not get this cluster info")
		zlog.Error(err)
		return res, err
	}
	processor, _ := statusProcessors.LoadOrStore(cluster.Name, NewStatusProcessor())
	var statusProcessor *StatusProcessor
	if sp, ok := processor.(*StatusProcessor); ok {
		statusProcessor = sp
	} else {
		statusProcessor = NewStatusProcessor()
		statusProcessors.Store(cluster.Name, statusProcessor)
	}

	res.ClusterName = clusterName
	// 确保状态处理器不为nil后再调用方法
	if statusProcessor != nil {
		res.ClusterStatus = statusProcessor.getStatus(cluster.Status)
	} else {
		// 极端情况处理
		zlog.Error("Critical: statusProcessor is nil for cluster ", clusterName)
		res.ClusterStatus = "Unknown" // 提供默认值
	}
	res.CreateTime, err = utils.ConvertToChinaTime(cluster.CreationTimestamp)
	if err != nil {
		zlog.Errorf("ConvertToChinaTime failed", err)
	}
	// populate OpenFuyaoVersion from cluster spec if available
	if cluster.Spec.ClusterConfig != nil {
		res.OpenFuyaoVersion = cluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion
	}
	res.Host = cluster.Spec.ControlPlaneEndpoint.Host
	res.IsHA, _ = c.isHA(name)
	res.DefaultResp, _ = c.GetDefaultConfig()
	return res, nil
}

// isHA判断是否是高可用集群
func (c *installerClient) isHA(clusterName string) (bool, error) {
	cluster, err := c.GetClustersByName(clusterName)
	if err != nil {
		zlog.Error("not get this cluster info")
		zlog.Error(err)
		return false, err
	}
	host := cluster.Spec.ControlPlaneEndpoint.Host
	// after splitting BKENode from BKECluster, nodes are stored as BKENode CRs
	if c.dynamicClient == nil {
		return true, fmt.Errorf("dynamic client is nil")
	}
	selector := fmt.Sprintf("cluster.x-k8s.io/cluster-name=%s", clusterName)
	list, err := c.dynamicClient.Resource(bkeNodeGVR).Namespace(clusterName).List(context.TODO(), metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		zlog.Errorf("failed to list BKENodes for %s: %v", clusterName, err)
		return true, err
	}
	for _, u := range list.Items {
		ip, _, _ := unstructured.NestedString(u.Object, "spec", "ip")
		if ip == host {
			return false, nil
		}
	}
	return true, nil
}

func (c *installerClient) GetNodesByQuery(req NodeRequest) (NodeResponse, error) {
	res := NodeResponse{}
	items := make([]NodeData, 0)

	// 获取节点
	nodes, err := c.GetNodes(req.ClusterName)
	if err != nil {
		zlog.Errorf("cluster is creating: %v", err)
		return res, err
	}

	// 处理节点数据
	items = c.processNodes(req, nodes)

	res.Items = items
	return res, nil
}

// GetClusterFull 返回合并后的集群配置信息和节点列表，供 API handler 直接使用
func (c *installerClient) GetClusterFull(clusterName string) (ClusterFullResponse, error) {
	var out ClusterFullResponse

	// 1. cluster config
	cfg, err := c.GetClusterConfig(clusterName)
	if err != nil {
		return out, err
	}

	// 2. nodes
	nodeReq := NodeRequest{ClusterName: clusterName}
	nodesResp, err := c.GetNodesByQuery(nodeReq)
	if err != nil {
		return out, err
	}

	// reuse ClusterConfig and NodeData shapes
	out.ClusterConfig = cfg
	out.Nodes = nodesResp.Items

	return out, nil
}

// processNodes 处理节点转换逻辑
func (c *installerClient) processNodes(req NodeRequest, nodes []corev1.Node) []NodeData {
	items := make([]NodeData, 0, len(nodes))
	for _, node := range nodes {
		if req.NodeName != "" && node.Name != req.NodeName {
			continue
		}

		ip := c.getNodeIP(node)
		status := c.getNodeStatus(node)
		role := c.getNodeRole(node)
		cpu, memGB := c.getNodeResources(node)

		items = append(items, NodeData{
			Hostname:     node.Name,
			Ip:           ip,
			Cpu:          cpu,
			Memory:       memGB,
			Status:       status,
			Architecture: node.Status.NodeInfo.Architecture,
			Os:           node.Status.NodeInfo.OSImage,
			Role:         role,
		})
	}
	return items
}

// 获取节点IP
func (c *installerClient) getNodeIP(node corev1.Node) string {
	for _, addr := range node.Status.Addresses {
		if addr.Type == corev1.NodeInternalIP {
			return addr.Address
		}
	}
	return ""
}

// 获取节点状态
func (c *installerClient) getNodeStatus(node corev1.Node) string {
	for _, condition := range node.Status.Conditions {
		if condition.Type != corev1.NodeReady {
			continue
		}
		if condition.Status == corev1.ConditionTrue {
			return "Ready"
		}
		// Prefer BKENode status.state carried in Reason (Failed/Pending/...).
		if condition.Reason != "" && condition.Reason != string(configv1beta1.NodeReady) {
			return condition.Reason
		}
		if condition.Status == corev1.ConditionUnknown {
			return "Unknown"
		}
		return "NotReady"
	}
	return "Unknown"
}

// 获取节点角色
func (c *installerClient) getNodeRole(node corev1.Node) []string {
	var roles []string
	if _, ok := node.Labels[constant.NodeRoleMasterLabel]; ok {
		roles = append(roles, "master")
	}
	if _, ok := node.Labels[constant.NodeRoleNodeLabel]; ok {
		roles = append(roles, "node")
	}
	return roles
}

// 计算节点资源
func (c *installerClient) getNodeResources(node corev1.Node) (int64, float64) {
	cpu := node.Status.Capacity[corev1.ResourceCPU]
	memory := node.Status.Capacity[corev1.ResourceMemory]
	memGB := math.Round(float64(memory.Value()) / (1024 * 1024 * 1024))
	return cpu.Value(), memGB
}

func (c *installerClient) UploadPatchFile(fileName string, fileContent string) error {
	configMap, err := k8sutil.GetConfigMap(c.clientset, BKEConfigCmKey().Name, BKEConfigCmKey().Namespace)
	if err != nil {
		return err
	}

	// If cluster is online mode, patch file should be fetched from online openFuyao repo instead
	// of uploading patch file. Raise error in this case.
	if online, err := IsOnlineMode(configMap); err != nil {
		return err
	} else if online {
		err = errors.New("online mode detected, upload patch file not supported")
		return err
	} else {
		// offline mode, continue to upload the patch file
	}

	// unmarshal yaml to get openFuyaoVersion
	var config PatchVersionInfo
	buffer := bytes.NewBufferString(fileContent)
	decoder := yamlutil.NewYAMLOrJSONDecoder(buffer, decoderBufferSize)
	if err := decoder.Decode(&config); err != nil && err != io.EOF {
		zlog.Errorf("Failed to parse YAML content: %v", err)
		return fmt.Errorf("failed to parse YAML content: %v", err)
	}

	err = validatePatchVersionInfo(&config)
	if err != nil {
		return err
	}
	version := strings.TrimSuffix(fileName, filepath.Ext(fileName))
	patchKey, patchValue, err := addPatchesToConfigMap(c.clientset, version, fileContent)
	if err != nil {
		return err
	}

	// update patchFile field in configMap.Data and save to cluster
	err = c.addPatchInfoToConfigMap(configMap, patchKey, patchValue)
	if err != nil {
		return err
	}

	zlog.Infof("Successfully uploaded patch file %s for %s:%s", fileName, patchKey, patchValue)
	return nil
}

func (c *installerClient) getVersionsFromSource(isUpgrade bool) ([]string, error) {
	configMap, err := k8sutil.GetConfigMap(c.clientset, BKEConfigCmKey().Name, BKEConfigCmKey().Namespace)
	if err != nil {
		return []string{}, err
	}

	if online, err := IsOnlineMode(configMap); err != nil {
		return []string{}, err
	} else if online {
		return c.getVersionsFromOnline(configMap, isUpgrade)
	} else {
		// offline mode, read versions from configmap patches
		return c.getVersionsFromOffline(configMap)
	}
}

func (c *installerClient) getVersionsFromOnline(configMap *corev1.ConfigMap, isUpgrade bool) ([]string, error) {
	var remotePath string
	if isUpgrade {
		remotePath = getRemotePatchPath()
	} else {
		remotePath = getRemoteDeployPath()
	}

	patchResponse, err := getPatchIndexFromRemoteRepo(http.DefaultClient, remotePath)
	if err != nil {
		return []string{}, fmt.Errorf("failed to fetch index.yaml: %v", err)
	}

	if err = c.cleanConfigMapPatches(configMap); err != nil {
		return []string{}, err
	}

	var versions []string
	for _, value := range patchResponse {
		versions = append(versions, value["openFuyaoVersion"])

		yamlData, err := getRemoteContent(http.DefaultClient, remotePath, value["filePath"])
		if err != nil {
			return []string{}, err
		}

		cleanData := bytes.ReplaceAll(yamlData, []byte("\r\n"), []byte("\n"))
		cleanData = bytes.ReplaceAll(cleanData, []byte("\r"), []byte("\n"))

		patchKey, patchValue, err := addPatchesToConfigMap(c.clientset, value["openFuyaoVersion"], string(cleanData))
		if err != nil {
			return []string{}, err
		}

		if err := c.addPatchInfoToConfigMap(configMap, patchKey, patchValue); err != nil {
			return []string{}, err
		}
	}

	return versions, nil
}

func (c *installerClient) getVersionsFromOffline(configMap *corev1.ConfigMap) ([]string, error) {
	var versions []string
	for key := range configMap.Data {
		if after, ok := strings.CutPrefix(key, constant.PatchKeyPrefix); ok {
			versions = append(versions, after)
		}
	}
	return versions, nil
}

// GetUpgradeVersions get upgrade version list.
func (c *installerClient) GetUpgradeVersions() ([]string, error) {
	return c.getVersionsFromSource(true)
}

// GetDeployVersions get deploy version list.
func (c *installerClient) GetDeployVersions() ([]string, error) {
	return c.getVersionsFromSource(false)
}

func (c *installerClient) GetOpenFuyaoVersions() ([]string, error) {
	versionList, err := c.GetDeployVersions()
	if err != nil {
		return []string{}, err
	}

	// 筛选可安装的version list
	var result []string
	for _, vStr := range versionList {
		if vStr == "latest" {
			result = append(result, vStr)
			continue
		}
		v, err := version.NewVersion(vStr)
		if err != nil {
			continue
		}

		segments := v.Segments64()
		isPatch := false

		if len(segments) >= MinSemverSegments && segments[PatchVersionIndex] > 0 {
			isPatch = true
		} else if len(segments) > MinSemverSegments {
			isPatch = true
		} else {
			// no non-zero patch segment, not a patch version, keep isPatch as false
		}

		if !isPatch {
			result = append(result, vStr)
		}
	}
	return result, nil
}

func (c *installerClient) newUpgradePathService() (*pathstore.Service, error) {
	if c.dynamicClient == nil {
		return nil, fmt.Errorf("dynamic client is nil")
	}

	list, err := c.dynamicClient.Resource(upgradePathGVR).List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var paths []upv1alpha1.UpgradePathRule
	var versions []upv1alpha1.VersionEntry
	digest := ""
	for i := range list.Items {
		upgradePath := &upv1alpha1.UpgradePath{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(list.Items[i].UnstructuredContent(), upgradePath); err != nil {
			return nil, err
		}
		paths = append(paths, upgradePath.Spec.Paths...)
		versions = append(versions, upgradePath.Spec.Versions...)
		if annotations := upgradePath.GetAnnotations(); annotations != nil {
			if value := annotations[pathstore.OCIDigestAnnotation]; value != "" {
				digest = value
			}
		}
	}

	service := pathstore.NewService()
	if err := service.Load(paths, versions, digest); err != nil {
		return nil, err
	}
	return service, nil
}

func (c *installerClient) GetInstallableVersions() ([]string, error) {
	service, err := c.newUpgradePathService()
	if err != nil {
		return nil, err
	}
	return service.GetInstallableVersions(), nil
}

func (c *installerClient) GetOpenFuyaoUpgradeVersions(currentVersion string) ([]string, error) {
	if currentVersion == "latest" {
		return []string{}, nil
	}
	versionList, err := c.GetUpgradeVersions()
	if err != nil {
		return []string{}, err
	}
	// 筛选可升级的version list
	current, err := version.NewVersion(currentVersion)
	if err != nil {
		return nil, fmt.Errorf("invalid current version: %w", err)
	}

	var result []string
	for _, vStr := range versionList {
		v, err := version.NewVersion(vStr)
		if err != nil {
			continue
		}

		if !v.GreaterThan(current) {
			continue
		}
		curSegments := current.Segments64() // [25, 9, 0]
		vSegments := v.Segments64()         // e.g. [25, 10, 2]  [26, 3, 0]

		// 1、Major和Minor版本相同，就说明是补丁版本，那么新版本大于旧版本，可直接加入
		// 2、Major版本相同，Minor版本不同，需要新版本是可直接安装的版本，才能加入
		// 3、Major版本不同，需要新版本是可直接安装的版本，才能加入
		if curSegments[0] == vSegments[0] && curSegments[1] == vSegments[1] {
			result = append(result, vStr)
		} else if curSegments[0] == vSegments[0] && curSegments[1] < vSegments[1] {
			if len(vSegments) < MinSemverSegments || vSegments[PatchVersionIndex] == 0 {
				result = append(result, vStr)
			}
		} else if curSegments[0] < vSegments[0] {
			if len(vSegments) < MinSemverSegments || vSegments[PatchVersionIndex] == 0 {
				result = append(result, vStr)
			}
		} else {
			// target version is older than the current version, skip it
		}
	}
	return result, nil
}

func (c *installerClient) GetUpgradeableVersions(currentVersion string) ([]string, error) {
	service, err := c.newUpgradePathService()
	if err != nil {
		return nil, err
	}
	return service.GetUpgradeableVersions(currentVersion), nil
}

func (c *installerClient) cleanConfigMapPatches(cm *corev1.ConfigMap) error {
	for key := range cm.Data {
		if strings.HasPrefix(key, constant.PatchKeyPrefix) {
			err := deletePatchesFromConfigMap(c.clientset, cm.Data[key])
			if err != nil {
				zlog.Errorf("Failed to delete ConfigMap: %v", err)
				return err
			}
			delete(cm.Data, key)
		}
	}
	_, err := c.clientset.CoreV1().ConfigMaps(cm.Namespace).Update(context.Background(), cm, metav1.UpdateOptions{})
	if err != nil {
		zlog.Errorf("Failed to update ConfigMap: %v", err)
		return err
	}
	return nil
}

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// This function is similar to http.Get, however http.Get is not easy to mock and test
// Here we use this function and use client as a parameter, so we can make a mock client
// and control the behavior easily.
func getRemoteContent(client httpClient, remoteDeployPath, filePath string) ([]byte, error) {
	url := remoteDeployPath + filePath
	zlog.Infof("remote file url is %s", url)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return []byte{}, fmt.Errorf("failed to create HTTP request for URL %s: %w", url, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return []byte{}, fmt.Errorf("failed to fetch remote content from %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return []byte{}, fmt.Errorf("HTTP request failed: status %d (%s) for URL %s",
			resp.StatusCode, http.StatusText(resp.StatusCode), url)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return []byte{}, fmt.Errorf("failed to read response body from %s: %w", url, err)
	}
	return body, nil
}

func getRemotePatchPath() string {
	if env := os.Getenv("REMOTE_PATCH_PATH"); env != "" {
		return env
	}
	return constant.DefaultRemotePatchPath
}

func getRemoteDeployPath() string {
	if env := os.Getenv("REMOTE_DEPLOY_PATH"); env != "" {
		return env
	}
	return constant.DefaultRemoteDeployPath
}

func getPatchIndexFromRemoteRepo(client httpClient, remoteDeployPath string) (RemotePatchIndexResponse, error) {
	body, err := getRemoteContent(client, remoteDeployPath, "index.yaml")
	if err != nil {
		return []map[string]string{}, err
	}

	var result RemotePatchIndexResponse
	buffer := bytes.NewBuffer(body)
	decoder := yamlutil.NewYAMLOrJSONDecoder(buffer, decoderBufferSize)
	if err := decoder.Decode(&result); err != nil && err != io.EOF {
		return []map[string]string{}, err
	}

	// check format here. Ensure openFuyaoVersion and filePath exists
	for i, item := range result {
		if _, exists := item["openFuyaoVersion"]; !exists {
			return []map[string]string{}, fmt.Errorf("item at index %d is missing required field 'openFuyaoVersion'", i)
		}
		if _, exists := item["filePath"]; !exists {
			return []map[string]string{}, fmt.Errorf("item at index %d is missing required field 'filePath'", i)
		}
		item["filePath"], _ = strings.CutPrefix(item["filePath"], "./")
	}

	return result, nil
}

// addPatchInfoToConfigMap updates the ConfigMap with patch information
func (c *installerClient) addPatchInfoToConfigMap(configMap *corev1.ConfigMap, patchKey, patchValue string) error {
	// update patchFile field in configMap.Data using flat structure
	patch := map[string]any{
		"data": map[string]string{
			patchKey: patchValue,
		},
	}
	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return err
	}

	// update ConfigMap
	tempMap, err := c.clientset.CoreV1().ConfigMaps(configMap.Namespace).
		Patch(context.Background(), configMap.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})

	if err != nil {
		zlog.Errorf("Failed to patch ConfigMap: %v", err)
		return fmt.Errorf("failed to patch ConfigMap: %v", err)
	}
	configMap = tempMap.DeepCopy()
	zlog.Infof("configMap updated %v", configMap.Data)

	return nil
}

func ensureNsExists(clientset kubernetes.Interface, namespace string) error {
	_, err := clientset.CoreV1().Namespaces().Get(context.TODO(), namespace, metav1.GetOptions{})
	if err == nil {
		return nil
	}

	if !apierrors.IsNotFound(err) {
		return fmt.Errorf("check namespace %s failed: %v", namespace, err)
	}

	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
		},
	}

	_, err = clientset.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create namespace %s failed: %v", namespace, err)
	}

	return nil
}

func addPatchesToConfigMap(clientset kubernetes.Interface, version, content string) (string, string, error) {
	key := fmt.Sprintf("%s%s", constant.PatchKeyPrefix, version)
	value := fmt.Sprintf("%s%s", constant.PatchValuePrefix, version)
	if err := ensureNsExists(clientset, constant.PatchNameSpace); err != nil {
		return "", "", fmt.Errorf("failed to ensure ns %s exists: %w", constant.PatchNameSpace, err)
	}
	// k8s configmap
	cm, err := clientset.CoreV1().ConfigMaps(constant.PatchNameSpace).Get(context.TODO(), value, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return "", "", err
	}
	// if exists, should delete old config map first
	if err == nil {
		if delErr := deletePatchesFromConfigMap(clientset, value); delErr != nil {
			return "", "", delErr
		}
	} else {
		if !apierrors.IsNotFound(err) {
			return "", "", err
		}
	}

	cm = &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      value,
			Namespace: constant.PatchNameSpace,
		},
		Data: map[string]string{
			version: content,
		},
	}
	_, err = clientset.CoreV1().ConfigMaps(constant.PatchNameSpace).Create(context.TODO(), cm, metav1.CreateOptions{})
	if err != nil {
		return "", "", err
	}

	return key, value, nil
}

func deletePatchesFromConfigMap(clientset kubernetes.Interface, name string) error {
	zlog.Infof("old configMap %s exists, need delete first", name)
	if err := clientset.CoreV1().ConfigMaps(constant.PatchNameSpace).Delete(context.TODO(), name, metav1.DeleteOptions{}); err != nil {
		return fmt.Errorf("delete configMap failed: %w", err)
	}
	zlog.Infof("delete config map success")
	return nil
}

// validatePatchVersionInfo 验证补丁版本信息的完整性
func validatePatchVersionInfo(info *PatchVersionInfo) error {
	if strings.TrimSpace(info.OpenFuyaoVersion) == "" {
		return errors.New("openFuyaoVersion is required but empty")
	}
	if strings.TrimSpace(info.KubernetesVersion) == "" {
		return errors.New("kubernetesVersion is required but empty")
	}
	if strings.TrimSpace(info.ContainerdVersion) == "" {
		return errors.New("containerdVersion is required but empty")
	}
	if strings.TrimSpace(info.EtcdVersion) == "" {
		return errors.New("etcdVersion is required but empty")
	}
	return nil
}

func patchVersion(c *installerClient, obj *unstructured.Unstructured) error {
	if obj.GetKind() != "BKECluster" { // 修改：非集群CR不需要补丁版本
		return nil
	}
	patchConfigMapName, err := getPatchInfo(c, obj)
	if err != nil {
		return err
	}

	clusterInfo, _, err := unstructured.NestedMap(obj.Object, "spec", "clusterConfig", "cluster")
	if err != nil {
		return errors.Wrap(err, "failed to get cluster info from unstructured object")
	}

	content, err := getVersionFromPatchCM(c.clientset, patchConfigMapName)
	if err != nil {
		return err
	}
	zlog.Infoln("content is", content)

	// apply patch
	clusterInfo["openFuyaoVersion"] = content.OpenFuyaoVersion
	clusterInfo["kubernetesVersion"] = content.KubernetesVersion
	clusterInfo["containerdVersion"] = content.ContainerdVersion
	clusterInfo["etcdVersion"] = content.EtcdVersion
	if err := unstructured.SetNestedMap(obj.Object, clusterInfo, "spec", "clusterConfig", "cluster"); err != nil {
		return errors.Wrap(err, "failed to update cluster configuration with patch version info")
	}

	return nil
}

func getOpenFuyaoVersionFromUnstructured(obj *unstructured.Unstructured) (string, error) {
	openFuyaoVersion, found, err := unstructured.NestedString(obj.Object,
		"spec", "clusterConfig", "cluster", "openFuyaoVersion")
	if err != nil {
		return "", errors.Wrap(err, "failed to extract openFuyaoVersion from cluster config")
	}
	if !found {
		return "", errors.New("openFuyaoVersion field is missing from spec.clusterConfig.cluster, " +
			"please ensure the field exists in your YAML configuration")
	}
	if strings.TrimSpace(openFuyaoVersion) == "" {
		return "", errors.New("openFuyaoVersion field is empty, please provide a valid version")
	}
	return openFuyaoVersion, nil
}

func getVersionFromPatchCM(clientset kubernetes.Interface, patchConfigMapName string) (PatchVersionInfo, error) {
	var content PatchVersionInfo
	// get version info from k8s config map
	ns := constant.PatchNameSpace
	cm, err := clientset.CoreV1().ConfigMaps(ns).Get(context.TODO(), patchConfigMapName, metav1.GetOptions{})
	if err != nil {
		return content, errors.Wrapf(err, "failed to get config map '%s'", patchConfigMapName)
	}
	version := strings.TrimPrefix(patchConfigMapName, constant.PatchValuePrefix)
	if patchInfo, exist := cm.Data[version]; exist {
		err = yaml.Unmarshal([]byte(patchInfo), &content)
		if err != nil {
			return content, errors.Wrapf(err, "failed to unmarshal yaml: %v", err)
		}
	}

	if err := validatePatchVersionInfo(&content); err != nil {
		return content, errors.Wrapf(err, "invalid patch version info in cm '%s'", patchConfigMapName)
	}
	return content, nil
}
