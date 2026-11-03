/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"context"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkevalidte "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/validation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/clientutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var (
	kubeconfig = fmt.Sprintf("%s/%s", utils.Workspace, "config")
)

const two = 2

type Plugin interface {
	Name() string
	Param() map[string]PluginParam
	// Execute runs the plugin. ctx is cancelled on Command delete / ActiveDeadline / SafeClose.
	// Non-kubeadm plugins may ignore ctx in P3; substantive consumption lands in P4.
	Execute(ctx context.Context, commands []string) ([]string, error)
}

type PluginParam struct {
	Key         string `json:"key"`
	Value       string `json:"value"`
	Required    bool   `json:"required"`
	Default     string `json:"default"`
	Description string `json:"description"`
}

type ClusterData struct {
	Cluster *bkev1beta1.BKECluster
	Nodes   bkenode.Nodes
}

// ParseCommands get plugin param map from commands
func ParseCommands(plugin Plugin, commands []string) (map[string]string, error) {
	externalParam := map[string]string{}
	args := commands
	if len(args) > 0 {
		args = args[1:]
	}
	for _, c := range args {
		arg := strings.SplitN(c, "=", two)
		if len(arg) != two {
			continue
		}
		externalParam[arg[0]] = arg[1]
	}
	// The parameter checking
	pluginParam := map[string]string{}
	for key, v := range plugin.Param() {
		log.Debugf("%q plugin param %q, required: %v, default: %s, description: %s", plugin.Name(), key, v.Required, v.Default, v.Description)
		if v, ok := externalParam[key]; ok {
			pluginParam[key] = v
			continue
		}
		if v.Required {
			return pluginParam, errors.Errorf("Missing required parameters %s", key)
		}
		pluginParam[key] = v.Default
	}
	LogDebugInfo(pluginParam, plugin.Name())
	return pluginParam, nil
}

func LogDebugInfo(parseCommand map[string]string, name string) {
	log.Debugf("%q plugin", name)
	for k, v := range parseCommand {
		log.Debugf("%s=%s", k, v)
	}
}

//go:noinline
func GetNodesDataFromNs(namespace string, name string) (bkenode.Nodes, error) {
	if namespace == "" || name == "" {
		return nil, errors.New("invalid namespace or name")
	}
	return GetNodesData(fmt.Sprintf("%s:%s", namespace, name))
}

func GetNodesData(bkeConfigNS string) (bkenode.Nodes, error) {
	clusterData, err := GetClusterData(bkeConfigNS)
	if err != nil {
		return nil, err
	}
	return clusterData.Nodes, nil
}

func GetClusterData(bkeConfigNS string) (*ClusterData, error) {
	c, err := clientutil.NewKubernetesClient(kubeconfig)
	if err != nil {
		return nil, err
	}

	bc, err := GetBKEClusterFromClient(c, bkeConfigNS)
	if err != nil {
		return nil, err
	}

	bkeNodeGVR := schema.GroupVersionResource{
		Group:    bkev1beta1.BKENodeGVK.Group,
		Version:  bkev1beta1.BKENodeGVK.Version,
		Resource: "bkenodes",
	}
	labelSelector := fmt.Sprintf("cluster.x-k8s.io/cluster-name=%s", bc.GetName())

	unstructuredList, err := c.DynamicClient.Resource(bkeNodeGVR).Namespace(bc.GetNamespace()).List(context.Background(), v1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to list BKENodes for cluster %s", bc.GetName())
	}
	bkeNodeList := &bkev1beta1.BKENodeList{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredList.UnstructuredContent(), bkeNodeList); err != nil {
		return nil, errors.Wrapf(err, "failed to convert unstructured BKENodeList")
	}

	nodes := bkenode.ConvertBKENodeListToNodes(bkeNodeList)
	nodes = bkenode.SetDefaultsForNodes(nodes)

	return &ClusterData{
		Cluster: bc,
		Nodes:   nodes,
	}, nil

}

func GetBKECluster(bkeConfigNS string) (*bkev1beta1.BKECluster, error) {
	c, err := clientutil.NewKubernetesClient(kubeconfig)
	if err != nil {
		return nil, err
	}

	return GetBKEClusterFromClient(c, bkeConfigNS)
}

func GetBKEClusterFromClient(c *clientutil.Client, bkeConfigNS string) (*bkev1beta1.BKECluster, error) {
	// build gvr
	gvr := schema.GroupVersionResource{
		Group:    bkev1beta1.GVK.Group,
		Version:  bkev1beta1.GVK.Version,
		Resource: strings.ToLower(bkev1beta1.GVK.Kind) + "s",
	}

	namespace, name, err := utils.SplitNameSpaceName(bkeConfigNS)
	if err != nil {
		return nil, err
	}
	bkeCluster, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(context.Background(), name, v1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("Get BKECluster %s error: %w", bkeConfigNS, err)
	}

	bc := &bkev1beta1.BKECluster{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(bkeCluster.Object, bc); err != nil {
		return nil, err
	}

	return bc, nil
}

func GetBkeConfig(bkeConfigNS string) (*bkev1beta1.BKEConfig, error) {
	bkeCluster, err := GetBKECluster(bkeConfigNS)
	if err != nil {
		return nil, err
	}
	return GetBkeConfigFromBkeCluster(bkeCluster)
}

func GetBkeConfigFromBkeCluster(bkeCluster *bkev1beta1.BKECluster) (*bkev1beta1.BKEConfig, error) {
	bkeConfig := bkeCluster.Spec.ClusterConfig
	// validate cluster which type is bke
	if annotations := bkeCluster.GetAnnotations(); annotations != nil {
		if v, ok := annotations[common.BKEClusterFromAnnotationKey]; !ok || v == common.BKEClusterFromAnnotationValueBKE {
			if err := bkevalidte.ValidateBKEConfig(*bkeConfig); err != nil {
				return nil, fmt.Errorf("BKECluster spec.clusterConfig is invalid: %w", err)
			}
		}
		if v, ok := annotations[common.BKEClusterFromAnnotationKey]; ok && v == common.BKEClusterFromAnnotationValueBocloud {
			if err := bkevalidte.ValidateNonStandardBKEConfig(*bkeConfig); err != nil {
				return nil, fmt.Errorf("BKECluster spec.clusterConfig is invalid: %w", err)
			}
		}
	}

	return bkeConfig, nil
}

func GetContainerdConfig(containerdCconfigNS string) (*bkev1beta1.ContainerdConfigSpec, error) {
	c, err := clientutil.NewKubernetesClient(kubeconfig)
	if err != nil {
		return nil, err
	}
	// build gvr
	gvr := bkev1beta1.ContainerdConfigGVR()

	namespace, name, err := utils.SplitNameSpaceName(containerdCconfigNS)
	if err != nil {
		return nil, err
	}
	containerdConfig, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(context.Background(), name, v1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("Get ContainerdConfig %s error: %w", containerdConfig, err)
	}

	// 先提取 spec 字段
	spec, found, err := unstructured.NestedMap(containerdConfig.Object, "spec")
	if err != nil {
		return nil, fmt.Errorf("get spec field failed: %w", err)
	}
	if !found {
		return nil, fmt.Errorf("spec field not found in containerd config")
	}

	// 将 spec 部分转换为目标结构体
	cc := &bkev1beta1.ContainerdConfigSpec{}
	if err = runtime.DefaultUnstructuredConverter.FromUnstructured(spec, cc); err != nil {
		return nil, fmt.Errorf("convert spec to ContainerdConfigSpec failed: %w", err)
	}

	return cc, nil
}
