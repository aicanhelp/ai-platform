package plugin

import (
	"context"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"gopkg.openfuyao.cn/bkecommon"
	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	bkevalidte "gopkg.openfuyao.cn/bkecommon/cluster/validation"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/clientutil"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

var (
	kubeconfig    = fmt.Sprintf("%s/%s", utils.Workspace, "config")
	containerdGVK = schema.GroupVersionKind{
		Group:   "bke.bocloud.com",
		Version: "v1beta1",
		Kind:    "ContainerdConfig",
	}
)

type Plugin interface {
	Name() string
	Param() map[string]PluginParam
	Execute(commands []string) ([]string, error)
}

type PluginParam struct {
	Key         string `json:"key"`
	Value       string `json:"value"`
	Required    bool   `json:"required"`
	Default     string `json:"default"`
	Description string `json:"description"`
}

// ParseCommands get plugin param map from commands
func ParseCommands(plugin Plugin, commands []string) (map[string]string, error) {
	externalParam := map[string]string{}
	for _, c := range commands[1:] {
		arg := strings.SplitN(c, "=", 2)
		if len(arg) != 2 {
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

func GetBKECluster(bkeConfigNS string) (*bkev1beta1.BKECluster, error) {
	c, err := clientutil.NewKubernetesClient(kubeconfig)
	if err != nil {
		return nil, err
	}
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
		return nil, errors.Errorf("Get BKECluster %s error: %v", bkeConfigNS, err)
	}

	bc := &bkev1beta1.BKECluster{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(bkeCluster.Object, bc); err != nil {
		return nil, err
	}
	bkeNodes := bkenode.Nodes(bc.Spec.ClusterConfig.Nodes)

	// get nodes info from configMap
	cm, err := c.ClientSet.CoreV1().ConfigMaps(bkeCluster.GetNamespace()).Get(context.Background(), bkeCluster.GetName(), v1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, errors.Errorf("Get BKECluster configMap %s/%s error: %v", bkeCluster.GetNamespace(), bkeCluster.GetName(), err)
		}
	}
	if v, ok := cm.Data["nodes"]; ok {
		nodes := []bkev1beta1.Node{}
		err := json.Unmarshal([]byte(v), &nodes)
		if err != nil {
			return nil, errors.Errorf("Unmarshal BKECluster configMap %s/%s nodes error: %v", bkeCluster.GetNamespace(), bkeCluster.GetName(), err)
		}
		for _, node := range nodes {
			if bkevalidte.ValidateSingleNode(bkenode.Node(node)) != nil {
				return nil, errors.Errorf("BKECluster configMap %s/%s nodes is invalid, err: %v", bkeCluster.GetNamespace(), bkeCluster.GetName(), err)
			}
			bkeNodes = append(bkeNodes, node)
		}
		bc.Spec.ClusterConfig.Nodes = bkeNodes
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
		if v, ok := annotations[bkecommon.BKEClusterFromAnnotationKey]; !ok || v == bkecommon.BKEClusterFromAnnotationValueBKE {
			if err := bkevalidte.ValidateBKEConfig(*bkeConfig); err != nil {
				return nil, errors.Errorf("BKECluster spec.clusterConfig is invalid: %v", err)
			}
		}
		if v, ok := annotations[bkecommon.BKEClusterFromAnnotationKey]; ok && v == bkecommon.BKEClusterFromAnnotationValueBocloud {
			if err := bkevalidte.ValidateNonStandardBKEConfig(*bkeConfig); err != nil {
				return nil, errors.Errorf("BKECluster spec.clusterConfig is invalid: %v", err)
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
	gvr := schema.GroupVersionResource{
		Group:    containerdGVK.Group,
		Version:  containerdGVK.Version,
		Resource: strings.ToLower(containerdGVK.Kind) + "s",
	}

	namespace, name, err := utils.SplitNameSpaceName(containerdCconfigNS)
	if err != nil {
		return nil, err
	}
	containerdConfig, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(context.Background(), name, v1.GetOptions{})
	if err != nil {
		return nil, errors.Errorf("Get ContainerdConfig %s error: %v", containerdConfig, err)
	}

	// 先提取 spec 字段
	spec, found, err := unstructured.NestedMap(containerdConfig.Object, "spec")
	if err != nil {
		return nil, fmt.Errorf("get spec field failed: %v", err)
	}
	if !found {
		return nil, fmt.Errorf("spec field not found in containerd config")
	}

	// 将 spec 部分转换为目标结构体
	cc := &bkev1beta1.ContainerdConfigSpec{}
	if err = runtime.DefaultUnstructuredConverter.FromUnstructured(spec, cc); err != nil {
		return nil, fmt.Errorf("convert spec to ContainerdConfigSpec failed: %v", err)
	}

	return cc, nil
}
