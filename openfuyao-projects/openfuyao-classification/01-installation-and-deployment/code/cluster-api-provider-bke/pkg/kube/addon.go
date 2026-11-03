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
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apierrors2 "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	metricrecord "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics/record"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
	v "gopkg.openfuyao.cn/cluster-api-provider-bke/version"
)

const (
	keyValueSeparatorCount     = 2  // for parseNodeSelector key=value splitting
	bocOperatorTimeoutMinutes  = 30 // timeout for bocoperator addon in minutes
	bocOperatorIntervalSeconds = 30 // interval for bocoperator addon in seconds
)

type Task struct {
	Name     string
	FilePath string
	// ManifestContent holds raw YAML when applying from an in-memory release bundle.
	ManifestContent []byte
	Param           map[string]interface{}
	// IgnoreError: if true, continue to apply the remaining yaml ignoring errors
	IgnoreError bool
	// Block: if true, wait for the task to complete
	Block    bool
	Timeout  time.Duration
	Interval time.Duration
	Operate  bkeaddon.AddonOperate
	// ApplyStrategy optionally refines UpgradeAddon behavior (e.g. CreateOnly).
	ApplyStrategy string
	recorder      *AddonRecorder
}

// ApplyStrategy values accepted by Task when Operate is UpgradeAddon.
const (
	ApplyStrategyCreateOnly = "CreateOnly"
)

type AddonRecorder struct {
	AddonName    string
	AddonVersion string
	AddonObjects []*AddonObject
}

type AddonObject struct {
	Name      string
	Kind      string
	NameSpace string
}

func NewAddonRecorder(addonT *bkeaddon.AddonTransfer) *AddonRecorder {
	return &AddonRecorder{
		AddonName:    addonT.Addon.Name,
		AddonVersion: addonT.Addon.Version,
		AddonObjects: []*AddonObject{},
	}
}

func (recorder *AddonRecorder) Record(obj *unstructured.Unstructured) {
	recorder.AddonObjects = append(recorder.AddonObjects, NewAddonObject(obj))
}

func NewAddonObject(obj *unstructured.Unstructured) *AddonObject {
	return &AddonObject{
		Name:      obj.GetName(),
		Kind:      obj.GetKind(),
		NameSpace: obj.GetNamespace(),
	}
}

func NewTask(name, filePath string, param map[string]interface{}) *Task {
	return &Task{
		Name:        name,
		Param:       param,
		FilePath:    filePath,
		IgnoreError: false,
		Block:       true,
	}
}

func (t *Task) SetWaiter(wait bool, timeout time.Duration, interval time.Duration) *Task {
	t.Block = wait
	t.Timeout = timeout
	t.Interval = interval
	return t
}

func (t *Task) AddRepo(repo string) *Task {
	if t.Param == nil {
		t.Param = make(map[string]interface{})
	}
	t.Param["repo"] = repo
	return t
}

func (t *Task) SetOperate(operate bkeaddon.AddonOperate) *Task {
	t.Operate = operate
	return t
}

func (t *Task) SetApplyStrategy(strategy string) *Task {
	t.ApplyStrategy = strategy
	return t
}

func (t *Task) RegisAddonRecorder(recorder *AddonRecorder) *Task {
	t.recorder = recorder
	return t
}

func (c *Client) InstallAddon(bkeCluster *bkev1beta1.BKECluster, addonT *bkeaddon.AddonTransfer, addonRecorder *AddonRecorder, localClient client.Client, bkeNodes bkenode.Nodes) error {
	var err error
	defer metricrecord.AddonInstallRecord(bkeCluster, addonT.Addon.Name, addonT.Addon.Version, err)()

	addon := addonT.Addon.DeepCopy()
	bkeConfig := bkeCluster.Spec.ClusterConfig
	cfg := bkeinit.BkeConfig(*bkeConfig)

	if addon.Type == bkeaddon.ChartAddon {
		return c.installChartAddon(addon, addonT.Operate, bkeCluster.Namespace, cfg, localClient)
	}
	return c.installYamlAddon(yamlAddonInstall{
		addon: addon, transfer: addonT, cluster: bkeCluster,
		config: cfg, recorder: addonRecorder, nodes: bkeNodes,
	})
}

type yamlAddonInstall struct {
	addon    *confv1beta1.Product
	transfer *bkeaddon.AddonTransfer
	cluster  *bkev1beta1.BKECluster
	config   bkeinit.BkeConfig
	recorder *AddonRecorder
	nodes    bkenode.Nodes
}

func (c *Client) installYamlAddon(input yamlAddonInstall) error {
	addon, addonT, bkeCluster := input.addon, input.transfer, input.cluster
	cfg, addonRecorder, bkeNodes := input.config, input.recorder, input.nodes
	files, err := c.getAddonYamlFiles(addon)
	if err != nil {
		return err
	}

	// For cluster-api addon, 003-manage.yaml (Namespace + BKENode) and
	// 004-manage-bkecluster.yaml (Secret + BKECluster) must be applied after postprocess.
	// They are deferred to the EnsureClusterAPIManagerManifest phase, which applies
	// 003 first (BKENodes before BKECluster) so that the master node creation webhook
	// sees no BKECluster yet and does not block the initial deployment.
	if addon != nil && addon.Name == "cluster-api" && addonT != nil && addonT.Operate == bkeaddon.CreateAddon {
		files = filterOutClusterAPIManageYaml(files)
		if len(files) == 0 {
			return errors.Errorf("no yaml file found after filtering 003-manage.yaml and 004-manage-bkecluster.yaml in addon %q dir", addon.Name)
		}
	}

	if addonT.Operate == bkeaddon.RemoveAddon {
		sort.Sort(sort.Reverse(sort.StringSlice(files)))
	} else {
		sort.Strings(files)
	}

	repo := cfg.ImageRepo()
	param, err := c.prepareAddonParameters(addonParameterConfig{
		bkeCluster: bkeCluster,
		addon:      addon,
		files:      files,
		repo:       repo,
		bkeNodes:   bkeNodes,
		operate:    addonT.Operate,
	})
	if err != nil {
		return err
	}

	// For cluster-api addon, ensure webhook TLS secrets and AES key exist on
	// the target cluster, then inject caBundle template params.
	if addon != nil && addon.Name == "cluster-api" {
		log.Infof("cluster-api addon detected, operate=%v, ensuring webhook secrets", addonT.Operate)
		var capiCaBundle, bkeCaBundle string
		if addonT.Operate != bkeaddon.RemoveAddon {
			capiCaBundle, bkeCaBundle, err = c.ensureClusterAPISecrets()
			if err != nil {
				log.Errorf("failed to ensure cluster-api webhook secrets: %v", err)
				return errors.Wrapf(err, "failed to ensure cluster-api webhook secrets")
			}
			log.Info("cluster-api webhook secrets ensured, injecting caBundle into template params")
		}
		for _, p := range param {
			if p == nil {
				continue
			}
			p["capiCaBundle"] = capiCaBundle
			p["bkeCaBundle"] = bkeCaBundle
		}
	}

	applyConfig := &addonApplyConfig{
		addon:         addon,
		addonT:        addonT,
		files:         files,
		param:         param,
		repo:          repo,
		addonRecorder: addonRecorder,
	}

	return c.applyAddonFiles(applyConfig)
}

// filterOutClusterAPIManageYaml filters out 003-manage.yaml and 004-manage-bkecluster.yaml
// from the file list. These two files are applied separately in the
// EnsureClusterAPIManagerManifest phase: 003 first (Namespace + BKENode) then
// 004 (Secret + BKECluster), so that BKENodes are created before BKECluster
// to avoid the master node creation webhook from blocking the initial deployment.
func filterOutClusterAPIManageYaml(files []string) []string {
	if len(files) == 0 {
		return files
	}
	out := make([]string, 0, len(files))
	for _, f := range files {
		base := filepath.Base(f)
		if strings.EqualFold(base, "003-manage.yaml") ||
			strings.EqualFold(base, "004-manage-bkecluster.yaml") {
			continue
		}
		out = append(out, f)
	}
	return out
}

func (c *Client) getAddonYamlFiles(addon *confv1beta1.Product) ([]string, error) {
	addonDir := filepath.Join(constant.K8sManifestsDir, addon.Name, addon.Version)
	if _, err := os.Stat(addonDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("addon dir %q not exist,err: %w", addonDir, err)
	}

	var files []string
	err := filepath.Walk(addonDir, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".yaml") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to traverse addon dir %q", addonDir)
	}
	if len(files) == 0 {
		return nil, errors.Errorf("no yaml file found in dir %q", addonDir)
	}
	return files, nil
}

type addonParameterConfig struct {
	bkeCluster *bkev1beta1.BKECluster
	addon      *confv1beta1.Product
	files      []string
	repo       string
	bkeNodes   bkenode.Nodes
	operate    bkeaddon.AddonOperate
}

// prepareAddonParameters 准备 Addon 渲染参数，并在 Calico 场景下追加 Provider 计算出的 Typha 参数。
func (c *Client) prepareAddonParameters(config addonParameterConfig) (map[string]map[string]interface{}, error) {
	bkeCluster := config.bkeCluster
	addon := config.addon
	files := config.files
	repo := config.repo
	bkeNodes := config.bkeNodes
	operate := config.operate
	commonParam, err := getCommonParamFromBKECluster(bkeCluster, bkeNodes)
	if err != nil {
		return nil, err
	}

	commonParam, err = c.enhanceCommonParamForSpecialAddons(bkeCluster, addon, commonParam, bkeNodes)
	if err != nil {
		return nil, err
	}

	filesBaseNames := make([]string, len(files))
	for i, file := range files {
		filesBaseNames[i] = strings.Split(filepath.Base(file), ".")[0]
	}

	param := prepareAddonParam(addon.Param, filesBaseNames, repo)
	if err := normalizeNodeSelector(param); err != nil {
		return nil, err
	}

	for i := range files {
		fileName := filesBaseNames[i]
		param[fileName] = mergeParam(commonParam, param[fileName])
	}
	if addon != nil && strings.EqualFold(addon.Name, calicoAddonName) {
		c.logInfof("prepare calico addon parameters, version=%s, files=%d, nodes=%d, operate=%s", addon.Version, len(files), bkeNodes.Length(), operate)
		if err := applyCalicoScaleParams(addon, files, param, bkeNodes, operate); err != nil {
			return nil, err
		}
	}
	return param, nil
}

const (
	calicoAddonName         = "calico"
	calicoTyphaName         = "calico-typha"
	allowTyphaParam         = "allowTypha"
	typhaReplicasParam      = "typhaReplicas"
	minTyphaReplicas        = 3
	maxTyphaReplicas        = 20
	typhaNodesPerReplica    = 200
	typhaEnableNodeBoundary = 50
)

// applyCalicoScaleParams 根据节点规模和目标 manifest 能力计算 Typha 开关及副本数，并覆盖用户旧参数。
func applyCalicoScaleParams(addon *confv1beta1.Product, files []string, param map[string]map[string]interface{}, bkeNodes bkenode.Nodes, operate bkeaddon.AddonOperate) error {
	if addon == nil || !strings.EqualFold(addon.Name, calicoAddonName) {
		return nil
	}

	nodeCount := bkeNodes.Length()
	allowTypha := "false"
	typhaReplicas := "0"
	if nodeCount > typhaEnableNodeBoundary {
		supported, err := calicoManifestSupportsTypha(files)
		if err != nil {
			return err
		}
		if !supported {
			if operate == bkeaddon.CreateAddon {
				return errors.Errorf("calico addon %s does not support typha, but node count %d is greater than %d", addon.Version, nodeCount, typhaEnableNodeBoundary)
			}
		} else {
			allowTypha = "true"
			typhaReplicas = strconv.Itoa(calculateTyphaReplicas(nodeCount))
		}
	}

	for _, values := range param {
		values[allowTyphaParam] = allowTypha
		values[typhaReplicasParam] = typhaReplicas
	}
	return nil
}

// calculateTyphaReplicas 按每 200 节点 1 副本、最少 3 副本、最多 20 副本计算 Typha 副本数。
func calculateTyphaReplicas(nodeCount int) int {
	replicas := ceilDiv(nodeCount, typhaNodesPerReplica)
	if replicas < minTyphaReplicas {
		replicas = minTyphaReplicas
	}
	if replicas > maxTyphaReplicas {
		replicas = maxTyphaReplicas
	}
	if nodeCount > 0 && replicas >= nodeCount {
		return nodeCount - 1
	}
	return replicas
}

// ceilDiv 计算向上取整的整数除法结果。
func ceilDiv(a, b int) int {
	if b == 0 {
		return 0
	}
	return (a + b - 1) / b
}

// calicoManifestSupportsTypha 通过模板内容锚点判断目标 Calico manifest 是否具备 Typha 渲染能力。
func calicoManifestSupportsTypha(files []string) (bool, error) {
	var combined strings.Builder
	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			return false, errors.Wrapf(err, "read calico manifest %q failed", file)
		}
		combined.Write(content)
		combined.WriteByte('\n')
	}
	manifest := combined.String()
	hasTyphaSwitch := strings.Contains(manifest, allowTyphaParam)
	hasReplicaParam := strings.Contains(manifest, typhaReplicasParam)
	hasTyphaDeployment := strings.Contains(manifest, "kind: Deployment") && strings.Contains(manifest, "name: calico-typha")
	hasTyphaService := strings.Contains(manifest, "kind: Service") && strings.Contains(manifest, "name: calico-typha")
	hasTyphaPDB := strings.Contains(manifest, "kind: PodDisruptionBudget") && strings.Contains(manifest, "name: calico-typha")
	hasNodeTyphaConfig := strings.Contains(manifest, "typha_service_name") && strings.Contains(manifest, "FELIX_TYPHAK8SSERVICENAME")
	return hasTyphaSwitch && hasReplicaParam && hasTyphaDeployment && hasTyphaService && hasTyphaPDB && hasNodeTyphaConfig, nil
}
func (c *Client) enhanceCommonParamForSpecialAddons(bkeCluster *bkev1beta1.BKECluster, addon *confv1beta1.Product, commonParam map[string]interface{}, bkeNodes bkenode.Nodes) (map[string]interface{}, error) {
	if commonParam == nil {
		commonParam = make(map[string]interface{})
	}

	if addon.Name == "bocoperator" {
		bocParam, err := getBocDefaultParam(bkeCluster, addon, bkeNodes)
		if err != nil {
			return nil, err
		}

		portalClusterToken, err := c.getPortalK8sToken()
		if err != nil {
			return nil, err
		}
		bocParam["portalClusterToken"] = portalClusterToken
		commonParam = mergeParam(commonParam, bocParam)
	}

	if addon.Name == "cluster-api" && addon.Param["manage"] == "true" {
		portalClusterToken, err := c.getPortalK8sToken()
		if err != nil {
			return nil, err
		}
		commonParam["clusterToken"] = portalClusterToken
		clusterName := ""
		if bkeCluster != nil {
			clusterName = bkeCluster.Name
		}
		commonParam["loadBalanceType"] = bkeCluster.Spec.ClusterConfig.Cluster.LoadBalanceType
		commonParam["nodes"] = convertNodesToManageTemplateData(bkeNodes, clusterName)
	}

	if addon.Name == "fabric" {
		fabricParam, err := parseFabricParam(addon)
		if err != nil {
			return nil, err
		}
		addon.Param = fabricParam
	}
	// nodelocaldns 组件需要额外的参数
	if addon.Name == "nodelocaldns" {
		commonParam["domain"] = bkeCluster.Spec.ClusterConfig.Cluster.Networking.DNSDomain
		proxymode := bkeCluster.Spec.ClusterConfig.CustomExtra["proxyMode"]
		if proxymode == "iptables" {
			commonParam["DNSserver"] = commonParam["dnsIP"]
			commonParam["clusterDNS"] = "__PILLAR__CLUSTER__DNS__"
		} else {
			if proxymode == "ipvs" {
				commonParam["DNSserver"] = ""
				commonParam["clusterDNS"] = commonParam["dnsIP"]
			}
		}
	}

	return commonParam, nil
}

type addonApplyConfig struct {
	addon         *confv1beta1.Product
	addonT        *bkeaddon.AddonTransfer
	files         []string
	param         map[string]map[string]interface{}
	repo          string
	addonRecorder *AddonRecorder
}

// PrepareRenderParamForAddonFile builds the same render parameters as addon installation,
// but only for a single yaml file.
//
// It returns a param map that can be passed into kube.NewTask(..., param).
// RenderParamsForCluster builds template parameters for release manifest YAML.
func (c *Client) RenderParamsForCluster(
	bkeCluster *bkev1beta1.BKECluster,
	bkeNodes bkenode.Nodes,
) (map[string]interface{}, error) {
	return getCommonParamFromBKECluster(bkeCluster, bkeNodes)
}

func (c *Client) PrepareRenderParamForAddonFile(
	bkeCluster *bkev1beta1.BKECluster,
	addon *confv1beta1.Product,
	filePath string,
	repo string,
	bkeNodes bkenode.Nodes,
) (map[string]interface{}, error) {
	if addon == nil {
		return nil, errors.Errorf("addon is nil")
	}
	commonParam, err := getCommonParamFromBKECluster(bkeCluster, bkeNodes)
	if err != nil {
		return nil, err
	}
	commonParam, err = c.enhanceCommonParamForSpecialAddons(bkeCluster, addon, commonParam, bkeNodes)
	if err != nil {
		return nil, err
	}
	fileBase := strings.Split(filepath.Base(filePath), ".")[0]
	param := prepareAddonParam(addon.Param, []string{fileBase}, repo)
	if err := normalizeNodeSelector(param); err != nil {
		return nil, err
	}
	param[fileBase] = mergeParam(commonParam, param[fileBase])
	return param[fileBase], nil
}

// applyAddonFiles 应用 Addon manifest，Calico 会进入专用路径以处理 Typha 升级和回退顺序。
func (c *Client) applyAddonFiles(config *addonApplyConfig) error {
	if config.addon != nil && strings.EqualFold(config.addon.Name, calicoAddonName) {
		return c.applyCalicoAddonFiles(config)
	}
	return c.applyAddonFilesInOrder(config)
}

// applyCalicoAddonFiles 按目标 Typha 状态执行 Calico manifest，保证启用和关闭 Typha 的顺序安全。
func (c *Client) applyCalicoAddonFiles(config *addonApplyConfig) error {
	if config == nil || config.addon == nil || config.addonT == nil {
		return errors.Errorf("invalid calico addon apply config")
	}
	if config.addonT.Operate == bkeaddon.RemoveAddon {
		c.logInfof("remove calico addon, skip typha upgrade or rollback gate")
		return c.applyAddonFilesInOrder(config)
	}

	allowTypha := getCalicoAllowTypha(config.param)
	if allowTypha {
		includeDependencies := config.addonT.Operate == bkeaddon.CreateAddon
		if includeDependencies {
			c.logInfof("calico create target enables typha, bootstrap calico base dependencies and typha before switching calico-node")
		} else {
			c.logInfof("calico target enables typha, bootstrap typha before switching calico-node")
		}
		if err := c.applyCalicoTyphaBootstrapObjects(config, includeDependencies); err != nil {
			return err
		}
		replicas := getCalicoTyphaReplicas(config.param)
		if err := c.waitCalicoTyphaReady(replicas, config.addon.Block); err != nil {
			return err
		}
		c.logInfof("calico typha is ready, continue applying full calico manifest")
		return c.applyAddonFilesInOrder(config)
	}

	//判断当前集群是否已启用 Typha
	typhaCurrentlyEnabled, err := c.calicoTyphaCurrentlyEnabled()
	if err != nil {
		return err
	}
	if err := c.applyAddonFilesInOrder(config); err != nil {
		return err
	}
	if typhaCurrentlyEnabled {
		c.logInfof("calico target disables typha, wait calico-node ready before deleting typha")
		if err := c.waitCalicoNodeReady(config.addon.Block); err != nil {
			return err
		}
		if err := c.deleteCalicoTyphaResources(); err != nil {
			return err
		}
	}
	return nil
}

// applyAddonFilesInOrder 按文件名顺序应用 Addon manifest，保留普通 Addon 的原有行为。
func (c *Client) applyAddonFilesInOrder(config *addonApplyConfig) error {
	filesBaseNames := c.extractFileBaseNames(config.files)

	for i, file := range config.files {
		fileName := filesBaseNames[i]
		task := c.createAddonTask(config, fileName, file)

		if err := c.ApplyYaml(task); err != nil {
			return c.handleApplyError(err, config.addon, file, config.addonT.Operate)
		}
	}
	return nil
}

// applyCalicoTyphaBootstrapObjects 预先应用 Typha 启动所需对象。
// 首次创建 Calico 且目标状态启用 Typha 时，需要先应用 CRD、RBAC、ServiceAccount、ConfigMap 等基础依赖，
// 再创建 Typha Deployment/Service/PDB，避免 calico-node 还未完整安装时 Typha 因缺少依赖而启动失败。
func (c *Client) applyCalicoTyphaBootstrapObjects(config *addonApplyConfig, includeDependencies bool) error {
	restMapper, err := c.getRestMapper()
	if err != nil {
		return err
	}
	filesBaseNames := c.extractFileBaseNames(config.files)
	var filtered []unstructured.Unstructured
	var bootstrapTask *Task
	bootstrapFile := "calico-typha-bootstrap"
	for i, file := range config.files {
		fileName := filesBaseNames[i]
		task := c.createAddonTask(config, fileName, file)
		if bootstrapTask == nil {
			bootstrapTask = task
		}
		decoder, err := RenderYamlToDecoder(task)
		if err != nil {
			return errors.Wrapf(err, "failed to render calico typha bootstrap yaml %q", file)
		}
		objects, err := GetUnStructListFromDecoder(decoder)
		if err != nil {
			return errors.Wrapf(err, "failed to decode calico typha bootstrap yaml %q", file)
		}
		matched := filterCalicoTyphaBootstrapObjects(c.processUnstructuredList(objects), includeDependencies)
		if len(matched) == 0 {
			continue
		}
		c.logInfof("collect calico typha bootstrap objects, file=%s, objects=%d, includeDependencies=%t", file, len(matched), includeDependencies)
		filtered = append(filtered, matched...)
	}
	if len(filtered) == 0 {
		c.logInfof("no calico typha bootstrap objects found, includeDependencies=%t", includeDependencies)
		return nil
	}
	if bootstrapTask == nil {
		return errors.Errorf("calico typha bootstrap task is nil")
	}
	bootstrapTask.FilePath = bootstrapFile
	bootstrapTask.FilePath = bootstrapFile
	filtered = c.sortUnstructuredList(filtered, bootstrapTask)
	c.logInfof("apply calico typha bootstrap objects, objects=%d, includeDependencies=%t", len(filtered), includeDependencies)
	if err := c.applyUnstructuredList(filtered, restMapper, bootstrapTask); err != nil {
		return c.handleApplyError(err, config.addon, bootstrapFile, config.addonT.Operate)
	}
	return nil
}

// filterCalicoTyphaBootstrapObjects 从渲染后的对象中筛选 Typha 启动所需资源。
// 更新/升级场景只提前应用 Typha 自身对象；首次创建场景会额外带上基础依赖，但仍跳过 calico-node DaemonSet。
func filterCalicoTyphaBootstrapObjects(objects []unstructured.Unstructured, includeDependencies bool) []unstructured.Unstructured {
	var filtered []unstructured.Unstructured
	for _, obj := range objects {
		if isCalicoTyphaObject(obj) || includeDependencies && isCalicoBootstrapDependency(obj) {
			filtered = append(filtered, obj)
		}
	}
	return filtered
}

// isCalicoTyphaObject 判断对象是否是 Typha bootstrap 必须提前创建的自身资源。
func isCalicoTyphaObject(obj unstructured.Unstructured) bool {
	if obj.GetName() != calicoTyphaName || obj.GetNamespace() != metav1.NamespaceSystem {
		return false
	}
	switch obj.GetKind() {
	case "Deployment", "Service", "PodDisruptionBudget":
		return true
	default:
		return false
	}
}

// isCalicoBootstrapDependency 判断对象是否是首次安装 Calico 时 Typha 启动前必须具备的基础依赖。
func isCalicoBootstrapDependency(obj unstructured.Unstructured) bool {
	switch obj.GetKind() {
	case "CustomResourceDefinition", "ServiceAccount", "ClusterRole", "ClusterRoleBinding", "Role", "RoleBinding", "ConfigMap", "Secret":
		return true
	default:
		return false
	}
}

// getCalicoAllowTypha 从文件级渲染参数中读取 Provider 计算出的 Typha 开关。
func getCalicoAllowTypha(param map[string]map[string]interface{}) bool {
	for _, values := range param {
		if fmt.Sprint(values[allowTyphaParam]) == "true" {
			return true
		}
	}
	return false
}

// getCalicoTyphaReplicas 从文件级渲染参数中读取 Provider 计算出的 Typha 副本数。
func getCalicoTyphaReplicas(param map[string]map[string]interface{}) int {
	for _, values := range param {
		replicas, err := strconv.Atoi(fmt.Sprint(values[typhaReplicasParam]))
		if err == nil && replicas > 0 {
			return replicas
		}
	}
	return minTyphaReplicas
}

// calicoTyphaCurrentlyEnabled 检查当前集群是否已经通过 calico-config 或 Typha Deployment 启用了 Typha。
func (c *Client) calicoTyphaCurrentlyEnabled() (bool, error) {
	if c.ClientSet == nil {
		return false, nil
	}
	cm, err := c.ClientSet.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(c.context(), "calico-config", metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return false, err
	}
	if err == nil && cm.Data["typha_service_name"] == calicoTyphaName {
		c.logInfof("current calico-config already points to typha service")
		return true, nil
	}
	_, err = c.ClientSet.AppsV1().Deployments(metav1.NamespaceSystem).Get(c.context(), calicoTyphaName, metav1.GetOptions{})
	if err == nil {
		c.logInfof("current calico typha deployment exists")
		return true, nil
	}
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	return false, err
}

// waitCalicoTyphaReady 等待 Typha Deployment 达到期望副本并确认 Service Endpoints 可用。
func (c *Client) waitCalicoTyphaReady(expectedReplicas int, block bool) error {
	if !block || c.ClientSet == nil {
		return nil
	}
	if expectedReplicas < minTyphaReplicas {
		expectedReplicas = minTyphaReplicas
	}
	c.logInfof("wait calico typha ready, expectedReplicas=%d", expectedReplicas)
	return wait.PollImmediate(bkeinit.DefaultAddonInterval, bkeinit.DefaultAddonTimeout, func() (bool, error) {
		deploy, err := c.ClientSet.AppsV1().Deployments(metav1.NamespaceSystem).Get(c.context(), calicoTyphaName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		if int(deploy.Status.ReadyReplicas) < expectedReplicas || int(deploy.Status.AvailableReplicas) < expectedReplicas {
			return false, nil
		}
		endpoints, err := c.ClientSet.CoreV1().Endpoints(metav1.NamespaceSystem).Get(c.context(), calicoTyphaName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return endpointHasReadyAddress(endpoints), nil
	})
}

// endpointHasReadyAddress 判断 Endpoints 是否至少存在一个可用后端地址。
func endpointHasReadyAddress(endpoints *corev1.Endpoints) bool {
	if endpoints == nil {
		return false
	}
	for _, subset := range endpoints.Subsets {
		if len(subset.Addresses) > 0 {
			return true
		}
	}
	return false
}

// waitCalicoNodeReady 等待 calico-node DaemonSet 完成切换并全部 Ready。
func (c *Client) waitCalicoNodeReady(block bool) error {
	if !block || c.ClientSet == nil {
		return nil
	}
	c.logInfof("wait calico-node daemonset ready")
	return wait.PollImmediate(bkeinit.DefaultAddonInterval, bkeinit.DefaultAddonTimeout, func() (bool, error) {
		ds, err := c.ClientSet.AppsV1().DaemonSets(metav1.NamespaceSystem).Get(c.context(), "calico-node", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		return ds.Status.DesiredNumberScheduled > 0 &&
			ds.Status.NumberReady == ds.Status.DesiredNumberScheduled &&
			ds.Status.UpdatedNumberScheduled == ds.Status.DesiredNumberScheduled, nil
	})
}

// deleteCalicoTyphaResources 在 calico-node 切回直连 API Server 后删除 Typha 相关资源。
func (c *Client) deleteCalicoTyphaResources() error {
	if c.ClientSet == nil {
		return nil
	}
	ctx := c.context()
	c.logInfof("delete calico typha deployment, service and pdb")
	var errs []error
	if err := c.ClientSet.AppsV1().Deployments(metav1.NamespaceSystem).Delete(ctx, calicoTyphaName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		errs = append(errs, err)
	}
	if err := c.ClientSet.CoreV1().Services(metav1.NamespaceSystem).Delete(ctx, calicoTyphaName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		errs = append(errs, err)
	}
	if err := c.ClientSet.PolicyV1().PodDisruptionBudgets(metav1.NamespaceSystem).Delete(ctx, calicoTyphaName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		errs = append(errs, err)
	}
	return errors.Wrap(kerrors.NewAggregate(errs), "delete calico typha resources failed")
}

// logInfof 在 kube Client 存在 logger 时打印 info 日志，避免单元测试中的 nil logger panic。
func (c *Client) logInfof(format string, args ...interface{}) {
	if c != nil && c.Log != nil {
		c.Log.Infof(format, args...)
	}
}

// context 返回 Kubernetes 调用使用的上下文，未设置时使用 Background。
func (c *Client) context() context.Context {
	if c.Ctx != nil {
		return c.Ctx
	}
	return context.Background()
}

// extractFileBaseNames 提取 manifest 文件名作为参数映射 key，例如 001-typha.yaml 对应 001-typha。
func (c *Client) extractFileBaseNames(files []string) []string {
	filesBaseNames := make([]string, len(files))
	for i, file := range files {
		filesBaseNames[i] = strings.Split(filepath.Base(file), ".")[0]
	}
	return filesBaseNames
}

// createAddonTask 基于单个 manifest 文件和渲染参数创建 ApplyYaml 使用的任务对象。
func (c *Client) createAddonTask(config *addonApplyConfig, fileName, file string) *Task {
	task := NewTask(config.addon.Name, file, config.param[fileName]).
		AddRepo(config.repo).
		SetWaiter(config.addon.Block, bkeinit.DefaultAddonTimeout, bkeinit.DefaultAddonInterval).
		SetOperate(config.addonT.Operate).
		RegisAddonRecorder(config.addonRecorder)

	if config.addonT.Operate == bkeaddon.CreateAddon && config.addon.Name == "bocoperator" {
		task.SetWaiter(config.addon.Block, bocOperatorTimeoutMinutes*time.Minute, bocOperatorIntervalSeconds*time.Second)
	}

	return task
}

// handleApplyError 统一处理 manifest 应用错误，卸载场景保持原有忽略失败的兼容行为。
func (c *Client) handleApplyError(err error, addon *confv1beta1.Product, file string, operate bkeaddon.AddonOperate) error {
	if operate == bkeaddon.RemoveAddon {
		c.Log.Warnf("(ignore)failed to remove addon %s/%s, err: %v", addon.Name, addon.Version, err)
		return nil
	}

	if apierrors2.IsNoMatchError(err) {
		c.Log.Errorf("The addon %s/%s yaml file contains resources that are not supported in the target cluster，err: %v", addon.Name, addon.Version, err)
	} else {
		c.Log.Warnf("failed to apply yaml %q, err: %v", file, err)
	}
	return err
}

func (c *Client) getPortalK8sToken() (string, error) {
	return c.NewK8sToken()
}

func getBocDefaultParam(bkeCluster *bkev1beta1.BKECluster, addon *confv1beta1.Product, bkeNodes bkenode.Nodes) (map[string]interface{}, error) {
	param := make(map[string]interface{})

	bkeConfig := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)

	masterNodes := bkeNodes.Master().Decrypt()
	param["masterSshIp"] = masterNodes[0].IP
	param["masterSshPort"] = masterNodes[0].Port
	param["masterSshUser"] = masterNodes[0].Username
	param["masterSshPwd"] = masterNodes[0].Password

	param["pipeLineServerIp"] = masterNodes[0].IP
	param["pipeLineServerPort"] = masterNodes[0].Port
	param["pipeLineServerUser"] = masterNodes[0].Username
	param["pipeLineServerPwd"] = masterNodes[0].Password
	param["pipeLineServerHostname"] = masterNodes[0].Hostname

	if ip, ok := bkeConfig.CustomExtra["pipelineServer"]; ok && ip != "" {
		if !bkenet.ValidIP(ip) {
			return nil, errors.Errorf("invalid pipelineServer IP %q", ip)
		}
		pipeLineNodes := masterNodes.Filter(bkenode.FilterOptions{"IP": ip})
		if len(pipeLineNodes) == 0 || len(pipeLineNodes) != 1 {
			log.Warnf("invalid pipelineServer IP %q, use master 0", ip)
		} else {
			param["pipeLineServerIp"] = pipeLineNodes[0].IP
			param["pipeLineServerPort"] = pipeLineNodes[0].Port
			param["pipeLineServerUser"] = pipeLineNodes[0].Username
			param["pipeLineServerPwd"] = pipeLineNodes[0].Password
			param["pipeLineServerHostname"] = pipeLineNodes[0].Hostname
		}
	}

	param["isLocalDB"] = "true"
	param["bocVersion"] = "v4.0"

	masterServersJson, err := json.Marshal(masterNodes)
	if err != nil {
		return nil, err
	}
	param["masterServers"] = string(masterServersJson)

	param["dbHost"] = bkeCluster.Spec.ControlPlaneEndpoint.Host

	tmp := make(map[string]interface{})
	for k, v := range addon.Param {
		tmp[k] = v
	}
	return mergeParam(param, tmp), nil
}

func getCommonParamFromBKECluster(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkenode.Nodes) (map[string]interface{}, error) {
	param := initializeDefaultParams()

	if bkeCluster == nil {
		return param, nil
	}

	if bkeCluster.Spec.ClusterConfig == nil {
		return param, nil
	}

	bkeConfig := bkeinit.BkeConfig(*bkeCluster.Spec.ClusterConfig)

	// 设置镜像仓库参数
	setImageRepoParams(&param, bkeConfig)

	// 设置Chart仓库参数
	setChartRepoParams(&param, bkeConfig)

	// 设置HTTP仓库参数
	setHTTPRepoParams(&param, bkeConfig)

	// 设置NTP服务器参数
	setNTPServer(&param, bkeConfig)

	// 设置Agent健康监听端口参数
	setAgentHealthPort(&param, bkeConfig)

	// 设置 bkeagent 业务日志级别（来自 customExtra.agentLogLevel）
	setAgentLogLevel(&param, bkeConfig)

	// 设置NFS参数
	setNFSParams(&param, bkeCluster)

	// 设置命名空间与集群名（自纳管清单继承引导侧管理集群名称）
	param["namespace"] = bkeCluster.Namespace
	param["clusterName"] = bkeCluster.Name

	// 设置网络参数
	setNetworkParams(&param, bkeConfig, bkeCluster)

	// 设置etcd端点
	// Note: bkeNodes is now passed as parameter instead of using cluster.GetNodesData
	setEtcdEndpoints(&param, bkeNodes)

	// 设置节点副本数
	setNodeReplicas(&param, bkeNodes)

	// 设置附加组件特定参数
	setAddonParams(&param, bkeConfig)

	// 设置DNS IP
	if err := setDNSIP(&param); err != nil {
		return nil, err
	}

	// 设置etcd IP列表
	setEtcdIPs(&param, bkeNodes)

	// 设置Kubelet数据根目录
	setKubeletDataRoot(&param, bkeConfig)

	// 设置Docker数据根目录
	setDockerDataRoot(&param, bkeConfig)

	// 设置Kubernetes版本
	setK8sVersion(&param, bkeConfig)

	// 设置Etcd版本
	setEtcdVersion(&param, bkeConfig)

	// 设置Containerd版本
	setContainerdVersion(&param, bkeConfig)

	// 设置 containerdConfigRef / KubeletConfigRef，供管理集群纳管 003-manage.yaml 透传
	setContainerdConfigRef(&param, bkeConfig)
	setKubeletConfigRef(&param, bkeCluster)

	return param, nil
}

// initializeDefaultParams 初始化默认参数
func initializeDefaultParams() map[string]interface{} {
	param := make(map[string]interface{})
	param["replicas"] = 1
	param["kubeConfigDir"] = "/etc/kubernetes"
	param["ipModVersion"] = "ipv4"
	param["apiServerSrcPort"] = bkeinit.DefaultAPIBindPort
	param["podSubnet"] = bkeinit.DefaultPodSubnet
	param["serviceSubnet"] = bkeinit.DefaultServicesSubnet
	param["dnsIP"] = bkeinit.DefaultClusterDNSIP
	param["dnsDomain"] = bkeinit.DefaultServiceDNSDomain
	param["ingressReplicas"] = 1
	param["etcdEndpoints"] = ""
	param["version"] = v.Version
	param["kubeletDataRoot"] = bkeinit.DefaultKubeletRootDir
	param["dockerDataRoot"] = bkeinit.DefaultCRIDockerDataRootDir
	param["k8sVersion"] = versions.KubernetesVersion()
	param["masterReplicas"] = "1"
	param["workerReplicas"] = "1"
	param["namespace"] = "cluster-system"
	param["clusterName"] = "bke-cluster"
	param["clusterNetworkMode"] = ""
	param["product"] = "boc"
	return param
}

// setImageRepoParams 设置镜像仓库参数
func setImageRepoParams(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	(*param)["imageRepo"] = fmt.Sprintf("%s:%s", bkeConfig.Cluster.ImageRepo.Domain, bkeConfig.Cluster.ImageRepo.Port)
	(*param)["imageRepoDomain"] = bkeConfig.Cluster.ImageRepo.Domain
	(*param)["imageRepoPort"] = bkeConfig.Cluster.ImageRepo.Port
	(*param)["imageRepoIp"] = bkeConfig.Cluster.ImageRepo.Ip
	(*param)["imageRepoPrefix"] = bkeConfig.Cluster.ImageRepo.Prefix
}

// setChartRepoParams 设置Chart仓库参数
func setChartRepoParams(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	chartRepo := bkeConfig.Cluster.ChartRepo
	(*param)["chartRepoDomain"] = chartRepo.Domain
	(*param)["chartRepoPort"] = chartRepo.Port
	(*param)["chartRepoIp"] = chartRepo.Ip
	(*param)["chartRepoPrefix"] = chartRepo.Prefix
	(*param)["chartRepoInsecureSkipTLSVerify"] = chartRepo.InsecureSkipTLSVerify
	if chartRepo.AuthSecretRef != nil {
		(*param)["chartRepoAuthSecretName"] = chartRepo.AuthSecretRef.Name
		(*param)["chartRepoAuthSecretNamespace"] = chartRepo.AuthSecretRef.Namespace
		(*param)["chartRepoAuthSecretUsernameKey"] = chartRepo.AuthSecretRef.UsernameKey
		(*param)["chartRepoAuthSecretPasswordKey"] = chartRepo.AuthSecretRef.PasswordKey
	}
	if chartRepo.TlsSecretRef != nil {
		(*param)["chartRepoTlsSecretName"] = chartRepo.TlsSecretRef.Name
		(*param)["chartRepoTlsSecretNamespace"] = chartRepo.TlsSecretRef.Namespace
		(*param)["chartRepoTlsSecretCertKey"] = chartRepo.TlsSecretRef.CertKey
		(*param)["chartRepoTlsSecretCAKey"] = chartRepo.TlsSecretRef.CaKey
		(*param)["chartRepoTlsSecretKeyKey"] = chartRepo.TlsSecretRef.KeyKey
	}
}

// setHTTPRepoParams 设置HTTP仓库参数
func setHTTPRepoParams(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	(*param)["httpRepo"] = bkeConfig.YumRepo()
	(*param)["httpRepoDomain"] = bkeConfig.Cluster.HTTPRepo.Domain
	(*param)["httpRepoPort"] = bkeConfig.Cluster.HTTPRepo.Port
	(*param)["httpRepoIp"] = bkeConfig.Cluster.HTTPRepo.Ip
	(*param)["httpRepoPrefix"] = bkeConfig.Cluster.HTTPRepo.Prefix
}

// setNTPServer 设置NTP服务器参数
func setNTPServer(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	(*param)["ntpServer"] = bkeConfig.Cluster.NTPServer
}

// setAgentHealthPort 设置Agent健康端口参数
func setAgentHealthPort(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	(*param)["agentHealthPort"] = bkeConfig.Cluster.AgentHealthPort
}

// setAgentLogLevel 设置 bkeagent 业务日志级别参数（供纳管 BC / addon 模板渲染）
func setAgentLogLevel(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	level := ""
	if bkeConfig.CustomExtra != nil {
		level = bkeConfig.CustomExtra["agentLogLevel"]
	}
	if canonical, ok := log.NormalizeAllowedLevel(level); ok {
		(*param)["agentLogLevel"] = canonical
		return
	}
	(*param)["agentLogLevel"] = "INFO"
}

// setNFSParams 设置NFS参数
func setNFSParams(param *map[string]interface{}, bkeCluster *bkev1beta1.BKECluster) {
	(*param)["nfsServer"] = bkeCluster.Spec.ClusterConfig.CustomExtra["nfsServer"]
	(*param)["nfsRootDir"] = bkeCluster.Spec.ClusterConfig.CustomExtra["nfsRootDir"]
	(*param)["nfsVersion"] = bkeCluster.Spec.ClusterConfig.CustomExtra["nfsVersion"]
}

// setNetworkParams 设置网络参数
func setNetworkParams(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig, bkeCluster *bkev1beta1.BKECluster) {
	if bkeConfig.Cluster.Networking.PodSubnet != "" {
		(*param)["podSubnet"] = bkeConfig.Cluster.Networking.PodSubnet
	}
	if bkeConfig.Cluster.Networking.ServiceSubnet != "" {
		(*param)["serviceSubnet"] = bkeConfig.Cluster.Networking.ServiceSubnet
	}
	if bkeConfig.Cluster.Networking.DNSDomain != "" {
		(*param)["dnsDomain"] = bkeConfig.Cluster.Networking.DNSDomain
	}
	if bkeCluster.Spec.ControlPlaneEndpoint.Host != "" {
		(*param)["apiServerSrcHost"] = bkeCluster.Spec.ControlPlaneEndpoint.Host
	}
	if bkeCluster.Spec.ControlPlaneEndpoint.Port != 0 {
		(*param)["apiServerSrcPort"] = bkeCluster.Spec.ControlPlaneEndpoint.Port
	}
}

// setEtcdEndpoints 设置etcd端点
func setEtcdEndpoints(param *map[string]interface{}, bkeNodes bkenode.Nodes) {
	var etcdEndpointsLi []string
	for _, node := range bkeNodes.Etcd() {
		tmp := fmt.Sprintf("https://%s:2379", node.IP)
		etcdEndpointsLi = append(etcdEndpointsLi, tmp)
	}
	(*param)["etcdEndpoints"] = strings.Join(etcdEndpointsLi, ",")
}

// setNodeReplicas 设置节点副本数
func setNodeReplicas(param *map[string]interface{}, bkeNodes bkenode.Nodes) {
	(*param)["masterReplicas"] = strconv.Itoa(bkeNodes.Master().Length())
	(*param)["workerReplicas"] = strconv.Itoa(bkeNodes.Worker().Length())
}

// setAddonParams 设置附加组件参数
func setAddonParams(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	// for beyondelb replicas
	for _, v := range bkeConfig.Addons {
		name := strings.ToLower(v.Name)
		if name == "beyondelb" {
			lbNodes := v.Param["lbNodes"]
			(*param)["ingressReplicas"] = len(strings.Split(lbNodes, ","))
		}
		if name == "calico" {
			(*param)["clusterNetworkMode"] = "calico"
		}
		if name == "fabric" {
			(*param)["clusterNetworkMode"] = "fabric"
		}
	}
}

// setDNSIP 设置DNS IP
func setDNSIP(param *map[string]interface{}) error {
	dnsIP, err := bkeinit.GetClusterDNSIP((*param)["serviceSubnet"].(string))
	if err != nil {
		return errors.New("failed to get cluster dns ip")
	}
	(*param)["dnsIP"] = dnsIP
	return nil
}

// setEtcdIPs 设置etcd IP列表
func setEtcdIPs(param *map[string]interface{}, bkeNodes bkenode.Nodes) {
	etcdNodes := bkeNodes.Etcd()
	var etcdEndpoints []string
	for _, node := range etcdNodes {
		etcdEndpoints = append(etcdEndpoints, node.IP)
	}
	(*param)["etcdIps"] = strings.Join(etcdEndpoints, ",")
}

// setKubeletDataRoot 设置Kubelet数据根目录
func setKubeletDataRoot(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	if bkeConfig.Cluster.Kubelet != nil && bkeConfig.Cluster.Kubelet.ExtraVolumes != nil {
		for _, volume := range bkeConfig.Cluster.Kubelet.ExtraVolumes {
			if volume.Name == "kubelet-root-dir" {
				(*param)["kubeletDataRoot"] = volume.HostPath
			}
		}
	}
}

// setDockerDataRoot 设置Docker数据根目录
func setDockerDataRoot(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	if bkeConfig.Cluster.ContainerRuntime.CRI == bkeinit.CRIDocker {
		if v, ok := bkeConfig.Cluster.ContainerRuntime.Param["data-root"]; ok {
			(*param)["dockerDataRoot"] = v
		}
	}
}

// setK8sVersion 设置Kubernetes版本
func setK8sVersion(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	if bkeConfig.Cluster.KubernetesVersion != "" {
		(*param)["k8sVersion"] = bkeConfig.Cluster.KubernetesVersion
	}
}

// setEtcdVersion 设置Etcd版本
func setEtcdVersion(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	if bkeConfig.Cluster.EtcdVersion != "" {
		(*param)["etcdVersion"] = bkeConfig.Cluster.EtcdVersion
	}
}

// setContainerdVersion 设置Containerd版本
func setContainerdVersion(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	if bkeConfig.Cluster.ContainerdVersion != "" {
		(*param)["containerdVersion"] = bkeConfig.Cluster.ContainerdVersion
	}
}

// setContainerdConfigRef 设置 containerdConfigRef，供管理集群纳管清单透传
func setContainerdConfigRef(param *map[string]interface{}, bkeConfig bkeinit.BkeConfig) {
	ref := bkeConfig.Cluster.ContainerdConfigRef
	if ref == nil || ref.Name == "" {
		return
	}
	(*param)["containerdConfigRefName"] = ref.Name
	if ref.Namespace != "" {
		(*param)["containerdConfigRefNamespace"] = ref.Namespace
	}
}

// setKubeletConfigRef 设置 KubeletConfigRef，供管理集群纳管清单透传
func setKubeletConfigRef(param *map[string]interface{}, bkeCluster *bkev1beta1.BKECluster) {
	if bkeCluster == nil || bkeCluster.Spec.KubeletConfigRef == nil || bkeCluster.Spec.KubeletConfigRef.Name == "" {
		return
	}
	ref := bkeCluster.Spec.KubeletConfigRef
	(*param)["kubeletConfigRefName"] = ref.Name
	if ref.Namespace != "" {
		(*param)["kubeletConfigRefNamespace"] = ref.Namespace
	}
}

// convertNodesToManageTemplateData converts Spec nodes to manage-template maps.
// BN name follows bkeadm convention: <clusterName>-<hostname>.
func convertNodesToManageTemplateData(bkeNodes bkenode.Nodes, clusterName string) []map[string]interface{} {
	nodesData := make([]map[string]interface{}, 0, len(bkeNodes))
	for _, node := range bkeNodes {
		name := node.Hostname
		if clusterName != "" && node.Hostname != "" {
			name = fmt.Sprintf("%s-%s", clusterName, node.Hostname)
		}
		nodeData := map[string]interface{}{
			"name":     name,
			"hostname": node.Hostname,
			"ip":       node.IP,
			"username": node.Username,
			"password": node.Password,
			"port":     node.Port,
			"role":     node.Role,
		}
		nodesData = append(nodesData, nodeData)
	}
	return nodesData
}

// ConvertBKENodesToManageTemplateData converts full BKENodes so self-manage
// inherits bootstrap-side metadata.name when present.
func ConvertBKENodesToManageTemplateData(bkeNodes bkev1beta1.BKENodes, clusterName string) []map[string]interface{} {
	nodesData := make([]map[string]interface{}, 0, len(bkeNodes))
	for _, node := range bkeNodes {
		name := strings.TrimSpace(node.Name)
		if name == "" {
			hostname := node.Spec.Hostname
			if clusterName != "" && hostname != "" {
				name = fmt.Sprintf("%s-%s", clusterName, hostname)
			} else {
				name = hostname
			}
		}
		nodeData := map[string]interface{}{
			"name":     name,
			"hostname": node.Spec.Hostname,
			"ip":       node.Spec.IP,
			"username": node.Spec.Username,
			"password": node.Spec.Password,
			"port":     node.Spec.Port,
			"role":     node.Spec.Role,
		}
		nodesData = append(nodesData, nodeData)
	}
	return nodesData
}

// normalizeNodeSelector normalize nodeSelector param from string to map[string]string
func normalizeNodeSelector(param map[string]map[string]interface{}) error {
	for file, kv := range param {
		raw, ok := kv["nodeSelector"]
		if !ok || raw == nil {
			continue
		}
		switch v := raw.(type) {
		case map[string]string, map[string]interface{}:
			continue
		case string:
			selector, err := parseNodeSelector(v)
			if err != nil {
				return errors.Wrapf(err, "%s.nodeSelector", file)
			}
			kv["nodeSelector"] = selector
		default:
			return errors.Errorf("%s.nodeSelector unsupported type %T", file, v)
		}
	}
	return nil
}

// parseNodeSelector parse nodeSelector param from string to map[string]string
func parseNodeSelector(raw string) (map[string]string, error) {
	out := make(map[string]string)
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return out, nil
	}
	if strings.HasPrefix(raw, "{") {
		if err := json.Unmarshal([]byte(raw), &out); err != nil {
			return nil, err
		}
		return out, nil
	}
	for _, pair := range strings.Split(raw, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		kv := strings.SplitN(pair, "=", keyValueSeparatorCount)
		key := strings.TrimSpace(kv[0])
		if key == "" {
			return nil, errors.Errorf("invalid nodeSelector entry %q", pair)
		}
		val := ""
		if len(kv) == keyValueSeparatorCount {
			val = strings.TrimSpace(kv[1])
		}
		out[key] = val
	}
	return out, nil
}

// parseFabricParam parse fabric param excludeIps
func parseFabricParam(addon *confv1beta1.Product) (map[string]string, error) {
	fabricParam := addon.Param

	if !hasExcludeIpsParam(fabricParam) {
		return fabricParam, nil
	}

	excludeIpsStr := fabricParam["excludeIps"]
	excludeIps, err := processExcludeIps(excludeIpsStr)
	if err != nil {
		return nil, err
	}

	fabricParam["excludeIps"] = excludeIps
	return fabricParam, nil
}

// hasExcludeIpsParam 检查是否包含excludeIps参数
func hasExcludeIpsParam(fabricParam map[string]string) bool {
	p, ok := fabricParam["excludeIps"]
	return ok && p != ""
}

// processExcludeIps 处理excludeIps参数
func processExcludeIps(excludeIpsStr string) (string, error) {
	items := strings.Split(excludeIpsStr, ",")
	var excludeIps []string

	for _, item := range items {
		validatedIP, err := processItem(item)
		if err != nil {
			return "", err
		}
		if validatedIP != "" {
			excludeIps = append(excludeIps, validatedIP)
		}
	}

	return strings.Join(excludeIps, ","), nil
}

// processItem 处理单个项目
func processItem(item string) (string, error) {
	if strings.Contains(item, "-") {
		ips, err := parseFabricExcludeIPRange(item)
		if err != nil {
			return "", err
		}
		return strings.Join(ips, ","), nil
	}

	ip := net.ParseIP(item)
	if ip != nil {
		return ip.String(), nil
	}

	return "", errors.Errorf("fabric param excludeIps contain a invalid ip %s", item)
}

func mergeParam(src, dst map[string]interface{}) map[string]interface{} {
	// Check if dst is nil, if so create a new map
	if dst == nil {
		dst = make(map[string]interface{})
	}

	// Check if src is nil, if so return dst as is
	if src == nil {
		return dst
	}

	for k, v := range src {
		if _, ok := dst[k]; !ok {
			dst[k] = v
		}
	}
	return dst
}

func prepareAddonParam(addonParam map[string]string, filesBaseNames []string, repo string) map[string]map[string]interface{} {
	param := make(map[string]map[string]interface{})
	for _, fileName := range filesBaseNames {
		param[fileName] = make(map[string]interface{})
	}
	for k, v := range addonParam {
		arg := strings.Split(k, ".")
		if len(arg) == 1 {
			for _, fileName := range filesBaseNames {
				param[fileName][k] = v
			}
		} else {
			if param[arg[0]] == nil {
				param[arg[0]] = make(map[string]interface{})
			}
			param[arg[0]][arg[1]] = v
		}
	}
	// add default param repo
	for _, v := range param {
		if _, ok := v["repo"]; !ok {
			v["repo"] = repo
		}
	}
	return param
}

func parseFabricExcludeIPRange(rangeStr string) ([]string, error) {
	ipRange := strings.Split(rangeStr, "-")

	if len(ipRange) != keyValueSeparatorCount || ipRange == nil {
		return nil, errors.Errorf("fabric param excludeIps contain a invalid ip range %s", rangeStr)
	}
	start := net.ParseIP(ipRange[0]).To4()
	end := net.ParseIP(ipRange[1]).To4()
	if start == nil || end == nil {
		return nil, errors.Errorf("fabric param excludeIps contain a invalid ip range %s", rangeStr)
	}

	ipLe := func(src, dst net.IP) bool {
		if src.Equal(dst) {
			return false
		}
		for i, value := range src {
			if value > dst[i] {
				return true
			}
		}
		return false
	}

	if ipLe(start, end) {
		tmp := start
		start = end
		end = tmp
	}

	var ips []string
	ips = append(ips, start.String())
	for ip := start; !ip.Equal(end); {
		for j := len(ip) - 1; j >= 0; j-- {
			ip[j]++
			if ip[j] > 0 {
				break
			}
		}
		ips = append(ips, ip.String())
	}
	if !utils.ContainsString(ips, end.String()) {
		ips = append(ips, end.String())
	}
	return ips, nil
}
