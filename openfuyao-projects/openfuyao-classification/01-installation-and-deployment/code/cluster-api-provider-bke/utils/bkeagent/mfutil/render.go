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

package mfutil

import (
	"bytes"
	"embed"
	_ "embed"
	"fmt"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v3"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"

	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/versionutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/clientutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	bkeetcd "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/etcd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

//
//go:embed  tmpl/*
var f embed.FS

const (
	keepalivedBaseTmpl = `
{{ range $instance := .instances }}
{{ $instance }}
{{ end }}
`
	// CertChainFileName defines certificate ca and chain name
	CertCAAndChainFileName = "ca-chain.crt"
)

func generateRandomString(n int) string {
	result := make([]byte, n)
	rand.Seed(time.Now().UnixNano())
	for i := range result {
		result[i] = randomStringLetters[rand.Intn(len(randomStringLetters))]
	}
	return string(result)
}

type TemplateRenderOptions struct {
	Name     string            // 模板名称
	Template []byte            // 模板内容
	Data     interface{}       // 模板数据
	FilePath string            // 输出文件路径
	FuncMap  *template.FuncMap // 模板函数映射
	FileMode os.FileMode       // 文件权限
}

func (o *TemplateRenderOptions) Validate() error {
	if o.Name == "" {
		return fmt.Errorf("template name is required")
	}
	if len(o.Template) == 0 {
		return fmt.Errorf("template content is empty")
	}
	if o.FilePath == "" {
		return fmt.Errorf("output file path is required")
	}
	return nil
}

// renderTemplateAndStore renders a Go template with the given configuration and stores it to a file
func renderTemplateAndStore(options *TemplateRenderOptions) error {
	if err := options.Validate(); err != nil {
		return fmt.Errorf("invalid template render options: %w", err)
	}

	var t *template.Template
	var err error

	if options.FuncMap != nil {
		t, err = template.New(options.Name).Funcs(*options.FuncMap).Parse(string(options.Template))
		if err != nil {
			return fmt.Errorf("failed to parse template with funcMap: %w", err)
		}
	} else {
		t, err = template.New(options.Name).Parse(string(options.Template))
		if err != nil {
			return fmt.Errorf("failed to parse template: %w", err)
		}
	}

	if t == nil {
		return errors.New("template is nil after parsing")
	}

	// 先渲染到内存，成功后一次性写盘：
	// 模板函数执行中途失败时直接流式写文件会留下部分渲染产物
	// （如 etcd.yaml 缺失 image/name 被 kubelet 拒绝），必须整体成功或完全不写。
	var buf bytes.Buffer
	if err := t.Execute(&buf, options.Data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}
	if err := os.WriteFile(options.FilePath, buf.Bytes(), options.FileMode); err != nil {
		return fmt.Errorf("failed to write file %s: %w", options.FilePath, err)
	}

	return nil
}

// renderK8sAndStore renders a Kubernetes component template and stores it
func renderK8sAndStore(c *BKEComponent, tmpl []byte, cfg interface{}, funcMap *template.FuncMap) error {
	filePath := pathForManifest(c)
	// 先渲染到临时文件，校验通过后才替换目标文件：
	// 1) 无效 manifest（如 image/name 为空）不允许覆盖 manifests 目录中的现有文件，
	//    否则 kubelet 拒绝处理后静态 Pod 无法重建（历史上曾渲染出无效 etcd.yaml）；
	// 2) 内容未变化时不替换，避免刷新 mtime 触发无谓的 Pod 重建。
	tmpPath := filePath + ".bketmp"
	options := &TemplateRenderOptions{
		Name:     c.Name,
		Template: tmpl,
		Data:     cfg,
		FilePath: tmpPath,
		FuncMap:  funcMap,
		FileMode: utils.RwRR,
	}
	if err := renderTemplateAndStore(options); err != nil {
		cleanupTempManifest(tmpPath)
		return err
	}
	if err := dedupRenderedPodCommand(tmpPath); err != nil {
		cleanupTempManifest(tmpPath)
		return err
	}
	if err := validatePodManifest(tmpPath); err != nil {
		cleanupTempManifest(tmpPath)
		return errors.Wrapf(err, "rendered %s static pod manifest is invalid, refuse to write %s", c.Name, filePath)
	}
	if sameFileContent(tmpPath, filePath) {
		cleanupTempManifest(tmpPath)
		return nil
	}
	return os.Rename(tmpPath, filePath)
}

// cleanupTempManifest 清理渲染临时文件。删除失败仅记录，不影响主流程返回的错误。
func cleanupTempManifest(tmpPath string) {
	if err := os.Remove(tmpPath); err != nil {
		log.Debugf("cleanup temp manifest %s failed: %v", tmpPath, err)
	}
}

// sameFileContent 判断两个文件内容是否一致（任一文件读取失败视为不一致）。
func sameFileContent(pathA, pathB string) bool {
	dataA, err := os.ReadFile(pathA)
	if err != nil {
		return false
	}
	dataB, err := os.ReadFile(pathB)
	if err != nil {
		return false
	}
	return bytes.Equal(dataA, dataB)
}

// validatePodManifest 校验渲染出的 static pod manifest 基础有效性：
// metadata.name、spec.containers 非空，且每个容器 name/image 必填。
// 缺失时 kubelet 会直接拒绝处理该 manifest（invalid pod），静态 Pod 无法创建或重建。
func validatePodManifest(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	var manifest struct {
		Metadata struct {
			Name string `yaml:"name"`
		} `yaml:"metadata"`
		Spec struct {
			Containers []struct {
				Name  string `yaml:"name"`
				Image string `yaml:"image"`
			} `yaml:"containers"`
		} `yaml:"spec"`
	}
	if err := yaml.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("failed to parse pod manifest: %w", err)
	}
	if manifest.Metadata.Name == "" {
		return errors.New("metadata.name is required")
	}
	if len(manifest.Spec.Containers) == 0 {
		return errors.New("spec.containers is required")
	}
	for i, c := range manifest.Spec.Containers {
		if c.Name == "" {
			return fmt.Errorf("spec.containers[%d].name is required", i)
		}
		if c.Image == "" {
			return fmt.Errorf("spec.containers[%d].image is required", i)
		}
	}
	return nil
}

// dedupRenderedPodCommand 对渲染后的 Pod manifest 执行命令行参数去重。
// 模板内置参数和 extraArgs 都展开后再处理，确保用户 extraArgs 能按 last-wins 覆盖默认参数。
func dedupRenderedPodCommand(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read rendered pod manifest %s: %w", filePath, err)
	}

	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("failed to parse rendered pod manifest %s: %w", filePath, err)
	}

	if dedupPodCommandNode(&doc) {
		out, err := yaml.Marshal(&doc)
		if err != nil {
			return fmt.Errorf("failed to marshal rendered pod manifest %s: %w", filePath, err)
		}
		if err := os.WriteFile(filePath, out, utils.RwRR); err != nil {
			return fmt.Errorf("failed to write rendered pod manifest %s: %w", filePath, err)
		}
	}
	return nil
}

// dedupPodCommandNode 查找 spec.containers[].command 并按 flag key 去重。
func dedupPodCommandNode(doc *yaml.Node) bool {
	root := doc
	if root.Kind == yaml.DocumentNode && len(root.Content) > 0 {
		root = root.Content[0]
	}
	spec := mappingValue(root, "spec")
	containers := mappingValue(spec, "containers")
	if containers == nil || containers.Kind != yaml.SequenceNode {
		return false
	}

	changed := false
	for _, container := range containers.Content {
		command := mappingValue(container, "command")
		if command == nil || command.Kind != yaml.SequenceNode {
			continue
		}
		args := make([]string, 0, len(command.Content))
		for _, item := range command.Content {
			args = append(args, item.Value)
		}
		deduped := DedupFlags(args)
		if utils.StringSlicesEqual(args, deduped) {
			continue
		}
		command.Content = command.Content[:0]
		for _, arg := range deduped {
			command.Content = append(command.Content, &yaml.Node{
				Kind:  yaml.ScalarNode,
				Tag:   "!!str",
				Value: arg,
			})
		}
		changed = true
	}
	return changed
}

// stringSlicesEqual 判断参数列表是否发生变化，避免无变更时重写 manifest。

// mappingValue 从 YAML mapping 节点中取指定 key 的 value 节点。
func mappingValue(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return node.Content[i+1]
		}
	}
	return nil
}

// renderHAAndStore renders a HA component template and stores it
func renderHAAndStore(c *BKEHAComponent, tmpl []byte, cfg map[string]interface{}, filePath string, funcMap *template.FuncMap) error {
	options := &TemplateRenderOptions{
		Name:     c.Name,
		Template: tmpl,
		Data:     cfg,
		FilePath: filePath,
		FuncMap:  funcMap,
		FileMode: utils.RwRR,
	}
	return renderTemplateAndStore(options)
}

func renderAPIServer(c *BKEComponent, cfg *BootScope) error {
	tmpl, err := fs.ReadFile(f, "tmpl/k8s/kube-apiserver.yaml.tmpl")
	if err != nil {
		return err
	}
	file, err := fs.ReadFile(f, "tmpl/k8s/audit-policy.yaml")
	if err != nil {
		return err
	}
	// write the audit policy file,default to /etc/kubernetes/audit-policy.yaml
	if err := os.WriteFile(GetAuditPolicyFilePath(), file, utils.RwRR); err != nil {
		return err
	}
	return renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), apiServerFuncMap()))
}

func renderEtcd(c *BKEComponent, cfg *BootScope) error {
	if cfg == nil {
		return errors.New("config is nil")
	}

	// Step 1: Handle etcd cluster membership
	if err := handleEtcdMembership(cfg); err != nil {
		return err
	}

	// Step 2: Render etcd yaml
	return renderEtcdYaml(c, cfg)
}

// handleEtcdMembership handles etcd cluster initialization or node addition.
// For config update / upgrade of an existing member, prefer reusing --initial-cluster
// from the local etcd static Pod manifest so rendering does not depend on the workload
// apiserver (NewFromCluster lists etcd pods via 6443).
func handleEtcdMembership(cfg *BootScope) error {
	etcdPeerAddress := etcd.GetPeerURL(&kubeadmapi.APIEndpoint{AdvertiseAddress: cfg.HostIP})

	// Safely get Init flag
	initValue, initExists := cfg.Extra["Init"]
	isInit := true // Default to initialization mode
	if initExists {
		initBool, ok := initValue.(bool)
		if !ok {
			return fmt.Errorf("init flag is not boolean, got: %T", initValue)
		}
		isInit = initBool
	}

	if isInit {
		// New cluster initialization
		cfg.Extra["EtcdInitialCluster"] = []etcd.Member{{Name: cfg.HostName, PeerURL: etcdPeerAddress}}
		return nil
	}

	if members, ok := etcdInitialClusterFromLocalManifest(cfg); ok {
		log.Infof("reusing etcd initial-cluster from local manifest (%d members)", len(members))
		cfg.Extra["EtcdInitialCluster"] = members
		return nil
	}

	// Join an existing cluster (no usable local etcd.yaml yet)
	return addNodeToExistingEtcdCluster(cfg, etcdPeerAddress)
}

// etcdManifestPath resolves the on-disk etcd static Pod path.
func etcdManifestPath(cfg *BootScope) string {
	dir := GetDefaultManifestsPath()
	if cfg != nil && cfg.BkeConfig != nil && cfg.BkeConfig.Cluster.Kubelet != nil &&
		cfg.BkeConfig.Cluster.Kubelet.ManifestsDir != "" {
		dir = cfg.BkeConfig.Cluster.Kubelet.ManifestsDir
	}
	if cfg != nil && cfg.CurrentNode.Kubelet != nil && cfg.CurrentNode.Kubelet.ManifestsDir != "" {
		dir = cfg.CurrentNode.Kubelet.ManifestsDir
	}
	return filepath.Join(dir, "etcd.yaml")
}

// parseEtcdInitialClusterMembers parses "name=https://ip:2380,..." into kubeadm etcd members.
func parseEtcdInitialClusterMembers(initialCluster string) ([]etcd.Member, bool) {
	initialCluster = strings.TrimSpace(initialCluster)
	if initialCluster == "" {
		return nil, false
	}
	parts := strings.Split(initialCluster, ",")
	members := make([]etcd.Member, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		idx := strings.Index(part, "=")
		if idx <= 0 || idx == len(part)-1 {
			return nil, false
		}
		members = append(members, etcd.Member{
			Name:    part[:idx],
			PeerURL: part[idx+1:],
		})
	}
	return members, len(members) > 0
}

// etcdInitialClusterFromLocalManifest reads --initial-cluster from the local etcd.yaml.
func etcdInitialClusterFromLocalManifest(cfg *BootScope) ([]etcd.Member, bool) {
	manifest := etcdManifestPath(cfg)
	data, err := os.ReadFile(manifest)
	if err != nil {
		log.Infof("local etcd manifest %q not available for initial-cluster reuse: %v", manifest, err)
		return nil, false
	}

	var pod struct {
		Spec struct {
			Containers []struct {
				Command []string `yaml:"command"`
			} `yaml:"containers"`
		} `yaml:"spec"`
	}
	if err := yaml.Unmarshal(data, &pod); err != nil {
		log.Warnf("failed to parse local etcd manifest %q: %v", manifest, err)
		return nil, false
	}
	for _, container := range pod.Spec.Containers {
		for _, arg := range container.Command {
			if strings.HasPrefix(arg, "--initial-cluster=") {
				return parseEtcdInitialClusterMembers(strings.TrimPrefix(arg, "--initial-cluster="))
			}
		}
	}
	return nil, false
}

// addNodeToExistingEtcdCluster adds node to existing etcd cluster
func addNodeToExistingEtcdCluster(cfg *BootScope, etcdPeerAddress string) error {
	// Safely get mccs
	mccsValue, mccsExists := cfg.Extra["mccs"]
	if !mccsExists {
		return errors.New("mccs not found in config")
	}

	mccs, ok := mccsValue.([]string)
	if !ok {
		return fmt.Errorf("mccs is not []string, got: %T", mccsValue)
	}

	// Create etcd client
	client, err := clientutil.ClientSetFromManagerClusterSecret(mccs...)
	if err != nil {
		return errors.Wrapf(err, "Failed to get kubernetes client")
	}

	etcdClient, err := etcd.NewFromCluster(client, cfg.BkeConfig.Cluster.CertificatesDir)
	if err != nil {
		return errors.Wrapf(err, "Failed to create etcd client")
	}
	if etcdClient == nil {
		return errors.New("etcd client is nil")
	}

	// Handle etcd membership
	initialCluster, err := etcdClient.ListMembers()
	if err != nil {
		return errors.Wrapf(err, "Failed to list etcd members")
	}

	// Check if member already exists
	log.Infof("checking if the etcd member already exists: %s", etcdPeerAddress)
	memberExists := false
	for i := range initialCluster {
		if initialCluster[i].PeerURL == etcdPeerAddress {
			memberExists = true
			if len(initialCluster[i].Name) == 0 {
				initialCluster[i].Name = cfg.HostName
			}
			break
		}
	}

	if memberExists {
		log.Infof("etcd member already exists: %q", etcdPeerAddress)
		cfg.Extra["EtcdInitialCluster"] = initialCluster
	} else {
		log.Infof("adding etcd member: %s", etcdPeerAddress)
		newMembers, err := etcdClient.AddMember(cfg.HostName, etcdPeerAddress)
		if err != nil {
			return errors.Wrapf(err, "Failed to add etcd member")
		}
		cfg.Extra["EtcdInitialCluster"] = newMembers
		log.Infof("Updated etcd member list: %v", newMembers)
	}

	return nil
}

// renderEtcdYaml renders etcd yaml manifest
func renderEtcdYaml(c *BKEComponent, cfg *BootScope) error {
	tmpl, err := fs.ReadFile(f, "tmpl/k8s/etcd.yaml.tmpl")
	if err != nil {
		return errors.Wrapf(err, "Failed to read etcd template")
	}

	cfg.Extra["EtcdAdvertiseUrls"] = etcd.GetClientURLByIP(cfg.HostIP)

	funcMap := mergeFuncMap(GlobalFuncMap(), etcdFuncMap())
	if err = renderK8sAndStore(c, tmpl, cfg, funcMap); err != nil {
		return errors.Wrapf(err, "Failed to render etcd yaml")
	}

	return nil
}

func renderController(c *BKEComponent, cfg *BootScope) error {
	tmpl, err := fs.ReadFile(f, "tmpl/k8s/kube-controller-manager.yaml.tmpl")
	if err != nil {
		return err
	}
	return renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), controllerFuncMap()))
}

func renderScheduler(c *BKEComponent, cfg *BootScope) error {
	// todo 还需要处理一个文件 gpu-admission.config
	if v, ok := cfg.Extra["gpuEnable"]; ok && v.(string) == "true" {
		policyFile, err := fs.ReadFile(f, "tmpl/k8s/scheduler-policy-config.json")
		if err != nil {
			return err
		}
		if err = os.WriteFile(GetSchedulerPolicyFilePath(), policyFile, utils.RwxRxRx); err != nil {
			return err
		}
		admissionConfig, err := fs.ReadFile(f, "tmpl/k8s/gpu-admission.config")
		if err != nil {
			return err
		}
		if err = os.WriteFile(GetSchedulerAdmissionConfigFilePath(), admissionConfig, utils.RwxRxRx); err != nil {
			return err
		}
	}

	tmpl, err := fs.ReadFile(f, "tmpl/k8s/kube-scheduler.yaml.tmpl")
	if err != nil {
		return err
	}
	return renderK8sAndStore(c, tmpl, cfg, mergeFuncMap(GlobalFuncMap(), schedulerFuncMap()))
}

// HA component yaml render func
func renderHAProxy(c *BKEHAComponent, cfg map[string]interface{}) error {
	v, ok := cfg["haproxyConfigDir"]
	if !ok {
		return errors.New("haproxyConfigDir not found in config")
	}

	c.ConfPath, ok = v.(string)
	if !ok {
		return fmt.Errorf("haproxyConfigDir is not a string: %v", v)
	}

	// step1 render conf file
	tmpl, err := fs.ReadFile(f, "tmpl/haproxy/haproxy.cfg.tmpl")
	if err != nil {
		return err
	}
	if err := renderHAAndStore(c, tmpl, cfg, pathForHAManifestConf(c), nil); err != nil {
		return err
	}
	// step2 render yaml file
	tmpl, err = fs.ReadFile(f, "tmpl/haproxy/haproxy.yaml.tmpl")
	if err != nil {
		return err
	}

	if err := renderHAAndStore(c, tmpl, cfg, pathForHAManifest(c), utilFuncMap()); err != nil {
		return err
	}
	return nil
}

func renderKeepalived(c *BKEHAComponent, cfg map[string]interface{}) error {
	if err := validateConfig(cfg); err != nil {
		return err
	}

	if err := setupConfigPath(c, cfg); err != nil {
		return err
	}

	isMasterHa, err := getIsMasterHA(cfg)
	if err != nil {
		return err
	}

	// Step 1: Render check script
	if err = renderCheckScript(c, cfg, isMasterHa); err != nil {
		return err
	}

	// Step 2: Render instance
	if err = renderKeepalivedInstance(c, cfg, isMasterHa); err != nil {
		return err
	}

	// Step 3: Get base template
	baseTmpl, err := getBaseTemplate(c, cfg, isMasterHa)
	if err != nil {
		return err
	}

	// Step 4: Render conf file
	if err = renderConfigFile(c, cfg, baseTmpl); err != nil {
		return err
	}

	// Step 5: Render yaml file
	return renderYamlFile(c, cfg)
}

// validateConfig validates the input configuration
func validateConfig(cfg map[string]interface{}) error {
	if cfg == nil {
		return errors.New("config is nil")
	}
	return nil
}

// setupConfigPath sets up the configuration path from cfg
func setupConfigPath(c *BKEHAComponent, cfg map[string]interface{}) error {
	if v, ok := cfg["keepAlivedConfigDir"]; ok {
		configDir, ok := v.(string)
		if !ok {
			return fmt.Errorf("keepAlivedConfigDir is not a string: %v", v)
		}
		c.ConfPath = configDir
	}
	return nil
}

// getIsMasterHA safely gets the isMasterHa value from config
func getIsMasterHA(cfg map[string]interface{}) (bool, error) {
	v, ok := cfg["isMasterHa"]
	if !ok {
		return false, errors.New("isMasterHa not found in config")
	}

	isMasterHa, ok := v.(bool)
	if !ok {
		return false, fmt.Errorf("isMasterHa is not a boolean: %v", v)
	}
	return isMasterHa, nil
}

// renderCheckScript renders the check script based on HA type
func renderCheckScript(c *BKEHAComponent, cfg map[string]interface{}, isMasterHa bool) error {
	var checkTmpl []byte
	var scriptName string
	var templatePath string

	if isMasterHa {
		templatePath = tmplCheckMaster
		scriptName = scriptCheckMaster
	} else {
		templatePath = tmplCheckIngress
		scriptName = scriptCheckIngress
	}

	checkTmpl, err := fs.ReadFile(f, templatePath)
	if err != nil {
		return fmt.Errorf("failed to read check script template %s: %w", templatePath, err)
	}

	log.Infof(logRenderScript, scriptName)
	return renderHAAndStore(c, checkTmpl, cfg, pathForHAManifestScript(c, scriptName), nil)
}

// renderKeepalivedInstance renders the keepalived instance configuration
func renderKeepalivedInstance(c *BKEHAComponent, cfg map[string]interface{}, isMasterHa bool) error {
	if cfg == nil {
		return fmt.Errorf("config map is nil")
	}
	cfg["instances"] = []string{}

	var templateName, logMessage string
	if isMasterHa {
		templateName = "keepalived.master.conf.tmpl"
		logMessage = logRenderMasterVIP
	} else {
		templateName = "keepalived.ingress.conf.tmpl"
		logMessage = logRenderIngressVIP
	}

	log.Info(logMessage)

	t, err := template.New(templateName).Funcs(*keepalivedConfFuncMap()).ParseFS(f, "tmpl/keepalived/"+templateName)
	if err != nil || t == nil {
		return fmt.Errorf("failed to parse template %s: %w", templateName, err)
	}

	content := bytes.NewBuffer(nil)
	if err = t.Execute(content, cfg); err != nil {
		return fmt.Errorf("failed to execute template %s: %w", templateName, err)
	}

	// Safely append to instances
	instances, ok := cfg["instances"].([]string)
	if !ok {
		instances = []string{}
	}
	cfg["instances"] = append(instances, content.String())

	return nil
}

// getBaseTemplate gets or creates the base template
func getBaseTemplate(c *BKEHAComponent, cfg map[string]interface{}, isMasterHa bool) (*template.Template, error) {
	confPath := pathForHAManifestConf(c)

	if utils.Exists(confPath) {
		return getExistingBaseTemplate(confPath, isMasterHa)
	}

	return createNewBaseTemplate()
}

// getExistingBaseTemplate reads and modifies existing template
func getExistingBaseTemplate(confPath string, isMasterHa bool) (*template.Template, error) {
	file, err := os.OpenFile(confPath, os.O_RDWR|os.O_APPEND, utils.RwRR)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat config file: %w", err)
	}

	buffer := make([]byte, stat.Size())
	if _, err = file.Read(buffer); err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// Remove old instance based on HA type
	if isMasterHa {
		buffer = regexp.MustCompile(MasterKeepalivedInstanceReg).ReplaceAll(buffer, []byte(""))
	} else {
		buffer = regexp.MustCompile(IngressKeepalivedInstanceReg).ReplaceAll(buffer, []byte(""))
	}

	// Append base template
	buffer = append(buffer, []byte(keepalivedBaseTmpl)...)

	return template.New("keepalived.conf").Funcs(*keepalivedConfFuncMap()).Parse(string(buffer))
}

// createNewBaseTemplate creates a new base template from file
func createNewBaseTemplate() (*template.Template, error) {
	baseTmpl, err := template.New("keepalived.base.conf.tmpl").Funcs(*keepalivedConfFuncMap()).ParseFS(f, tmplKeepalivedBase)
	if err != nil {
		return nil, fmt.Errorf("failed to parse base template: %w", err)
	}

	if baseTmpl == nil {
		return nil, errors.New("template is nil")
	}

	return baseTmpl, nil
}

// renderConfigFile renders the main configuration file
func renderConfigFile(c *BKEHAComponent, cfg map[string]interface{}, baseTmpl *template.Template) error {
	log.Infof(logRenderConfFile)

	confPath := pathForHAManifestConf(c)
	writer, err := os.OpenFile(confPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, utils.RwRR)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer writer.Close()

	if err = baseTmpl.Execute(writer, cfg); err != nil {
		return fmt.Errorf("failed to execute base template: %w", err)
	}

	return nil
}

// renderYamlFile renders the YAML manifest file
func renderYamlFile(c *BKEHAComponent, cfg map[string]interface{}) error {
	tmpl, err := fs.ReadFile(f, tmplKeepalivedYaml)
	if err != nil {
		return fmt.Errorf("failed to read YAML template: %w", err)
	}

	funcMap := mergeFuncMap(keepalivedConfFuncMap(), utilFuncMap())
	return renderHAAndStore(c, tmpl, cfg, pathForHAManifest(c), funcMap)
}

func keepalivedConfFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"randomString": generateRandomString,
		"computeWeight": func(nodes []HANode) string {
			return strconv.Itoa(weightMultiplier * len(nodes))
		},
		"isMaster": KeepalivedInstanceIsMaster,
		"priority": func(nodes []HANode) string {
			ips, err := bkenet.GetAllInterfaceIP()
			if err != nil {
				return strconv.Itoa(defaultPriority)
			}
			for i, n := range nodes {
				// 跳过master
				if i == 0 {
					continue
				}
				for _, ip := range ips {
					if bkenet.AddressMatchesIP(ip, n.IP) {
						return strconv.Itoa(defaultPriority - priorityDecrementStep*i)
					}
				}
			}
			return strconv.Itoa(defaultPriority)
		},
	}
}

func KeepalivedInstanceIsMaster(nodes []HANode) bool {
	ips, err := bkenet.GetAllInterfaceIP()
	if err != nil {
		return false
	}
	for _, ip := range ips {
		if bkenet.AddressMatchesIP(ip, nodes[0].IP) {
			return true
		}
	}
	return false
}

func GlobalFuncMap() *template.FuncMap {
	base := &template.FuncMap{
		"imageRepo": func(cfg *BootScope) string {
			bkeCfg := bkeinit.BkeConfig(*cfg.BkeConfig)
			return bkeCfg.ImageFuyaoRepo()
		},
	}
	return mergeFuncMap(base, versionutil.K8sVersionFuncMap())
}

func utilFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"randomString": generateRandomString,
	}
}

func apiServerFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"advertiseAddress": func(cfg *BootScope) string {
			if cfg.CurrentNode.APIServer != nil && cfg.CurrentNode.APIServer.Host != "" {
				return cfg.CurrentNode.APIServer.Host
			}
			return cfg.CurrentNode.IP
		},
		"etcdServers": func(cfg *BootScope) string {
			nodesData, err := cluster.GetNodesData(cfg.ClusterNamespace, cfg.ClusterName)
			if err != nil {
				return etcd.GetClientURLByIP(cfg.HostIP)
			}
			bkeNodes := bkenode.Nodes(nodesData)
			etcdNodes := bkeNodes.Etcd()
			if etcdNodes.Length() == 0 {
				return etcd.GetClientURLByIP(cfg.HostIP)
			}
			etcdEndpoints := make([]string, 0, len(etcdNodes))
			for _, n := range etcdNodes {
				etcdEndpoints = append(etcdEndpoints, etcd.GetClientURLByIP(n.IP))
			}
			return strings.Join(etcdEndpoints, ",")
		},
		"apiServerPort": func(cfg *BootScope) int32 {
			if cfg.CurrentNode.APIServer != nil && cfg.CurrentNode.APIServer.Port != 0 {
				return cfg.CurrentNode.APIServer.Port
			}
			return cfg.BkeConfig.Cluster.APIServer.Port
		},
		"imageInfo": func(cfg *BootScope) string {
			k8sVersion := strings.TrimPrefix(cfg.BkeConfig.Cluster.KubernetesVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultAPIServerImageName, k8sVersion)
		},
		"clientCAFile": func(cfg *BootScope) string {
			return clientCAFilePath(cfg)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.APIServer != nil && cfg.CurrentNode.APIServer.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.APIServer.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.APIServer.ExtraArgs)
		},
		"upgradeWithOpenFuyao": isUpgradeWithOpenFuyao,
	}
}

func isUpgradeWithOpenFuyao(cfg *BootScope) bool {
	log.Info("detect upgradeWithOpenFuyao flag")

	if cfg == nil {
		log.Warn("boot scope is nil, skip boot-extra check and use file-based fallback")
	} else {
		// 1) 原有：BkeConfig.Addons 里声明了 openfuyao
		if v, ok := cfg.Extra["upgradeWithOpenFuyao"].(bool); ok {
			log.Infof("upgradeWithOpenFuyao from boot extra: %t", v)
			if v {
				return true
			}
		} else {
			log.Info("upgradeWithOpenFuyao bool not found in boot extra, fallback to file detection")
		}
	}

	// 2) 兜底：节点上已存在 webhook 配置文件，说明之前装过 openfuyao
	webhookCfgPath := "/etc/kubernetes/webhook/webhook-config.yaml"
	_, err := os.Stat(webhookCfgPath)
	if err == nil {
		log.Infof("detected openfuyao webhook config file %q", webhookCfgPath)
		return true
	}
	if !os.IsNotExist(err) {
		log.Warnf("stat webhook config file %q failed: %v", webhookCfgPath, err)
	}

	// 3) 兜底：现有 apiserver 静态 Pod 已经带了 webhook 参数
	manifest := filepath.Join(GetDefaultManifestsPath(), "kube-apiserver.yaml")
	if data, err := os.ReadFile(manifest); err == nil {
		hasWebhookArg := strings.Contains(string(data), "authentication-token-webhook-config-file")
		log.Infof("detected kube-apiserver manifest %q webhook arg present: %t", manifest, hasWebhookArg)
		if hasWebhookArg {
			return true
		}
	} else {
		log.Warnf("read kube-apiserver manifest %q failed: %v", manifest, err)
	}

	log.Info("upgradeWithOpenFuyao detection result: false")
	return false
}

// etcdImageTagFromBootScope resolves the etcd image tag for manifest templates.
// Declarative upgrade may inject target version via BootScope.Extra["etcdVersion"]
// or override BkeConfig.Cluster.EtcdVersion on the agent before rendering.
func etcdImageTagFromBootScope(cfg *BootScope) string {
	if cfg == nil {
		return strings.TrimPrefix(versions.EtcdImageTag(), "v")
	}
	var etcdVersion string
	if cfg.Extra != nil {
		if v, ok := cfg.Extra["etcdVersion"].(string); ok && v != "" {
			etcdVersion = v
		}
	}
	if etcdVersion == "" && cfg.BkeConfig != nil && cfg.BkeConfig.Cluster.EtcdVersion != "" {
		etcdVersion = cfg.BkeConfig.Cluster.EtcdVersion
	}
	if etcdVersion == "" {
		etcdVersion = versions.EtcdImageTag()
	}
	return strings.TrimPrefix(etcdVersion, "v")
}

func etcdFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"initialCluster": func(members []etcd.Member) string {
			if len(members) == 0 {
				ip, _ := bkenet.GetExternalIP()
				hostname := utils.HostName()
				return fmt.Sprintf("%s=https://%s:%d", hostname, ip, bkeetcd.EtcdListenPeerPort)
			}
			var result []string
			for _, member := range members {
				result = append(result, fmt.Sprintf("%s=%s", member.Name, member.PeerURL))
			}
			return strings.Join(result, ",")
		},
		"imageInfo": func(cfg *BootScope) string {
			etcdVersion := etcdImageTagFromBootScope(cfg)
			return fmt.Sprintf("%s:%s", bkeinit.DefaultEtcdImageName, etcdVersion)
		},
		"dataDir": func(cfg *BootScope) string {
			if cfg.CurrentNode.Etcd != nil && cfg.CurrentNode.Etcd.DataDir != "" {
				return cfg.CurrentNode.Etcd.DataDir
			}
			return cfg.BkeConfig.Cluster.Etcd.DataDir
		},
		"etcdAdvertiseUrls": func(cfg *BootScope) string {
			if v, ok := cfg.Extra["EtcdAdvertiseUrls"]; ok {
				return v.(string)
			}
			return etcd.GetClientURLByIP(cfg.HostIP)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.Etcd != nil && cfg.CurrentNode.Etcd.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.Etcd.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.Etcd.ExtraArgs)
		},
	}

}

func clientCAFilePath(cfg *BootScope) string {
	caChainPath := fmt.Sprintf("%s/%s", cfg.BkeConfig.Cluster.CertificatesDir, CertCAAndChainFileName)
	if utils.Exists(caChainPath) {
		return caChainPath
	}
	return fmt.Sprintf("%s/%s", cfg.BkeConfig.Cluster.CertificatesDir, pkiutil.CACertName)
}

func controllerFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"clientCAFile": func(cfg *BootScope) string {
			return clientCAFilePath(cfg)
		},
		"imageInfo": func(cfg *BootScope) string {
			k8sVersion := strings.TrimPrefix(cfg.BkeConfig.Cluster.KubernetesVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultControllerManagerImageName, k8sVersion)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.ControllerManager != nil && cfg.CurrentNode.ControllerManager.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.ControllerManager.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.ControllerManager.ExtraArgs)
		},
		"getSubnetMask": func(cidr string) string {
			res := strings.Split(cidr, "/")
			if len(res) != cidrPartsCount {
				return ""
			}
			mask, err := strconv.Atoi(res[cidrMaskIndex])
			if err != nil {
				return ""
			}
			if mask > subnetMaskThreshold {
				return res[cidrMaskIndex]
			}
			return strconv.Itoa(minSubnetMaskBits)
		},
	}
}

func schedulerFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"imageInfo": func(cfg *BootScope) string {
			k8sVersion := strings.TrimPrefix(cfg.BkeConfig.Cluster.KubernetesVersion, "v") // 去掉前缀字符 v
			return fmt.Sprintf("%s:%s", bkeinit.DefaultSchedulerImageName, k8sVersion)
		},
		"extraArgs": func(cfg *BootScope) []string {
			if cfg.CurrentNode.Scheduler != nil && cfg.CurrentNode.Scheduler.ExtraArgs != nil {
				return getExtraArgs(cfg.CurrentNode.Scheduler.ExtraArgs)
			}
			return getExtraArgs(cfg.BkeConfig.Cluster.Scheduler.ExtraArgs)
		},
	}
}

// mergeFuncMap merge two funcMap
func mergeFuncMap(f1, f2 *template.FuncMap) *template.FuncMap {
	f := *f1
	for k, v := range *f2 {
		f[k] = v
	}
	return &f
}

func getExtraArgs(argsMap map[string]string) []string {
	var args []string
	for k, v := range argsMap {
		args = append(args, fmt.Sprintf("%s=%s", k, v))
	}
	return args
}
