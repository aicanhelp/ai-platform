/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package config

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	yaml2 "sigs.k8s.io/yaml"

	"gopkg.openfuyao.cn/bkeadm/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type Options struct {
	root.Options
	Directory       string   `json:"directory"`
	File            string   `json:"file"`
	Args            []string `json:"args"`
	Product         string   `json:"product"`
	Domain          string   `json:"domain"`
	ImageRepoPort   string   `json:"imageRepoPort"`
	NtpServer       string   `json:"ntpServer"`
	AgentHealthPort string   `json:"agentHealthPort"`
}

// GenerateControllerParam generate containerd needed params
func GenerateControllerParam(domain string) (string, string) {
	sandbox := fmt.Sprintf("%s/%s:%s", utils.DefaultThirdMirror, configinit.DefaultPauseImageName, versions.PauseImageTag())
	offline := "false"
	if global.CustomExtra["otherRepo"] != "" {
		repoPrefixList := strings.Split(global.CustomExtra["otherRepo"], "/")
		repoPrefix := strings.Join(repoPrefixList[:len(repoPrefixList)-1], "/")
		sandbox = fmt.Sprintf("%s/%s:%s", repoPrefix, configinit.DefaultPauseImageName, versions.PauseImageTag())
	}
	if global.CustomExtra["otherRepo"] == "" && global.CustomExtra["onlineImage"] == "" {
		offline = "true"
	}
	if offline == "true" {
		sandbox = fmt.Sprintf("%s/%s/%s:%s", domain, configinit.ImageRegistryKubernetes, configinit.DefaultPauseImageName, versions.PauseImageTag())
	}
	return sandbox, offline
}

func (op *Options) Config(customExtra map[string]string, imageRepo, yumRepo, chartRepo confv1beta1.Repo, ntpServer string) {
	if err := op.ensureDirectory(); err != nil {
		return
	}

	bkeCluster := op.createBKECluster()
	cfg := op.initBaseConfig(customExtra, imageRepo, yumRepo, chartRepo, ntpServer)

	op.optimizeKubeClient(cfg)
	op.applyBocCustomConfig(cfg, yumRepo)

	op.generateConfigFiles(cfg, bkeCluster)

	log.Infof("Generate the bkecluster configuration file directory %s", op.Directory)
}

// ensureDirectory 确保目录存在
func (op *Options) ensureDirectory() error {
	if utils.Exists(op.Directory) {
		return nil
	}

	if err := os.MkdirAll(op.Directory, utils.DefaultDirPermission); err != nil {
		log.Errorf("Unable to create directory %v", err)
		return err
	}
	return nil
}

// createBKECluster 创建基础的 BKECluster 对象
func (op *Options) createBKECluster() configv1beta1.BKECluster {
	return configv1beta1.BKECluster{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "bke.bocloud.com/v1beta1",
			Kind:       "BKECluster",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bke-cluster",
			Namespace: "bke-cluster",
		},
		Spec: configv1beta1.BKEClusterSpec{
			Pause:         false,
			DryRun:        false,
			Reset:         false,
			ClusterConfig: nil,
			KubeletConfigRef: &confv1beta1.KubeletConfigRef{
				Name:      "bke-kubelet",
				Namespace: "bke-kubelet",
			},
		},
	}
}

// initBaseConfig 初始化基础配置
func (op *Options) initBaseConfig(customExtra map[string]string, imageRepo, yumRepo, chartRepo confv1beta1.Repo,
	ntpServer string) *configv1beta1.BKEConfig {
	cfg, _ := configinit.ConvertBkEConfig(configinit.GetDefaultBKEConfig())
	op.applyCustomConfig(cfg, customConfigOptions{
		customExtra: customExtra,
		imageRepo:   imageRepo,
		yumRepo:     yumRepo,
		chartRepo:   chartRepo,
		ntpServer:   ntpServer,
	})

	return cfg
}

type customConfigOptions struct {
	customExtra map[string]string
	imageRepo   confv1beta1.Repo
	yumRepo     confv1beta1.Repo
	chartRepo   confv1beta1.Repo
	ntpServer   string
}

// applyCustomConfig 应用自定义配置
func (op *Options) applyCustomConfig(cfg *configv1beta1.BKEConfig, custom customConfigOptions) {
	if len(custom.customExtra) > 0 {
		cfg.CustomExtra = custom.customExtra
	}

	if len(custom.imageRepo.Domain) > 0 {
		cfg.Cluster.ImageRepo = custom.imageRepo
	}

	if len(custom.yumRepo.Domain) > 0 {
		cfg.Cluster.HTTPRepo = custom.yumRepo
	}

	if len(custom.chartRepo.Domain) > 0 || len(custom.chartRepo.Ip) > 0 {
		cfg.Cluster.ChartRepo = custom.chartRepo
	}

	if len(custom.ntpServer) > 0 {
		cfg.Cluster.NTPServer = custom.ntpServer
	}
	cfg.Cluster.AgentHealthPort = op.AgentHealthPort
}

// optimizeKubeClient 优化 Kubernetes 客户端配置
func (op *Options) optimizeKubeClient(cfg *configv1beta1.BKEConfig) {
	cfg.Cluster.APIServer = &confv1beta1.APIServer{
		ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
			ExtraArgs: map[string]string{
				"max-mutating-requests-inflight": "3000",
				"max-requests-inflight":          "1000",
				"watch-cache-sizes":              "node#1000,pod#5000",
			},
		},
	}

	cfg.Cluster.ControllerManager = &confv1beta1.ControlPlaneComponent{
		ExtraArgs: map[string]string{
			"kube-api-qps":   "1000",
			"kube-api-burst": "1000",
		},
	}

	cfg.Cluster.Scheduler = &confv1beta1.ControlPlaneComponent{
		ExtraArgs: map[string]string{
			"kube-api-qps": "1000",
		},
	}

	cfg.Cluster.Kubelet = &confv1beta1.Kubelet{
		ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
			ExtraArgs: map[string]string{
				"kube-api-qps":   "1000",
				"kube-api-burst": "2000",
			},
			ExtraVolumes: []confv1beta1.HostPathMount{
				{
					Name:     "kubelet-root-dir",
					HostPath: "/var/lib/kubelet",
				},
			},
		},
	}
}

// applyBocCustomConfig 应用初始自定义配置
func (op *Options) applyBocCustomConfig(cfg *configv1beta1.BKEConfig, yumRepo confv1beta1.Repo) {
	cfg.Cluster.KubernetesVersion = versions.KubernetesVersion()
	cfg.Cluster.EtcdVersion = versions.EtcdVersion()
	cfg.Cluster.ContainerdVersion = versions.ContainerdVersion()
	cfg.Cluster.OpenFuyaoVersion = versions.OpenFuyaoVersion()
	cfg.Cluster.ContainerdConfigRef = &configv1beta1.ContainerdConfigRef{
		Namespace: "bke-containerd",
		Name:      "bke-containerd",
	}
	cfg.Cluster.ContainerRuntime = confv1beta1.ContainerRuntime{
		CRI:     "containerd",
		Runtime: "runc",
		Param: map[string]string{
			"data-root": "/var/lib/containerd",
		},
	}

	cfg = op.product(cfg, yumRepo)
}

// generateConfigFiles 生成配置文件
func (op *Options) generateConfigFiles(cfg *configv1beta1.BKEConfig, bkeCluster configv1beta1.BKECluster) {
	master1(op.Directory, bkeCluster, cfg)
	master1node1(op.Directory, bkeCluster, cfg)
	master3(op.Directory, bkeCluster, cfg)
}

// writeClusterAndNodesConfig writes BKECluster and BKENodes to separate files
func writeClusterAndNodesConfig(directory, filenamePrefix string, cluster configv1beta1.BKECluster, nodes []confv1beta1.BKENode, singleNode bool) {
	clusterFile := path.Join(directory, filenamePrefix+"-cluster.yaml")
	b, err := yaml2.Marshal(cluster)
	if err != nil {
		log.Errorf("Marshal cluster config %s failed: %v", clusterFile, err)
		return
	}
	if err := os.WriteFile(clusterFile, b, utils.DefaultFilePermission); err != nil {
		log.Errorf("Failed to write cluster config: %v", err)
		return
	}

	nodesSuffix := "-nodes.yaml"
	if singleNode {
		nodesSuffix = "-node.yaml"
	}
	nodesFile := path.Join(directory, filenamePrefix+nodesSuffix)
	var nodesData []byte
	for i, node := range nodes {
		nodeBytes, err := yaml2.Marshal(node)
		if err != nil {
			log.Errorf("Marshal node config for %s failed: %v", nodesFile, err)
			return
		}
		if i > 0 {
			nodesData = append(nodesData, []byte("---\n")...)
		}
		nodesData = append(nodesData, nodeBytes...)
	}
	if err := os.WriteFile(nodesFile, nodesData, utils.DefaultFilePermission); err != nil {
		log.Errorf("Failed to write nodes config: %v", err)
		return
	}
}

// createBKENodes creates BKENode resources from node definitions
func createBKENodes(clusterName, namespace string, nodeDefs []nodeDef) []confv1beta1.BKENode {
	var nodes []confv1beta1.BKENode
	for _, def := range nodeDefs {
		node := confv1beta1.BKENode{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "bke.bocloud.com/v1beta1",
				Kind:       "BKENode",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%s-%s", clusterName, def.Hostname),
				Namespace: namespace,
				Labels: map[string]string{
					"cluster.x-k8s.io/cluster-name": clusterName,
				},
			},
			Spec: confv1beta1.BKENodeSpec{
				Hostname: def.Hostname,
				IP:       def.IP,
				Username: def.Username,
				Password: def.Password,
				Port:     def.Port,
				Role:     def.Role,
			},
		}
		nodes = append(nodes, node)
	}
	return nodes
}

// nodeDef is a helper struct for node definition
type nodeDef struct {
	Hostname string
	IP       string
	Username string
	Password string
	Port     string
	Role     []string
}

// newMasterEtcdNodeDef creates a nodeDef with master/node and etcd roles using default port 22.
func newMasterEtcdNodeDef(hostname, ip, username, password string) nodeDef {
	return nodeDef{
		Hostname: hostname,
		IP:       ip,
		Username: username,
		Password: password,
		Port:     "22",
		Role: []string{
			"master/node",
			"etcd",
		},
	}
}

func (op *Options) product(cfg *confv1beta1.BKEConfig, yumRepo confv1beta1.Repo) *configv1beta1.BKEConfig {
	sandbox, offline := op.generateControllerParams()

	op.setBaseAddons(cfg)
	op.applyProductSpecificConfig(cfg, sandbox, offline)

	return cfg
}

// generateControllerParams 生成控制器参数
func (op *Options) generateControllerParams() (string, string) {
	return GenerateControllerParam(fmt.Sprintf("%s:%s", op.Domain, op.ImageRepoPort))
}

// setBaseAddons 设置基础插件配置
func (op *Options) setBaseAddons(cfg *confv1beta1.BKEConfig) {
	cfg.Addons = []confv1beta1.Product{
		{
			Name:    "kubeproxy",
			Version: versions.KubernetesVersion(),
			Param: map[string]string{
				"clusterNetworkMode": "calico",
			},
		},
		{
			Block:   true,
			Name:    "calico",
			Version: "v3.31.3",
			Param: map[string]string{
				"calicoMode":            "vxlan",
				"ipAutoDetectionMethod": "skip-interface=nerdctl*",
			},
		},
		{
			Name:    "coredns",
			Version: "v1.12.2-of.1",
		},
	}
}

// updateCorednsAntiAffinityByCount 根据节点数量更新 CoreDNS 反亲和性配置
func updateCorednsAntiAffinityByCount(cfg *confv1beta1.BKEConfig, nodeCount int) {
	enableAntiAffinity := "false"
	if nodeCount > 1 {
		enableAntiAffinity = "true"
	}

	for i := range cfg.Addons {
		if cfg.Addons[i].Name == "coredns" {
			if cfg.Addons[i].Param == nil {
				cfg.Addons[i].Param = make(map[string]string)
			}
			cfg.Addons[i].Param["EnableAntiAffinity"] = enableAntiAffinity
			break
		}
	}
}

// applyProductSpecificConfig 应用产品特定配置
func (op *Options) applyProductSpecificConfig(cfg *confv1beta1.BKEConfig, sandbox, offline string) {
	switch op.Product {
	case "fuyao-portal":
		op.applyFuyaoPortalConfig(cfg, sandbox, offline)
	case "fuyao-business":
		// fuyao-business 只需要设置 Kubernetes 版本，已在上面设置
	case "fuyao-allinone":
		op.applyFuyaoAllInOneConfig(cfg, sandbox, offline)
	default:
		op.logUnsupportedProduct()
	}
}

// applyFuyaoPortalConfig 应用扶摇门户配置
func (op *Options) applyFuyaoPortalConfig(cfg *confv1beta1.BKEConfig, sandbox, offline string) {
	op.applyFuyaoCommonConfig(cfg, sandbox, offline)
}

// applyFuyaoAllInOneConfig 应用扶摇全量配置
func (op *Options) applyFuyaoAllInOneConfig(cfg *confv1beta1.BKEConfig, sandbox, offline string) {
	op.applyFuyaoCommonConfig(cfg, sandbox, offline)
}

// applyFuyaoCommonConfig 应用扶摇通用配置
func (op *Options) applyFuyaoCommonConfig(cfg *confv1beta1.BKEConfig, sandbox, offline string) {
	clusterAPIAddon := op.createClusterAPIAddon(sandbox, offline)
	systemControllerAddon := op.createSystemControllerAddon()

	cfg.Addons = append(cfg.Addons, clusterAPIAddon, systemControllerAddon)
}

// createClusterAPIAddon 创建集群 API 插件
func (op *Options) createClusterAPIAddon(sandbox, offline string) confv1beta1.Product {
	ntpServer := op.NtpServer
	if ntpServer == "" {
		ntpServer = configinit.DefaultNTPServer
	}
	healthPort := op.AgentHealthPort
	if healthPort == "" {
		healthPort = utils.DefaultAgentHealthPort
	}
	return confv1beta1.Product{
		Name:    "cluster-api",
		Version: "v1.4.3",
		Block:   true,
		Param: map[string]string{
			"manage":            "true",
			"offline":           offline,
			"sandbox":           sandbox,
			"replicas":          "1",
			"containerdVersion": versions.ContainerdVersion(),
			"openFuyaoVersion":  versions.OpenFuyaoVersion(),
			"manifestsVersion":  "latest",
			"providerVersion":   "latest",
			"ntpServer":         ntpServer,
			"healthPort":        healthPort,
		},
	}
}

// createSystemControllerAddon 创建系统控制器插件
func (op *Options) createSystemControllerAddon() confv1beta1.Product {
	var helmRepo string
	if global.CustomExtra["otherRepo"] == "" && global.CustomExtra["onlineImage"] == "" {
		helmRepo = fmt.Sprintf("http://%s:%s", global.CustomExtra["host"], global.CustomExtra["chartRepoPort"])
	} else if global.CustomExtra["otherChart"] != "" {
		hostPart := strings.Split(global.CustomExtra["otherChart"], "/")[0]
		parts := strings.SplitN(hostPart, ":", 2)
		port := "443"
		if len(parts) > 1 {
			port = parts[1]
		}
		scheme := "https"
		if port != "443" {
			scheme = "http"
		}
		helmRepo = fmt.Sprintf("%s://%s:%s", scheme, parts[0], port)
	} else {
		helmRepo = "https://helm.openfuyao.cn/_core"
	}
	return confv1beta1.Product{
		Name:    "openfuyao-system-controller",
		Version: "latest",
		Param: map[string]string{
			"helmRepo":   helmRepo,
			"tagVersion": "latest",
		},
	}
}

// logUnsupportedProduct 记录不支持的产品日志
func (op *Options) logUnsupportedProduct() {
	log.Warnf("The product %s is not supported", op.Product)
}

// EncryptFile 加密文件
func (op *Options) EncryptFile() error {
	return op.processPasswordFile(true)
}

// DecryptFile 解密文件
func (op *Options) DecryptFile() error {
	return op.processPasswordFile(false)
}

func (op *Options) processPasswordFile(isEncrypt bool) error {
	// 加密/解密 BKECluster 文件（保留原逻辑以防需要）
	conf, err := op.loadClusterConfig()
	if err != nil {
		return err
	}

	// BKECluster 本身不再包含节点信息，直接保存
	return op.saveProcessedConfig(conf, isEncrypt)
}

func (op *Options) loadClusterConfig() (*configv1beta1.BKECluster, error) {
	conf := &configv1beta1.BKECluster{}
	yamlFile, err := os.ReadFile(op.File)
	if err != nil {
		return nil, err
	}

	err = yaml2.Unmarshal(yamlFile, conf)
	if err != nil {
		return nil, err
	}

	if conf.Spec.ClusterConfig == nil {
		return nil, errors.New("cluster configuration is empty. ")
	}

	return conf, nil
}

// EncryptNodesFile 加密 BKENode 文件中的密码
func (op *Options) EncryptNodesFile(nodesFile string) error {
	return op.processNodesFile(nodesFile, true)
}

// DecryptNodesFile 解密 BKENode 文件中的密码
func (op *Options) DecryptNodesFile(nodesFile string) error {
	return op.processNodesFile(nodesFile, false)
}

func (op *Options) processNodesFile(nodesFile string, isEncrypt bool) error {
	nodes, err := op.loadBKENodes(nodesFile)
	if err != nil {
		return err
	}

	for i := range nodes {
		if isEncrypt {
			nodes[i] = op.encryptBKENodePassword(nodes[i])
		} else {
			nodes[i] = op.decryptBKENodePassword(nodes[i])
		}
	}

	return op.saveBKENodes(nodesFile, nodes, isEncrypt)
}

func (op *Options) loadBKENodes(nodesFile string) ([]confv1beta1.BKENode, error) {
	yamlFile, err := os.ReadFile(nodesFile)
	if err != nil {
		return nil, err
	}

	var nodes []confv1beta1.BKENode
	docs := strings.Split(string(yamlFile), "---")
	for _, doc := range docs {
		doc = strings.TrimSpace(doc)
		if doc == "" {
			continue
		}
		var node confv1beta1.BKENode
		if err := yaml2.Unmarshal([]byte(doc), &node); err != nil {
			return nil, err
		}
		if node.Kind == "BKENode" {
			nodes = append(nodes, node)
		}
	}
	return nodes, nil
}

func (op *Options) encryptBKENodePassword(node confv1beta1.BKENode) confv1beta1.BKENode {
	_, err := security.AesDecrypt(node.Spec.Password)
	if err == nil {
		// 已经加密，跳过
		return node
	}

	encryptedPassword, err := security.AesEncrypt(node.Spec.Password)
	if err != nil {
		return node
	}

	node.Spec.Password = encryptedPassword
	return node
}

func (op *Options) decryptBKENodePassword(node confv1beta1.BKENode) confv1beta1.BKENode {
	decryptedPassword, err := security.AesDecrypt(node.Spec.Password)
	if err != nil {
		// 未加密或解密失败，跳过
		return node
	}

	node.Spec.Password = decryptedPassword
	return node
}

func (op *Options) saveBKENodes(originalFile string, nodes []confv1beta1.BKENode, isEncrypt bool) error {
	var nodesData []byte
	for i, node := range nodes {
		nodeBytes, err := yaml2.Marshal(node)
		if err != nil {
			return err
		}
		if i > 0 {
			nodesData = append(nodesData, []byte("---\n")...)
		}
		nodesData = append(nodesData, nodeBytes...)
	}

	pwd, err := os.Getwd()
	if err != nil {
		log.Warnf("Failed to get current working directory: %v", err)
	}

	action := "encrypt"
	if !isEncrypt {
		action = "decrypt"
	}

	baseName := filepath.Base(originalFile)
	ext := filepath.Ext(baseName)
	nameWithoutExt := strings.TrimSuffix(baseName, ext)
	outputFile := filepath.Join(pwd, fmt.Sprintf("%s-%s%s", nameWithoutExt, action, ext))

	err = os.WriteFile(outputFile, nodesData, utils.DefaultFilePermission)
	if err != nil {
		return err
	}

	log.Infof("%s is complete, output file: %s",
		strings.Title(action), filepath.Base(outputFile))
	return nil
}

func (op *Options) saveProcessedConfig(conf *configv1beta1.BKECluster, isEncrypt bool) error {
	by, err := yaml2.Marshal(conf)
	if err != nil {
		return err
	}

	pwd, err := os.Getwd()
	if err != nil {
		log.Warnf("Failed to get current working directory: %v", err)
	}
	action := "encrypt"
	if !isEncrypt {
		action = "decrypt"
	}

	bkefile := filepath.Join(pwd, fmt.Sprintf("%s-%s.yaml", conf.Name, action))
	err = os.WriteFile(bkefile, by, utils.DefaultFilePermission)
	if err != nil {
		return err
	}

	log.Infof("%s is complete, %s file in the current directory",
		strings.Title(action), conf.Name+"-"+action+".yaml")
	return nil
}

func (op *Options) EncryptString() error {
	for _, arg := range op.Args {
		p, err := security.AesEncrypt(arg)
		if err != nil {
			return err
		}
		fmt.Println(p)
	}
	return nil
}

func (op *Options) DecryptString() error {
	for _, arg := range op.Args {
		p, err := security.AesDecrypt(arg)
		if err != nil {
			return err
		}
		fmt.Println(p)
	}
	return nil
}

func writeClusterConfig(directory, filename string, cluster configv1beta1.BKECluster, cfg *confv1beta1.BKEConfig) {
	cluster.Spec.ClusterConfig = cfg
	b, err := yaml2.Marshal(cluster)
	if err != nil {
		log.Errorf("Marshal cluster config %s failed: %v", filename, err)
		return
	}

	err = os.WriteFile(path.Join(directory, filename), b, utils.DefaultFilePermission)
	if err != nil {
		log.Errorf("Failed to write the configuration file. Procedure %v", err)
		return
	}
}

func master1(directory string, cluster configv1beta1.BKECluster, cfg *confv1beta1.BKEConfig) {
	cfg1 := cfg.DeepCopy()

	// Create BKENodes for 1 master
	nodeDefs := []nodeDef{
		newMasterEtcdNodeDef("m1", "127.0.0.2", "root", "******"),
	}
	nodes := createBKENodes(cluster.Name, cluster.Namespace, nodeDefs)
	// Update CoreDNS anti-affinity based on node count
	updateCorednsAntiAffinityByCount(cfg1, len(nodes))

	cluster.Spec.ClusterConfig = cfg1
	writeClusterAndNodesConfig(directory, "1master", cluster, nodes, true)
}

func master1node1(directory string, cluster configv1beta1.BKECluster, cfg *confv1beta1.BKEConfig) {
	cfg1 := cfg.DeepCopy()

	nodeDefs := []nodeDef{
		newMasterEtcdNodeDef("m1", "127.0.0.1", "root", "******"),
		{
			Hostname: "n1",
			IP:       "127.0.0.2",
			Username: "root",
			Password: "******",
			Port:     "22",
			Role: []string{
				"node",
			},
		},
	}
	nodes := createBKENodes(cluster.Name, cluster.Namespace, nodeDefs)

	updateCorednsAntiAffinityByCount(cfg1, len(nodes))

	cluster.Spec.ClusterConfig = cfg1
	writeClusterAndNodesConfig(directory, "1master1node", cluster, nodes, false)
}

func master3(directory string, cluster configv1beta1.BKECluster, cfg *confv1beta1.BKEConfig) {
	cfg1 := cfg.DeepCopy()

	nodeDefs := []nodeDef{
		newMasterEtcdNodeDef("master-1", "127.0.0.1", "root", "******"),
		newMasterEtcdNodeDef("master-2", "127.0.0.2", "root", "******"),
		newMasterEtcdNodeDef("master-3", "127.0.0.3", "root", "******"),
	}
	// Create BKENodes for 3 masters
	nodes := createBKENodes(cluster.Name, cluster.Namespace, nodeDefs)

	updateCorednsAntiAffinityByCount(cfg1, len(nodes))

	cluster.Spec.ControlPlaneEndpoint = confv1beta1.APIEndpoint{
		Host: "0.0.0.0(vip)",
		Port: 36443,
	}

	cluster.Spec.ClusterConfig = cfg1
	writeClusterAndNodesConfig(directory, "3master", cluster, nodes, false)
}
