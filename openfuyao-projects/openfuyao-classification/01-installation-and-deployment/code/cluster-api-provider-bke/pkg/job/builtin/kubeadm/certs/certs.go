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

package certs

import (
	"context"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name = "Cert"
	// CertChainFileName defines certificate ca and chain name
	CertCAAndChainFileName = "ca-chain.crt"
	// LocalCertDir is the directory where certificates are pushed by ensureBKEAgent
	LocalCertDir = "/etc/openFuyao/certs"
	// LocalTrustChainPath is the local path for trust-chain.crt
	LocalTrustChainPath = LocalCertDir + "/trust-chain.crt"
	// LocalGlobalCACertPath is the local path for global-ca.crt
	LocalGlobalCACertPath = LocalCertDir + "/global-ca.crt"
	// LocalGlobalCAKeyPath is the local path for global-ca.key
	LocalGlobalCAKeyPath = LocalCertDir + "/global-ca.key"
	two                  = 2
)

type CertPlugin struct {
	k8sClient   client.Client
	bkeConfig   *bkev1beta1.BKEConfig
	exec        exec.Executor
	clusterName string
	namespace   string
	currentNode *bkenode.Node
	nodes       bkenode.Nodes

	altNames []string
	pkiPath  string
}

// kubeConfigServerConfig contains the server configuration for kubeconfig generation
type kubeConfigServerConfig struct {
	isWorker   bool
	serverPort int
	nodeIP     string
	user       string
}

func New(c client.Client, exec exec.Executor, cfg *bkev1beta1.BKEConfig) plugin.Plugin {
	return &CertPlugin{
		k8sClient: c,
		bkeConfig: cfg,
		exec:      exec,
	}
}

func (cp *CertPlugin) Name() string {
	return Name
}
func (cp *CertPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"generate":              {Key: "generate", Value: "true,false", Required: false, Default: "true", Description: "generate certificates and keys"},
		"altDNSNames":           {Key: "altDNSNames", Value: "abc.com", Required: false, Default: "", Description: "alternative DNS names for cert generation, split by ','"},
		"altIPs":                {Key: "altIPs", Value: "", Required: false, Default: "127.0.0.1", Description: "alternative IPs for cert generation, split by ','"},
		"generateKubeConfig":    {Key: "kubeConfig", Value: "", Required: false, Default: "false", Description: "generate kubeConfig"},
		"localKubeConfigScope":  {Key: "localKubeConfigScope", Value: "kubeconfig,controller-manager,scheduler,kubelet,kube-proxy", Required: false, Default: "kubeconfig,controller-manager,scheduler,kubelet,kube-proxy", Description: "Specify the scope to generate kubeConfig"},
		"loadCACert":            {Key: "loadCACert", Value: "true,false", Required: false, Default: "false", Description: "load CA cert from cluster api secret"},
		"caCertNames":           {Key: "caCertNames", Value: "Names1,Names2,...", Required: false, Default: "ca,sa,etcd,proxy", Description: "NameSpace/Names of CA cert from cluster api secret,Names optional [ca,sa,etcd,proxy]"},
		"loadTargetClusterCert": {Key: "loadTargetClusterCert", Value: "true,false", Required: false, Default: "false", Description: "load all cert from cluster api secret, which create by bke"},
		"loadAdminKubeconfig":   {Key: "loadAdminKubeconfig", Value: "true,false", Required: false, Default: "false", Description: "load admin kubeconfig from cluster api secret"},
		"uploadCerts":           {Key: "upload", Value: "true,false", Required: false, Default: "false", Description: "upload certs to manager k8s as secret,"},
		"certificatesDir":       {Key: "certificatesDir", Value: "", Required: false, Default: pkiutil.GetDefaultPkiPath(), Description: "Path to the PKI file,default is /etc/kubernetes/pki "},
		"clusterName":           {Key: "clusterName", Value: "", Required: false, Default: "", Description: "cluster name"},
		"namespace":             {Key: "namespace", Value: "", Required: false, Default: "", Description: "bke cluster namespace,which is used to load cert secret"},
		"tlsScope":              {Key: "tlsScope", Value: "", Required: false, Default: "", Description: "tls server or client"},
		"isManagerCluster":      {Key: "isManagerCluster", Value: "true,false", Required: false, Default: "false", Description: "whether the BKECluster is manager"},
	}
}

// Execute  implements the plugin.Plugin interface.
// example:
// 1. load CA cert from cluster api secret,and generate other certs with CA cert
// ["Cert", "generate=true","loadCACert=true","caCertNames=ca,sa,kubeconfig,etcd,proxy"]
// 2. generate all certificates required for k8s to run
// ["Cert", "generate=true"]
// 3. load CA cert from cluster api secret, but only loads one or more to generate other certs
// ["Cert", "generate=true","loadCACert=true","caCertNames=ca"]
func (cp *CertPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	cpc := *cp
	cpc.exec = exec.Bound(cp.exec, ctx)
	cp = &cpc

	certParamMap, err := plugin.ParseCommands(cp, commands)
	if err != nil {
		return nil, err
	}

	if err := cp.initializeParams(certParamMap); err != nil {
		return nil, err
	}

	// loadCACert has highest priority
	if err := cp.handleLoadCACert(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleLoadAdminKubeconfig(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleCertChainAndGlobalCert(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleLoadTargetClusterCert(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleGenerateCerts(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleGenerateKubeConfig(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleGenerateTLSCerts(certParamMap); err != nil {
		return nil, err
	}

	if err := cp.handleUploadCerts(certParamMap); err != nil {
		return nil, err
	}

	return nil, nil
}

// initializeParams initializes plugin parameters and current node from certParamMap
func (cp *CertPlugin) initializeParams(certParamMap map[string]string) error {
	cp.pkiPath = certParamMap["certificatesDir"]
	cp.namespace = certParamMap["namespace"]
	cp.clusterName = certParamMap["clusterName"]

	nodesData, err := plugin.GetNodesDataFromNs(cp.namespace, cp.clusterName)

	if err != nil {
		return nil
	}

	cp.nodes = bkenode.Nodes(nodesData)
	currentNode, err := cp.nodes.CurrentNode()
	if err == nil {
		cp.currentNode = &currentNode
	}

	return nil
}

// handleLoadCACert handles loading CA certificates from cluster API secret
func (cp *CertPlugin) handleLoadCACert(certParamMap map[string]string) error {
	if certParamMap["loadCACert"] != "true" {
		return nil
	}

	if certParamMap["caCertNames"] == "" {
		return errors.New("caCertNames is required when loadCACert is true")
	}

	caCertNames := strings.Split(certParamMap["caCertNames"], ",")
	if cp.clusterName == "" || cp.namespace == "" {
		return errors.New("clusterName and namespace are required when caCertNames is not empty")
	}

	log.Infof("load CA cert from cluster api secret")
	// get CA cert from cluster api secret
	for _, secretName := range caCertNames {
		if err := cp.getCertFromSecret(secretName, cp.pkiPath); err != nil {
			return errors.Wrapf(err, "failed to get CA cert from namespace %q secret %q", cp.namespace, secretName)
		}
	}

	return nil
}

// handleLoadAdminKubeconfig handles loading admin kubeconfig from cluster API secret
func (cp *CertPlugin) handleLoadAdminKubeconfig(certParamMap map[string]string) error {
	if certParamMap["loadAdminKubeconfig"] != "true" {
		return nil
	}

	log.Infof("load admin kubeconfig from cluster api secret")
	if cp.namespace == "" {
		return errors.New("'namespace' is required when 'loadAdminKubeconfig' is true")
	}
	// get kubeconfig from cluster api secret，该kubeconfig是集群的入口
	if err := cp.getCertFromSecret("kubeconfig", pkiutil.KubernetesDir); err != nil {
		return errors.Wrapf(err, "failed to get kubeConfig from namespace %q secret %q", cp.namespace, "kubeconfig")
	}

	if cp.currentNode != nil {
		cp.copyAdminKubeConfig(cp.currentNode.Username)
	} else {
		cp.copyAdminKubeConfig("root")
	}

	return nil
}

// getAdminKubeConfigServer parses the admin kubeconfig file to extract server IP and port
// Returns serverIP, serverPort, and error
func (cp *CertPlugin) getAdminKubeConfigServer() (string, int, error) {
	kubeConfigPath := pkiutil.GetDefaultKubeConfigPath()
	if !utils.Exists(kubeConfigPath) {
		return "", 0, errors.Errorf("admin kubeconfig not found at %s", kubeConfigPath)
	}

	config, err := clientcmd.LoadFromFile(kubeConfigPath)
	if err != nil {
		return "", 0, errors.Wrapf(err, "failed to load admin kubeconfig from %s", kubeConfigPath)
	}

	if config.CurrentContext == "" {
		return "", 0, errors.New("no current context in admin kubeconfig")
	}

	context, ok := config.Contexts[config.CurrentContext]
	if !ok {
		return "", 0, errors.Errorf("current context %s not found in kubeconfig", config.CurrentContext)
	}

	cluster, ok := config.Clusters[context.Cluster]
	if !ok {
		return "", 0, errors.Errorf("cluster %s not found in kubeconfig", context.Cluster)
	}

	if cluster.Server == "" {
		return "", 0, errors.New("server address not found in kubeconfig cluster")
	}

	// Parse server URL (format: https://IP:PORT or https://HOSTNAME:PORT)
	serverURL, err := url.Parse(cluster.Server)
	if err != nil {
		return "", 0, errors.Wrapf(err, "failed to parse server URL: %s", cluster.Server)
	}

	// Extract host and port
	host := serverURL.Hostname()
	portStr := serverURL.Port()

	// If no port in URL, use default HTTPS port
	if portStr == "" {
		portStr = "443"
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return "", 0, errors.Wrapf(err, "failed to parse port: %s", portStr)
	}

	// Check if host is an IP address or hostname
	ip := net.ParseIP(host)
	if ip != nil {
		log.Debugf("parsed admin kubeconfig server: %s:%d", ip.String(), port)
		return ip.String(), port, nil
	}

	// If it's a hostname, return it as-is
	log.Debugf("parsed admin kubeconfig server: %s:%d", host, port)
	return host, port, nil
}

// handleCertChainAndGlobalCert handles management cluster specific logic
func (cp *CertPlugin) handleCertChainAndGlobalCert(certParamMap map[string]string) error {
	if cp.currentNode.IsWorker() {
		return nil
	}
	if certParamMap["isManagerCluster"] != "true" {
		if err := cp.saveCertChainFromLocal(); err != nil {
			log.Warnf("Failed to save cert chain from local: %v", err)
			return nil
		}
		return nil
	}

	log.Infof("detected management cluster, loading global CA from local files")
	if err := cp.loadGlobalCACertFromLocal(); err != nil {
		// 这里不返回错误，如果用户不传入global-ca，走自签证书逻辑，则不需要这一步
		log.Warnf("No global CA from local files: %v", err)
	}

	if err := cp.saveCertChainFromLocal(); err != nil {
		log.Warnf("failed to save cert chain for manager cluster: %v", err)
		// 如果用户不导入证书链，不影响自定义证书签发功能，不返回错误
	}

	return nil
}

// handleLoadTargetClusterCert handles loading all certificates from cluster API secret
func (cp *CertPlugin) handleLoadTargetClusterCert(certParamMap map[string]string) error {
	if certParamMap["loadTargetClusterCert"] != "true" {
		return nil
	}

	log.Infof("load all cert from cluster api secret, which create by bke")
	if cp.namespace == "" {
		return errors.New("'namespace' is required when 'loadTargetClusterCert' is true")
	}
	certList := pkiutil.GetTargetClusterCertList()
	certList.SetPkiPath(cp.pkiPath)
	for _, cert := range certList {
		if err := cp.getCertFromSecret(cert.Name, cp.pkiPath); err != nil {
			return errors.Wrapf(err, "failed to get cert from namespace %q secret %q", cp.namespace, cert.Name)
		}
	}

	return nil
}

// handleGenerateCerts handles certificate generation
func (cp *CertPlugin) handleGenerateCerts(certParamMap map[string]string) error {
	if certParamMap["generate"] != "true" {
		return nil
	}

	log.Infof("generate certificates and keys,if ca.crt and ca.key exist,will generate other certs with ca.crt and ca.key")
	if cp.altNames == nil {
		cp.altNames = []string{}
	}
	if v, ok := certParamMap["altIPs"]; ok {
		altIPs := strings.Split(v, ",")
		cp.altNames = append(cp.altNames, altIPs...)
		log.Infof("add extra altIPs for generate certs: %v", altIPs)
	}
	if v, ok := certParamMap["altDNSNames"]; ok {
		altDNSNames := strings.Split(v, ",")
		cp.altNames = append(cp.altNames, altDNSNames...)
		log.Infof("add extra altDNSNames for generate certs: %v", altDNSNames)
	}

	if err := cp.generateCerts(); err != nil {
		return errors.Wrap(err, "failed to generate certs")
	}

	return nil
}

// handleGenerateKubeConfig handles kubeconfig generation
func (cp *CertPlugin) handleGenerateKubeConfig(certParamMap map[string]string) error {
	if certParamMap["generateKubeConfig"] != "true" {
		return nil
	}

	log.Debug("generate kubeConfig")

	// Determine node type and get server configuration
	config := cp.getKubeConfigServerConfig()
	log.Debugf("apiserver port is %d, nodeIP is %s", config.serverPort, config.nodeIP)

	// Determine scopes to generate
	scopes := strings.Split(certParamMap["localKubeConfigScope"], ",")

	// Generate kubeconfigs for each scope
	if err := cp.generateKubeConfigsForScopes(scopes, config.serverPort, config.nodeIP, config.isWorker); err != nil {
		return err
	}

	// Copy admin kubeconfig to user's home directory (only for master nodes)
	if !config.isWorker && utils.ContainsString(scopes, "kubeconfig") {
		cp.copyAdminKubeConfig(config.user)
	}

	return nil
}

func (cp *CertPlugin) handleGenerateTLSCerts(certParamMap map[string]string) error {
	log.Debug("generate kubeConfig")

	nodeIP := "127.0.0.1"
	if cp.currentNode != nil {
		if cp.currentNode.IP != "" {
			nodeIP = cp.currentNode.IP
		}
	}
	log.Debugf("nodeIP is %s", nodeIP)

	// Determine scopes to generate
	tlsScopes := strings.Split(certParamMap["tlsScope"], ",")

	// Generate tls cert for each scope
	if err := cp.generateTLSCertsForScopes(tlsScopes, nodeIP); err != nil {
		return err
	}
	return nil
}

// getKubeConfigServerConfig determines the server configuration based on node type
func (cp *CertPlugin) getKubeConfigServerConfig() kubeConfigServerConfig {
	config := kubeConfigServerConfig{
		serverPort: bkeinit.DefaultAPIBindPort,
		nodeIP:     "127.0.0.1",
		user:       "root",
		isWorker:   false,
	}

	if cp.currentNode == nil {
		return config
	}

	if cp.currentNode.IsWorker() {
		config.isWorker = true
		log.Infof("current node is worker node, will only generate kubelet kubeconfig")
		config.serverPort, config.nodeIP, _ = cp.getNodeServerConfig(true)

	} else {
		config.serverPort, config.nodeIP, config.user = cp.getNodeServerConfig(false)
	}

	return config
}

// getNodeServerConfig gets the server configuration for nodes
// isWorker: true for worker nodes, false for master nodes
// Returns serverPort, nodeIP, and user (user is only meaningful for master nodes)
func (cp *CertPlugin) getNodeServerConfig(isWorker bool) (serverPort int, nodeIP string, user string) {
	serverPort = bkeinit.DefaultAPIBindPort
	nodeIP = "127.0.0.1"
	user = "root"

	if cp.bkeConfig == nil {
		log.Warn("bkeConfig is nil, using default serverPort and nodeIP")
		return serverPort, nodeIP, user
	}
	if cp.bkeConfig.Cluster.APIServer != nil && cp.bkeConfig.Cluster.APIServer.Port != 0 {
		serverPort = int(cp.bkeConfig.Cluster.APIServer.Port)
	}

	if isWorker {
		if clusterutil.LoadBalanceType(cp.bkeConfig) == "nginx" {
			// nginx mode: worker-local components reach the apiserver via the local nginx static pod
			log.Infof("worker node: loadBalanceType is nginx, using local nginx proxy 127.0.0.1 with port %d", serverPort)
			return serverPort, "127.0.0.1", user
		}
		// For worker nodes, always use server IP and port from admin kubeconfig
		adminServerIP, adminServerPort, err := cp.getAdminKubeConfigServer()
		if err == nil && adminServerIP != "" && adminServerPort > 0 {
			nodeIP = adminServerIP
			serverPort = adminServerPort
			log.Infof("worker node: using server %s with port %d from admin kubeconfig", nodeIP, serverPort)
			return serverPort, nodeIP, user
		}
		// Fallback: if admin kubeconfig not available, use master node IP
		log.Debugf("failed to get server from admin kubeconfig: %v", err)

		nodesData, err := plugin.GetNodesDataFromNs(cp.namespace, cp.clusterName)
		if err != nil {
			return serverPort, nodeIP, user
		}
		nodes := bkenode.Nodes(nodesData)
		masterNodes := nodes.Master()
		if len(masterNodes) > 0 {
			nodeIP = masterNodes[0].IP
			log.Infof("worker node: using master node IP %s with port %d (fallback, admin kubeconfig not available)", nodeIP, serverPort)
			return serverPort, nodeIP, user
		}
		log.Warnf("worker node: admin kubeconfig not available and no master nodes found, using default %s:%d", nodeIP, serverPort)
		return serverPort, nodeIP, user
	}

	if cp.currentNode != nil {
		if cp.currentNode.APIServer != nil && cp.currentNode.APIServer.Port != 0 {
			serverPort = int(cp.currentNode.APIServer.Port)
		}
		if cp.currentNode.IP != "" {
			nodeIP = cp.currentNode.IP
		}
		if cp.currentNode.Username != "" {
			user = cp.currentNode.Username
		}
	}

	return serverPort, nodeIP, user
}

// generateKubeConfigsForScopes generates kubeconfigs for the given scopes
func (cp *CertPlugin) generateKubeConfigsForScopes(scopes []string, serverPort int, nodeIP string, isWorker bool) error {
	for _, scope := range scopes {
		// 该方法创建的kubeconfig文件，只能用于当前节点，固定存储在/etc/kubernetes目录下
		// For worker node's kubelet kubeconfig, nodeIP is already set correctly:
		// - HA cluster: nodeIP is MasterHADomain (master.bocloud.com)
		// - Single master: nodeIP is the master node IP
		kubeConfigGenerater := pkiutil.NewKubeConfigGenerater(pkiutil.KubeConfigOptions{
			PkiPath:     cp.pkiPath,
			ClusterName: cp.clusterName,
			FileName:    scope,
			ServerPort:  strconv.Itoa(serverPort),
			HostIP:      nodeIP,
			BKEConfig:   cp.bkeConfig,
			Nodes:       cp.nodes,
		})
		if err := kubeConfigGenerater.Generate(); err != nil {
			return fmt.Errorf("failed to generate local kubeconfig for %s, err: %w", scope, err)
		}
	}
	return nil
}

// generateTLSCertsForScopes generates TLS certificates for the given scopes using CA certificate
// This function generates client certificates and keys for each scope, similar to generateKubeConfigsForScopes
// but only generates TLS certificates without creating kubeconfig files
func (cp *CertPlugin) generateTLSCertsForScopes(scopes []string, nodeIP string) error {
	for _, scope := range scopes {
		// Find BKECert spec for this scope
		kubeConfigs := pkiutil.GetTlsConfigs()
		var certSpec *pkiutil.BKECert
		for _, kubeConfigSpec := range kubeConfigs {
			if kubeConfigSpec.Name == scope {
				certSpec = kubeConfigSpec
				break
			}
		}
		if certSpec == nil {
			log.Warnf("not found BKE kubeconfig spec for %s, skip TLS cert generation", scope)
			continue
		}

		// Set PKI path, /etc/kubernetes
		certSpec.PkiPath = utils.KubernetesDir

		// Initialize AltNames if not already initialized
		if certSpec.Config.AltNames.DNSNames == nil {
			certSpec.Config.AltNames.DNSNames = []string{}
		}
		if certSpec.Config.AltNames.IPs == nil {
			certSpec.Config.AltNames.IPs = []net.IP{}
		}

		sanList := []string{
			"127.0.0.1",
			"0.0.0.0",
			"localhost",
		}
		if nodeIP != "" {
			sanList = append(sanList, nodeIP)
		}

		// Append SANs to AltNames using AppendSANsToAltNames which handles validation and deduplication
		if err := pkiutil.AppendSANsToAltNames(&certSpec.Config.AltNames, sanList, certSpec.BaseName); err != nil {
			return fmt.Errorf("failed to append SANs to TLS cert for %s: %w", scope, err)
		}

		// Load CA certificate and key
		caCertSpec := pkiutil.BKECertRootCA()
		caCertSpec.PkiPath = cp.pkiPath
		if err := pkiutil.CertExists(caCertSpec); err != nil {
			return fmt.Errorf("CA certificate not found for generating TLS cert for %s: %w", scope, err)
		}

		// Generate TLS certificate using CA certificate
		// This will generate client certificate and key signed by the CA with SAN fields
		if err := pkiutil.GenerateCertWithCA(certSpec, caCertSpec); err != nil {
			return fmt.Errorf("failed to generate TLS cert for %s, err: %w", scope, err)
		}

		sanLog := "IP:127.0.0.1,IP:0.0.0.0,DNS:localhost"
		if nodeIP != "" {
			sanLog += ",IP:" + nodeIP
		}
		log.Infof("successfully generated TLS cert for %s with SAN: %s", scope, sanLog)
	}
	return nil
}

// handleUploadCerts handles uploading certificates to manager k8s as secret
func (cp *CertPlugin) handleUploadCerts(certParamMap map[string]string) error {
	if certParamMap["uploadCerts"] != "true" {
		return nil
	}

	log.Infof("upload all certs to manager k8s as secret")
	if cp.namespace == "" {
		return errors.New("'namespace' is required when 'uploadCerts' is true")
	}
	if err := cp.uploadCerts(cp.namespace); err != nil {
		return errors.Wrap(err, "failed to upload certs")
	}

	return nil
}

// generateCerts generates all the certificates and keys necessary to run k8s
// If the CA certificate exists, use the existing CA certificate to generate other certificates
func (cp *CertPlugin) generateCerts() error {
	certList, err := cp.prepareCertList()
	if err != nil {
		return err
	}
	certList.SetPkiPath(cp.pkiPath)
	var lastCACert *pkiutil.BKECert
	for _, cert := range certList {
		if cert.CAName == "" {
			if err := pkiutil.GenerateCACert(cert); err != nil {
				return err
			}
			lastCACert = cert
		} else {
			if err := pkiutil.GenerateCertWithCA(cert, lastCACert); err != nil {
				return err
			}
		}
	}

	if err := pkiutil.GenerateRSACert(pkiutil.BKECertServiceAccount()); err != nil {
		return err
	}

	return nil
}

// uploadCerts uploads all the certificates and keys to the manager k8s, expect the CA cert
func (cp *CertPlugin) uploadCerts(nameSpace string) error {
	errInfo := "failed to upload cert"
	certList := pkiutil.GetCertsWithoutCA()
	certList.SetPkiPath(cp.pkiPath)
	for _, cert := range certList {
		if err := pkiutil.CertExists(cert); err != nil {
			return errors.Wrapf(err, "%s %q", errInfo, cert.Name)
		}
	}
	for _, cert := range certList {
		if err := cp.uploadCertToSecret(cert, nameSpace); err != nil {
			return errors.Wrapf(err, "%s %q", errInfo, cert.Name)
		}
	}
	return nil
}

func (cp *CertPlugin) getCertFromSecret(name, saveTo string) error {
	certSecret := &corev1.Secret{}

	certName := fmt.Sprintf("%s-%s", cp.clusterName, name)

	log.Debugf("get cert %q from secret %s", name, cp.namespace+"/"+certName)

	err := cp.k8sClient.Get(context.Background(), client.ObjectKey{Namespace: cp.namespace, Name: certName}, certSecret)
	if err != nil {
		return err
	}

	if err := pkiutil.StoreClusterAPICert(certSecret, saveTo); err != nil {
		return err
	}
	return nil
}

func (cp *CertPlugin) uploadCertToSecret(certSpec *pkiutil.BKECert, namespace string) error {
	return pkiutil.UploadBKECertToClusterAPI(cp.k8sClient, certSpec, namespace, cp.clusterName)
}

// prepareCertList returns a pkiutil.Certificates object
// fill each certificate the AltNames fields from the kubeadm config and node.json
func (cp *CertPlugin) prepareCertList() (pkiutil.Certificates, error) {
	certList := pkiutil.GetDefaultCertList()

	if cp.bkeConfig == nil {
		return certList, nil
	}

	for _, cert := range certList {
		switch cert.BaseName {
		case pkiutil.APIServerCertAndKeyBaseName:
			altNames, err := pkiutil.GetAPIServerCertAltNamesWithNodes(cp.bkeConfig, cp.nodes)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to get alt names from bke config")
			}
			cert.Config.AltNames.DNSNames = append(cert.Config.AltNames.DNSNames, altNames.DNSNames...)
			cert.Config.AltNames.IPs = append(cert.Config.AltNames.IPs, altNames.IPs...)
			if err := pkiutil.AppendSANsToAltNames(&cert.Config.AltNames, cp.altNames, cert.BaseName); err != nil {
				return nil, errors.Wrapf(err, "failed to append alt names to %q", cert.BaseName)
			}
		case pkiutil.EtcdServerCertAndKeyBaseName:
			altnames, err := pkiutil.GetEtcdCertAltNamesWithNodes(cp.bkeConfig, cp.nodes, true)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to get alt names from bke config")
			}
			cert.Config.AltNames.DNSNames = append(cert.Config.AltNames.DNSNames, altnames.DNSNames...)
			cert.Config.AltNames.IPs = append(cert.Config.AltNames.IPs, altnames.IPs...)
			if err := pkiutil.AppendSANsToAltNames(&cert.Config.AltNames, cp.altNames, cert.BaseName); err != nil {
				return nil, errors.Wrapf(err, "failed to append alt names to %q", cert.BaseName)
			}
		case pkiutil.EtcdPeerCertAndKeyBaseName:
			altnames, err := pkiutil.GetEtcdCertAltNamesWithNodes(cp.bkeConfig, cp.nodes, true)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to get alt names from bke config")
			}
			cert.Config.AltNames.DNSNames = append(cert.Config.AltNames.DNSNames, altnames.DNSNames...)
			cert.Config.AltNames.IPs = append(cert.Config.AltNames.IPs, altnames.IPs...)
			if err := pkiutil.AppendSANsToAltNames(&cert.Config.AltNames, cp.altNames, cert.BaseName); err != nil {
				return nil, errors.Wrapf(err, "failed to append alt names to %q", cert.BaseName)
			}
		default:
			// Unknown certificate type, skip alt names processing
			log.Debugf("Skipping alt names processing for unknown certificate type: %s", cert.BaseName)
		}
	}

	return certList, nil
}

// splitNameSpaceName split name space and name
func splitNameSpaceName(nn string) (string, []string, error) {
	ns := strings.Split(nn, ":")
	if len(ns) != two {
		return "", nil, errors.New("invalid namespace:name format")
	}
	return ns[0], strings.Split(ns[1], ","), nil
}

// copyAdminKubeConfig copy admin kube config to user's home dir and root's home dir
func (cp *CertPlugin) copyAdminKubeConfig(user string) {
	// ignore error
	rootDir := "/root/.kube"
	configDir := "/root/.kube"

	if user != "root" {
		configDir = fmt.Sprintf("/home/%s/.kube", user)
	}

	cmd := fmt.Sprintf("cp -f /etc/kubernetes/admin.conf %s/config", configDir)

	if configDir == rootDir && !utils.Exists(configDir) {
		err := os.MkdirAll(configDir, utils.RwxRxRx)
		if err != nil {
			log.Warnf("(ignore) failed to mkdir %s, err: %v", rootDir, err)
		}
	} else {
		err := os.MkdirAll(configDir, utils.RwxRxRx)
		if err != nil {
			log.Warnf("(ignore) failed to mkdir %s, err: %v", configDir, err)
		}
		err = os.MkdirAll(rootDir, utils.RwxRxRx)
		if err != nil {
			log.Warnf("(ignore) failed to mkdir %s, err: %v", rootDir, err)
		}
		cmd = cmd + fmt.Sprintf(" && cp -f /etc/kubernetes/admin.conf %s/config", rootDir)
		cmd = cmd + fmt.Sprintf(" && chown -R %s:%s %s", user, user, configDir)
	}

	// todo 合并到一个文件中并设置kubectl context
	out, err := cp.exec.ExecuteCommandWithOutput("/bin/sh", "-c", cmd)
	if err != nil {
		log.Warnf("(ignore) failed to cp /etc/kubernetes/admin.conf, output: %s, error: %v", out, err)
	}
}

// loadGlobalCACertFromLocal loads the global CA from local files and saves it to the local filesystem
// It also saves the cert chain (ca-chain.crt) and global-ca.crt/global-ca.key
func (cp *CertPlugin) loadGlobalCACertFromLocal() error {
	// 从本地读取 global-ca.crt 和 global-ca.key
	certBytes, err := os.ReadFile(LocalGlobalCACertPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Infof("global-ca.crt not found at %s, skipping global CA loading", LocalGlobalCACertPath)
			return nil
		}
		return errors.Wrapf(err, "failed to read global-ca.crt from %s", LocalGlobalCACertPath)
	}

	keyBytes, err := os.ReadFile(LocalGlobalCAKeyPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Infof("global-ca.key not found at %s, skipping global CA loading", LocalGlobalCAKeyPath)
			return nil
		}
		return errors.Wrapf(err, "failed to read global-ca.key from %s", LocalGlobalCAKeyPath)
	}

	if len(certBytes) == 0 || len(keyBytes) == 0 {
		log.Infof("global-ca.crt or global-ca.key is empty")
		return nil
	}

	cert, key, err := cp.parseGlobalCACertAndKey(certBytes, keyBytes)
	if err != nil {
		return err
	}

	// 保存 global-ca.crt 和 global-ca.key 到 pki 目录
	if err := cp.saveGlobalCACertAndKey(cert, key); err != nil {
		return err
	}

	return nil
}

// validateGlobalCASecretData validates whether the global CA Secret data is valid
func (cp *CertPlugin) validateGlobalCASecretData(secret *corev1.Secret) ([]byte, []byte, error) {
	secretNamespace := utils.GlobalCANamespace
	secretName := utils.GlobalCASecretName

	if secret.Data == nil {
		log.Warnf("global CA secret %s/%s has no data", secretNamespace, secretName)
		return nil, nil, nil
	}

	certBytes, hasCert := secret.Data[pkiutil.TLSCrtDataName]
	keyBytes, hasKey := secret.Data[pkiutil.TLSKeyDataName]

	if !hasCert || !hasKey {
		log.Warnf("global CA secret %s/%s missing certificate or key data", secretNamespace, secretName)
		return nil, nil, nil
	}

	if len(certBytes) == 0 || len(keyBytes) == 0 {
		log.Warnf("global CA secret %s/%s has empty data", secretNamespace, secretName)
		return nil, nil, nil
	}

	return certBytes, keyBytes, nil
}

// parseGlobalCACertAndKey parses the global CA certificate and key
func (cp *CertPlugin) parseGlobalCACertAndKey(certBytes, keyBytes []byte) (*x509.Certificate, *rsa.PrivateKey, error) {
	certs, err := pkiutil.ParseCertsPEM(certBytes)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to parse global CA certificate")
	}

	key, err := pkiutil.ParsePrivateKeyPEM(keyBytes)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to parse global CA key")
	}

	return certs[0], key, nil
}

// saveGlobalCACertAndKey saves the global CA certificate and key to the local filesystem
func (cp *CertPlugin) saveGlobalCACertAndKey(cert *x509.Certificate, key *rsa.PrivateKey) error {
	globalCACert := &pkiutil.BKECert{
		Name:     "global-ca",
		BaseName: pkiutil.GlobalCACertAndKeyBaseName,
		IsCA:     true,
		Config:   pkiutil.CertConfig{},
		PkiPath:  cp.pkiPath,
	}

	if err := pkiutil.WriteCertAndKey(globalCACert, cert, key); err != nil {
		return errors.Wrap(err, "failed to write global CA certificate and key")
	}
	return nil
}

// saveCertChain saves the certificate chain to local filesystem. If the chain exists in the Secret,
// it gets the CA cert from cluster CA secret and merges them together
func (cp *CertPlugin) saveCertChain(secret *corev1.Secret) {
	if secret.Data == nil {
		log.Infof("global CA secret %s/%s has no data for cert chain", secret.Namespace, secret.Name)
		return
	}

	chainBytes, hasChain := secret.Data[pkiutil.ChainCrtDataName]
	if !hasChain || len(chainBytes) == 0 {
		log.Infof("global CA secret %s/%s missing chain data", secret.Namespace, secret.Name)
		return
	}

	caCertBytes, err := cp.getCACertFromClusterSecret()
	if err != nil {
		log.Warnf("failed to get CA cert from cluster secret: %v", err)
		return
	}

	chainCerts, err := cp.parseChainCerts(chainBytes)
	if err != nil {
		log.Warnf("failed to parse certificate chain: %v", err)
		return
	}

	allCerts, err := cp.mergeCertChain(caCertBytes, chainBytes)
	if err != nil {
		log.Warnf("failed to merge certificate chain: %v", err)
		return
	}

	chainPath := filepath.Join(cp.pkiPath, pkiutil.CertChainFileName)
	caChainPath := filepath.Join(cp.pkiPath, CertCAAndChainFileName)

	if err := cp.writeCertChainToFile(chainPath, chainCerts); err != nil {
		log.Warnf("failed to write chain only file: %v", err)
	}

	if err := cp.writeCertChainToFile(caChainPath, allCerts); err != nil {
		log.Warnf("failed to write CA and chain file: %v", err)
		return
	}
}

// saveCertChainFromLocal saves the certificate chain from local filesystem
// It reads trust-chain.crt from local file and gets CA cert from cluster secret, then merges them
// If trust-chain.crt doesn't exist, it returns without saving anything (ca.crt is only for serving chain.crt)
// If CA cert cannot be obtained, it will only save chain (if available)
func (cp *CertPlugin) saveCertChainFromLocal() error {
	// 从本地读取 trust-chain.crt（如果存在）
	chainBytes, err := os.ReadFile(LocalTrustChainPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Infof("trust-chain.crt not found at %s, skipping cert chain generation (ca.crt is only for serving chain.crt)", LocalTrustChainPath)
			// 如果没有 trust-chain，不需要保存 CA cert，因为 ca.crt 只是为了服务 chain.crt
			return nil
		}
		return errors.Wrapf(err, "failed to read trust-chain.crt from %s", LocalTrustChainPath)
	}

	if len(chainBytes) == 0 {
		log.Infof("trust-chain.crt at %s is empty, skipping cert chain generation (ca.crt is only for serving chain.crt)", LocalTrustChainPath)
		// 如果 trust-chain 为空，不需要保存 CA cert
		return nil
	}

	// 从 secret 获取 ca.crt
	caCertBytes, err := cp.getCACertFromClusterSecret()
	if err != nil {
		log.Warnf("failed to get CA cert from cluster secret: %v, will only save chain", err)
		// 如果无法获取 CA cert，只保存 chain
		chainCerts, err := cp.parseChainCerts(chainBytes)
		if err != nil {
			return errors.Wrap(err, "failed to parse certificate chain")
		}
		chainPath := filepath.Join(cp.pkiPath, pkiutil.CertChainFileName)
		return cp.writeCertChainToFile(chainPath, chainCerts)
	}

	chainCerts, err := cp.parseChainCerts(chainBytes)
	if err != nil {
		return errors.Wrap(err, "failed to parse certificate chain")
	}

	allCerts, err := cp.mergeCertChain(caCertBytes, chainBytes)
	if err != nil {
		return errors.Wrap(err, "failed to merge certificate chain")
	}

	chainPath := filepath.Join(cp.pkiPath, pkiutil.CertChainFileName)
	caChainPath := filepath.Join(cp.pkiPath, CertCAAndChainFileName)

	if err := cp.writeCertChainToFile(chainPath, chainCerts); err != nil {
		// 证书链写入失败不影响后续ca-chain.crt写入并挂载apiserver
		log.Warnf("failed to write chain only file: %v", err)
	}

	if err := cp.writeCertChainToFile(caChainPath, allCerts); err != nil {
		return errors.Wrap(err, "failed to write CA and chain file")
	}

	log.Infof("successfully saved cert chain from local files")
	return nil
}

// getCACertFromClusterSecret gets the CA certificate from the cluster CA secret
func (cp *CertPlugin) getCACertFromClusterSecret() ([]byte, error) {
	if cp.clusterName == "" || cp.namespace == "" {
		return nil, errors.New("clusterName and namespace are required to get CA cert")
	}

	caSecretName := fmt.Sprintf("%s-%s", cp.clusterName, pkiutil.CACertAndKeyBaseName)
	caSecret := &corev1.Secret{}

	log.Debugf("get CA cert from secret %s/%s", cp.namespace, caSecretName)

	err := cp.k8sClient.Get(context.Background(), client.ObjectKey{Namespace: cp.namespace, Name: caSecretName}, caSecret)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get CA secret %s/%s", cp.namespace, caSecretName)
	}

	if caSecret.Data == nil {
		return nil, errors.Errorf("CA secret %s/%s has no data", cp.namespace, caSecretName)
	}

	caCertBytes, hasCert := caSecret.Data[pkiutil.TLSCrtDataName]
	if !hasCert || len(caCertBytes) == 0 {
		return nil, errors.Errorf("CA secret %s/%s missing tls.crt data", cp.namespace, caSecretName)
	}

	return caCertBytes, nil
}

// parseChainCerts parses the certificate chain byte array into a certificate list
func (cp *CertPlugin) parseChainCerts(chainBytes []byte) ([]*x509.Certificate, error) {
	chainCerts, err := pkiutil.ParseCertsPEM(chainBytes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse certificate chain")
	}
	return chainCerts, nil
}

// mergeCertChain merges the CA certificate and certificate chain
func (cp *CertPlugin) mergeCertChain(caCertBytes, chainBytes []byte) ([]*x509.Certificate, error) {
	caCerts, err := pkiutil.ParseCertsPEM(caCertBytes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse CA certificate")
	}

	chainCerts, err := pkiutil.ParseCertsPEM(chainBytes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse certificate chain")
	}

	allCerts := append(caCerts, chainCerts...)
	return allCerts, nil
}

// writeCertChainToFile writes the certificate chain to the specified file path
func (cp *CertPlugin) writeCertChainToFile(filePath string, certs []*x509.Certificate) error {
	if !utils.Exists(cp.pkiPath) {
		if err := os.MkdirAll(cp.pkiPath, utils.RwxRxRx); err != nil {
			return errors.Wrapf(err, "failed to create directory %s", cp.pkiPath)
		}
	}

	chainPEM := cp.encodeCertsToPEM(certs)
	if err := os.WriteFile(filePath, chainPEM, utils.RwRR); err != nil {
		return errors.Wrapf(err, "failed to write certificate chain to %s", filePath)
	}
	log.Infof("saved certificate chain to %s", filePath)
	return nil
}

// encodeCertsToPEM encodes the certificate list to PEM format
func (cp *CertPlugin) encodeCertsToPEM(certs []*x509.Certificate) []byte {
	if len(certs) == 0 {
		return nil
	}
	chainPEM := pkiutil.EncodeCertToPEM(certs[0])
	for i := 1; i < len(certs); i++ {
		chainPEM = append(chainPEM, pkiutil.EncodeCertToPEM(certs[i])...)
	}
	return chainPEM
}
