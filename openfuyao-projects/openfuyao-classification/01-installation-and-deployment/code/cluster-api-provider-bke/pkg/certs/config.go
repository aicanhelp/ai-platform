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

// Package certs provides functionality for loading and managing certificate configurations
// from Kubernetes ConfigMaps and local files, and applying them to certificate generation.
package certs

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	configDir = "/etc/openFuyao/certs/cert_config"
	// defaultCertValidity represents the default certificate validity period (1 year)
	defaultCertValidity = 365 * 24 * time.Hour
)

// CertConfigData holds all certificate configuration data from ConfigMap
type CertConfigData struct {
	ClusterCAPolicy pkiutil.CertPolicy `json:"cluster-ca-policy"`
	ClusterCACSR    pkiutil.CertCSR    `json:"cluster-ca-csr"`
	SignPolicy      pkiutil.CertPolicy `json:"sign-policy"`

	APIServerCSR              pkiutil.CertCSR `json:"apiserver-csr"`
	APIServerEtcdClientCSR    pkiutil.CertCSR `json:"apiserver-etcd-client-csr"`
	FrontProxyClientCSR       pkiutil.CertCSR `json:"front-proxy-client-csr"`
	APIServerKubeletClientCSR pkiutil.CertCSR `json:"apiserver-kubelet-client-csr"`

	FrontProxyCACSR pkiutil.CertCSR `json:"front-proxy-ca-csr"`

	EtcdCACSR                pkiutil.CertCSR `json:"etcd-ca-csr"`
	EtcdServerCSR            pkiutil.CertCSR `json:"etcd-server-csr"`
	EtcdHealthcheckClientCSR pkiutil.CertCSR `json:"etcd-healthcheck-client-csr"`
	EtcdPeerCSR              pkiutil.CertCSR `json:"etcd-peer-csr"`

	// KubeletCSR                pkiutil.CertCSR    `json:"kubelet-csr"`
	// KubeConfig CSR templates
	AdminKubeConfigCSR   pkiutil.CertCSR `json:"admin-kubeconfig-csr"`
	KubeletKubeConfigCSR pkiutil.CertCSR `json:"kubelet-kubeconfig-csr"`
	ControllerManagerCSR pkiutil.CertCSR `json:"controller-manager-csr"`
	SchedulerCSR         pkiutil.CertCSR `json:"scheduler-csr"`

	// AvailableKeys tracks which configurations are available in the ConfigMap
	AvailableKeys map[string]bool
}

// CertConfigLoader handles loading certificate configuration from ConfigMap
type CertConfigLoader struct {
	client     client.Client
	ctx        context.Context
	log        *log.Logger
	bkeCluster *bkev1beta1.BKECluster
}

// NewCertConfigLoader creates a new certificate configuration loader
func NewCertConfigLoader(
	ctx context.Context, client client.Client,
	bkeCluster *bkev1beta1.BKECluster, l *log.Logger,
) *CertConfigLoader {
	return &CertConfigLoader{
		client:     client,
		ctx:        ctx,
		log:        l,
		bkeCluster: bkeCluster,
	}
}

// LoadConfigMapData loads certificate configuration from ConfigMap
func (l *CertConfigLoader) LoadConfigMapData() (*CertConfigData, error) {
	configMap, err := l.getCertConfigMap()
	if err != nil {
		// If ConfigMap doesn't exist, return empty config with all keys marked as unavailable
		l.log.Warnf("ConfigMap not found, will use default certificate logic")
		return &CertConfigData{AvailableKeys: make(map[string]bool)}, nil
	}

	return l.parseConfigMapData(configMap)
}

// getCertConfigMap retrieves the certificate configuration ConfigMap
func (l *CertConfigLoader) getCertConfigMap() (*corev1.ConfigMap, error) {
	configMap := &corev1.ConfigMap{}
	err := l.client.Get(l.ctx, types.NamespacedName{
		Namespace: CertConfigMapNamespace,
		Name:      CertConfigMapName,
	}, configMap)

	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, errors.Errorf("certificate configuration ConfigMap %s/%s not found",
				CertConfigMapNamespace, CertConfigMapName)
		}
		return nil, fmt.Errorf("failed to get ConfigMap %s/%s: %w", CertConfigMapNamespace, CertConfigMapName, err)
	}

	return configMap, nil
}

// parseConfigMapData parses JSON data from ConfigMap into CertConfigData
func (l *CertConfigLoader) parseConfigMapData(configMap *corev1.ConfigMap) (*CertConfigData, error) {
	configData := &CertConfigData{
		AvailableKeys: make(map[string]bool),
	}

	// Define mapping between ConfigMap keys and target fields
	configMappings := map[string]interface{}{
		ConfigKeyClusterCAPolicy: &configData.ClusterCAPolicy,
		ConfigKeyClusterCACSR:    &configData.ClusterCACSR,
		ConfigKeySignPolicy:      &configData.SignPolicy,

		ConfigKeyAPIServerCSR:              &configData.APIServerCSR,
		ConfigKeyAPIServerEtcdClientCSR:    &configData.APIServerEtcdClientCSR,
		ConfigKeyFrontProxyClientCSR:       &configData.FrontProxyClientCSR,
		ConfigKeyAPIServerKubeletClientCSR: &configData.APIServerKubeletClientCSR,

		ConfigKeyFrontProxyCACSR: &configData.FrontProxyCACSR,

		ConfigKeyEtcdCACSR:                &configData.EtcdCACSR,
		ConfigKeyEtcdServerCSR:            &configData.EtcdServerCSR,
		ConfigKeyEtcdHealthcheckClientCSR: &configData.EtcdHealthcheckClientCSR,
		ConfigKeyEtcdPeerCSR:              &configData.EtcdPeerCSR,

		// ConfigKeyKubeletCSR: &configData.KubeletCSR,
		// KubeConfig CSR templates
		ConfigKeyAdminKubeConfigCSR:   &configData.AdminKubeConfigCSR,
		ConfigKeyKubeletKubeConfigCSR: &configData.KubeletKubeConfigCSR,
		ConfigKeyControllerManagerCSR: &configData.ControllerManagerCSR,
		ConfigKeySchedulerCSR:         &configData.SchedulerCSR,
	}

	// Parse all configurations using a loop
	for key, target := range configMappings {
		if err := l.parseJSONFromConfigMap(configMap, key, target); err != nil {
			l.log.Warnf("Failed to parse key %s from ConfigMap: %v", key, err)
			configData.AvailableKeys[key] = false
		} else {
			configData.AvailableKeys[key] = true
		}
	}

	return configData, nil
}

// parseJSONFromConfigMap parses a specific JSON field from ConfigMap
func (l *CertConfigLoader) parseJSONFromConfigMap(configMap *corev1.ConfigMap, key string, target interface{}) error {
	data, exists := configMap.Data[key]
	if !exists {
		return errors.Errorf("key %s not found in ConfigMap", key)
	}

	return json.Unmarshal([]byte(data), target)
}

// ApplyConfigToCerts applies the loaded configuration to certificates
func (l *CertConfigLoader) ApplyConfigToCerts(certs pkiutil.Certificates, configData *CertConfigData, clusterName string) error {
	for i, cert := range certs {
		skip, err := l.applyConfigToCert(certs[i], configData, clusterName)
		if err != nil {
			return errors.Wrapf(err, "failed to apply config to cert %s", cert.Name)
		}
		if skip {
			l.log.Infof("Skipping cert %s (%s) - no configuration found in ConfigMap, will use default logic", cert.Name, cert.BaseName)
		}
	}

	return nil
}

// applyConfigToCert applies configuration to a specific certificate
// Returns (skip, error), where skip=true means the config is not available
func (l *CertConfigLoader) applyConfigToCert(cert *pkiutil.BKECert, configData *CertConfigData, clusterName string) (bool, error) {
	// Apply CSR template based on certificate type
	csr, skip, err := l.getCSRForCert(cert, configData)
	if err != nil {
		return false, err
	}
	if skip {
		return true, nil
	}

	// Process template variables in hosts
	processedHosts, err := l.processTemplateHosts(csr.Hosts, clusterName)
	if err != nil {
		return false, err
	}

	// Apply CSR configuration to certificate
	l.applyCSRToCert(cert, csr, processedHosts, configData, cert.BaseName)

	return false, nil
}

// getCSRForCert returns the appropriate CSR template for the certificate
// Returns (csr, skip, error), where skip=true means config is not available
func (l *CertConfigLoader) getCSRForCert(cert *pkiutil.BKECert, configData *CertConfigData) (*pkiutil.CertCSR, bool, error) {
	// Define mapping between certificate types and their corresponding CSR fields and ConfigMap keys
	certTypeToCSR := map[string]struct {
		csr *pkiutil.CertCSR
		key string
	}{
		// 集群根CA证书
		pkiutil.CACertAndKeyBaseName: {&configData.ClusterCACSR, ConfigKeyClusterCACSR},

		// APIServer服务端证书，作为client访问Etcd，访问FrontProxy，访问kubelet的证书
		pkiutil.APIServerCertAndKeyBaseName:              {&configData.APIServerCSR, ConfigKeyAPIServerCSR},
		pkiutil.APIServerEtcdClientCertAndKeyBaseName:    {&configData.APIServerEtcdClientCSR, ConfigKeyAPIServerEtcdClientCSR},
		pkiutil.FrontProxyClientCertAndKeyBaseName:       {&configData.FrontProxyClientCSR, ConfigKeyFrontProxyClientCSR},
		pkiutil.APIServerKubeletClientCertAndKeyBaseName: {&configData.APIServerKubeletClientCSR, ConfigKeyAPIServerKubeletClientCSR},

		// 4个KubeConfig
		// Admin KubeConfig
		pkiutil.AdminKubeConfigFileName: {&configData.AdminKubeConfigCSR, ConfigKeyAdminKubeConfigCSR},
		// Kubelet 作为client访问APIServer的 KubeConfig
		pkiutil.KubeletKubeConfigFileName: {&configData.KubeletKubeConfigCSR, ConfigKeyKubeletKubeConfigCSR},
		// ControllerManager 作为client访问APIServer的 KubeConfig
		pkiutil.ControllerManagerKubeConfigFileName: {&configData.ControllerManagerCSR, ConfigKeyControllerManagerCSR},
		// Scheduler 作为client访问APIServer的 KubeConfig
		pkiutil.SchedulerKubeConfigFileName: {&configData.SchedulerCSR, ConfigKeySchedulerCSR},

		// FrontProxy的根CA证书，它只需要签发client证书给APIServer使用，kube-apiserver 内置的 front-proxy（Aggregator）不需要单独的服务端证书
		pkiutil.FrontProxyCACertAndKeyBaseName: {&configData.FrontProxyCACSR, ConfigKeyFrontProxyCACSR},

		// Etcd 根CA证书
		pkiutil.EtcdCACertAndKeyBaseName: {&configData.EtcdCACSR, ConfigKeyEtcdCACSR},
		// Etcd 挂载作为服务端的server证书，还有peer证书和healthcheck证书
		pkiutil.EtcdServerCertAndKeyBaseName:            {&configData.EtcdServerCSR, ConfigKeyEtcdServerCSR},
		pkiutil.EtcdHealthcheckClientCertAndKeyBaseName: {&configData.EtcdHealthcheckClientCSR, ConfigKeyEtcdHealthcheckClientCSR},
		pkiutil.EtcdPeerCertAndKeyBaseName:              {&configData.EtcdPeerCSR, ConfigKeyEtcdPeerCSR},
	}

	// Look up the CSR template for the certificate type
	if mapping, exists := certTypeToCSR[cert.BaseName]; exists {
		// Check if configuration exists in ConfigMap
		if available, ok := configData.AvailableKeys[mapping.key]; ok && available {
			return mapping.csr, false, nil
		}
		// Configuration not available, skip this cert
		return mapping.csr, true, nil
	}

	return nil, false, errors.Errorf("unknown certificate type: %s", cert.BaseName)
}

// processTemplateHosts processes template variables in host list
func (l *CertConfigLoader) processTemplateHosts(hosts []string, clusterName string) ([]string, error) {
	processedHosts := make([]string, 0, len(hosts))

	for _, host := range hosts {
		processedHost, err := l.processTemplateString(host, clusterName)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to process template host: %s", host)
		}
		processedHosts = append(processedHosts, processedHost)
	}

	return processedHosts, nil
}

// processTemplateString processes template variables in a string
func (l *CertConfigLoader) processTemplateString(templateStr string, clusterName string) (string, error) {
	tmpl, err := template.New("cert").Parse(templateStr)
	if err != nil {
		return "", err
	}

	data := l.getTemplateData(clusterName)
	var result strings.Builder
	if err := tmpl.Execute(&result, data); err != nil {
		return "", err
	}

	return result.String(), nil
}

// getTemplateData returns template data for certificate generation
func (l *CertConfigLoader) getTemplateData(clusterName string) map[string]interface{} {
	data := map[string]interface{}{
		"ClusterName": clusterName,
	}

	if l.bkeCluster != nil && l.bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		data["AdvertiseAddress"] = l.bkeCluster.Spec.ControlPlaneEndpoint.Host
	}

	return data
}

// applyCSRToCert applies CSR configuration to certificate
func (l *CertConfigLoader) applyCSRToCert(cert *pkiutil.BKECert, csr *pkiutil.CertCSR, hosts []string, configData *CertConfigData, certBaseName string) {
	// Apply basic CSR fields
	cert.Config.Config.CommonName = csr.CN
	cert.Config.Config.Organization = []string{csr.O}
	cert.Config.Country = []string{csr.C}
	cert.Config.Province = []string{csr.ST}
	cert.Config.Locality = []string{csr.L}
	cert.Config.OrganizationalUnit = []string{csr.OU}

	// Apply hosts to AltNames with deduplication
	l.applyHostsToAltNames(cert, hosts)

	// Apply extended certificate configuration fields from sign-policy or cluster-ca-policy
	l.applyExtendedConfig(cert, csr, configData, certBaseName)
}

// applyExtendedConfig applies extended certificate configuration fields from sign-policy or cluster-ca-policy
func (l *CertConfigLoader) applyExtendedConfig(cert *pkiutil.BKECert, csr *pkiutil.CertCSR, configData *CertConfigData, certBaseName string) {
	// Apply key configuration
	if csr.Key.Size > 0 {
		cert.Config.KeySize = csr.Key.Size
	}
	if csr.Key.Algo != "" {
		cert.Config.PublicKeyAlgorithm = pkiutil.ParsePublicKeyAlgorithm(csr.Key.Algo)
	}

	// Get policy based on certificate type
	policy := l.getPolicyForCert(certBaseName, configData)

	// Apply policy configuration (usages, expiry, CA constraints)
	l.applyPolicyToCert(cert, policy, certBaseName)
}

// getPolicyForCert returns the appropriate policy (ClusterCA or Sign) for the certificate
func (l *CertConfigLoader) getPolicyForCert(certBaseName string, configData *CertConfigData) *pkiutil.CertPolicy {
	if l.isClusterCACertificate(certBaseName) {
		return &configData.ClusterCAPolicy
	}
	return &configData.SignPolicy
}

// applyPolicyToCert applies policy configuration (usages, expiry, CA constraints) to the certificate
func (l *CertConfigLoader) applyPolicyToCert(cert *pkiutil.BKECert, policy *pkiutil.CertPolicy, certBaseName string) {
	// Try to find profile by BaseName, fall back to default
	var usages []string
	var expiry string
	var isCA bool

	if certBaseName != "" && policy.Signing.Profiles != nil {
		if profile, exists := policy.Signing.Profiles[certBaseName]; exists {
			l.log.Debugf("Using profile '%s' for certificate '%s'", certBaseName, certBaseName)
			usages = profile.Usages
			expiry = profile.Expiry
			if profile.CAConstraint != nil {
				isCA = profile.CAConstraint.IsCA
			}
			l.applyProfileConfig(cert, usages, expiry, isCA)
			return
		}
	}

	// Use default
	l.log.Debugf("Using default profile for certificate '%s'", certBaseName)
	usages = policy.Signing.Default.Usages
	expiry = policy.Signing.Default.Expiry
	if policy.Signing.Default.CAConstraint != nil {
		isCA = policy.Signing.Default.CAConstraint.IsCA
	}
	l.applyProfileConfig(cert, usages, expiry, isCA)
}

// applyProfileConfig applies a profile's configuration to the certificate
func (l *CertConfigLoader) applyProfileConfig(cert *pkiutil.BKECert, usages []string, expiry string, isCA bool) {
	// Apply usages
	if len(usages) > 0 {
		cert.Config.Config.Usages = l.parseExtKeyUsages(usages)
		cert.Config.BaseUsages = l.parseKeyUsages(usages)
	}
	// Apply expiry
	if expiry != "" {
		cert.Config.Validity = l.parseDuration(expiry)
	}
	// Apply CA constraint
	cert.IsCA = isCA
}

// isCACertificate checks if the certificate is a CA certificate
func (l *CertConfigLoader) isClusterCACertificate(certBaseName string) bool {

	return certBaseName == pkiutil.CACertAndKeyBaseName
}

// applyHostsToAltNames applies hosts to AltNames with deduplication
func (l *CertConfigLoader) applyHostsToAltNames(cert *pkiutil.BKECert, hosts []string) {
	ipSet, dnsSet := l.buildAltNameSets(cert)
	l.addHostsToSets(hosts, ipSet, dnsSet)
	l.addF5IPToSet(ipSet)
	l.assignSetsToCert(cert, ipSet, dnsSet)
}

// buildAltNameSets builds IP and DNS name sets from certificate's AltNames
func (l *CertConfigLoader) buildAltNameSets(cert *pkiutil.BKECert) (map[string]net.IP, map[string]bool) {
	ipSet := make(map[string]net.IP)
	dnsSet := make(map[string]bool)
	for _, ip := range cert.Config.AltNames.IPs {
		ipSet[ip.String()] = ip
	}
	for _, dns := range cert.Config.AltNames.DNSNames {
		dnsSet[dns] = true
	}
	return ipSet, dnsSet
}

// addHostsToSets adds hosts to IP set or DNS set based on their type
func (l *CertConfigLoader) addHostsToSets(hosts []string, ipSet map[string]net.IP, dnsSet map[string]bool) {
	if ipSet == nil || dnsSet == nil {
		return
	}
	for _, host := range hosts {
		if l.isIPAddress(host) {
			ip := l.parseIPAddress(host)
			if ip == nil {
				continue
			}
			ipStr := ip.String()
			if _, exists := ipSet[ipStr]; !exists {
				ipSet[ipStr] = ip
			}
			continue
		}
		if !dnsSet[host] {
			dnsSet[host] = true
		}
	}
}

// addF5IPToSet adds F5 load balancer IP from cluster configuration to IP set
func (l *CertConfigLoader) addF5IPToSet(ipSet map[string]net.IP) {
	if ipSet == nil {
		return
	}
	if l.bkeCluster == nil ||
		l.bkeCluster.Spec.ClusterConfig == nil ||
		l.bkeCluster.Spec.ClusterConfig.CustomExtra == nil {
		return
	}
	if v, ok := l.bkeCluster.Spec.ClusterConfig.CustomExtra["extraLoadBalanceIP"]; ok && v != "" {
		if ip := l.parseIPAddress(v); ip != nil {
			ipStr := ip.String()
			if _, exists := ipSet[ipStr]; !exists {
				ipSet[ipStr] = ip
			}
		}
	}
}

// assignSetsToCert assigns IP and DNS sets to certificate's AltNames
func (l *CertConfigLoader) assignSetsToCert(cert *pkiutil.BKECert, ipSet map[string]net.IP, dnsSet map[string]bool) {
	cert.Config.AltNames.IPs = make([]net.IP, 0, len(ipSet))
	for _, ip := range ipSet {
		cert.Config.AltNames.IPs = append(cert.Config.AltNames.IPs, ip)
	}
	cert.Config.AltNames.DNSNames = make([]string, 0, len(dnsSet))
	for dns := range dnsSet {
		cert.Config.AltNames.DNSNames = append(cert.Config.AltNames.DNSNames, dns)
	}
}

// isIPAddress checks if a string is an IP address
func (l *CertConfigLoader) isIPAddress(host string) bool {
	return strings.Contains(host, ".") && !strings.Contains(host, "localhost") &&
		!strings.Contains(host, "kubernetes") && !strings.Contains(host, "127.0.0.1")
}

// parseIPAddress parses IP address string to net.IP
func (l *CertConfigLoader) parseIPAddress(host string) net.IP {
	// Simple IP parsing - in production, use net.ParseIP
	if host == "127.0.0.1" {
		return net.ParseIP("127.0.0.1")
	}
	// For other IPs, try to parse them
	if ip := net.ParseIP(host); ip != nil {
		return ip
	}
	return nil
}

// parseKeyUsages converts string slice to []x509.KeyUsage (basic key usage)
func (l *CertConfigLoader) parseKeyUsages(usages []string) []x509.KeyUsage {
	return pkiutil.ParseKeyUsages(usages)
}

// parseExtKeyUsages converts string slice to []x509.ExtKeyUsage (extended key usage)
func (l *CertConfigLoader) parseExtKeyUsages(usages []string) []x509.ExtKeyUsage {
	return pkiutil.ParseExtKeyUsages(usages)
}

// parseDuration converts duration string to time.Duration
func (l *CertConfigLoader) parseDuration(durationStr string) time.Duration {
	// Parse duration like "24h", "1h30m", "8760h" (1 year)
	duration, err := time.ParseDuration(durationStr)
	if err != nil {
		l.log.Warnf("Failed to parse duration %s: %v", durationStr, err)
		// Return default 1 year if parsing fails
		return defaultCertValidity
	}
	return duration
}

// LoadLocalConfigData reads JSON files from /etc/openFuyao/certs/cert_config and builds CertConfigData
func (l *CertConfigLoader) LoadLocalConfigData() (*CertConfigData, error) {
	l.log.Infof("Attempting to load certificate configuration from local directory: %s", configDir)

	if ok, err := l.ensureLocalConfigDir(); err != nil || !ok {
		return nil, err
	}

	cfg := &CertConfigData{AvailableKeys: make(map[string]bool)}
	mappings := l.localFileMappings(cfg)

	hasData, loadedCount := l.readAndParseLocalFiles(mappings, cfg)
	if !hasData {
		l.log.Infof("No valid configuration files found in %s", configDir)
		return nil, nil
	}

	l.log.Infof("Successfully loaded %d configuration file(s) from local directory", loadedCount)
	return cfg, nil
}

// ensureLocalConfigDir checks the local directory exists and is a directory
func (l *CertConfigLoader) ensureLocalConfigDir() (bool, error) {
	info, err := os.Stat(configDir)
	if err != nil {
		if os.IsNotExist(err) {
			l.log.Infof("Directory %s does not exist", configDir)
			return false, nil
		}
		l.log.Warnf("Failed to stat directory %s: %v", configDir, err)
		return false, err
	}
	if !info.IsDir() {
		l.log.Warnf("Path %s exists but is not a directory", configDir)
		return false, nil
	}
	l.log.Infof("Found configuration directory %s", configDir)
	return true, nil
}

// readLocalJSON reads a local JSON file content
func (l *CertConfigLoader) readLocalJSON(filename string) (string, bool) {
	path := filepath.Join(configDir, filename)
	b, err := os.ReadFile(path)
	if err != nil || len(b) == 0 {
		return "", false
	}
	return string(b), true
}

// localFileMappings returns the filename to target mapping for local files
func (l *CertConfigLoader) localFileMappings(cfg *CertConfigData) map[string]interface{} {
	return map[string]interface{}{
		ConfigKeyClusterCAPolicy: &cfg.ClusterCAPolicy,
		ConfigKeyClusterCACSR:    &cfg.ClusterCACSR,
		ConfigKeySignPolicy:      &cfg.SignPolicy,

		ConfigKeyAPIServerCSR:              &cfg.APIServerCSR,
		ConfigKeyAPIServerEtcdClientCSR:    &cfg.APIServerEtcdClientCSR,
		ConfigKeyFrontProxyClientCSR:       &cfg.FrontProxyClientCSR,
		ConfigKeyAPIServerKubeletClientCSR: &cfg.APIServerKubeletClientCSR,

		ConfigKeyFrontProxyCACSR: &cfg.FrontProxyCACSR,

		ConfigKeyEtcdCACSR:                &cfg.EtcdCACSR,
		ConfigKeyEtcdServerCSR:            &cfg.EtcdServerCSR,
		ConfigKeyEtcdHealthcheckClientCSR: &cfg.EtcdHealthcheckClientCSR,
		ConfigKeyEtcdPeerCSR:              &cfg.EtcdPeerCSR,

		ConfigKeyAdminKubeConfigCSR:   &cfg.AdminKubeConfigCSR,
		ConfigKeyKubeletKubeConfigCSR: &cfg.KubeletKubeConfigCSR,
		ConfigKeyControllerManagerCSR: &cfg.ControllerManagerCSR,
		ConfigKeySchedulerCSR:         &cfg.SchedulerCSR,
	}
}

// readAndParseLocalFiles iterates files, parses JSON, and updates availability
func (l *CertConfigLoader) readAndParseLocalFiles(mappings map[string]interface{}, cfg *CertConfigData) (bool, int) {
	hasData := false
	loadedCount := 0
	for fileKey, target := range mappings {
		if content, ok := l.readLocalJSON(fileKey); ok {
			l.log.Debugf("Found configuration file: %s", fileKey)
			if err := json.Unmarshal([]byte(content), target); err != nil {
				l.log.Warnf("Failed to parse local file %s: %v", fileKey, err)
				cfg.AvailableKeys[fileKey] = false
				continue
			}
			cfg.AvailableKeys[fileKey] = true
			hasData = true
			loadedCount++
			l.log.Debugf("Successfully parsed configuration from file: %s", fileKey)
		} else {
			cfg.AvailableKeys[fileKey] = false
		}
	}
	return hasData, loadedCount
}

// SaveConfigMapData saves the provided configuration into the Kubernetes ConfigMap
func (l *CertConfigLoader) SaveConfigMapData(cfg *CertConfigData) error {
	if cfg == nil {
		return nil
	}
	data := l.buildConfigMapData(cfg)
	if len(data) == 0 {
		l.log.Infof("No data to save into ConfigMap")
		return nil
	}
	return l.upsertCertConfigMap(data)
}

// buildConfigMapData builds ConfigMap data from CertConfigData by marshaling available configurations
func (l *CertConfigLoader) buildConfigMapData(cfg *CertConfigData) map[string]string {
	data := make(map[string]string)
	add := func(key string, v interface{}) {
		if ok, exists := cfg.AvailableKeys[key]; exists && ok {
			config, err := json.Marshal(v)
			if err != nil {
				l.log.Warnf("Failed to marshal key %s for ConfigMap save: %v", key, err)
				return
			}
			data[key] = string(config)
		}
	}
	add(ConfigKeyClusterCAPolicy, cfg.ClusterCAPolicy)
	add(ConfigKeyClusterCACSR, cfg.ClusterCACSR)
	add(ConfigKeySignPolicy, cfg.SignPolicy)
	add(ConfigKeyAPIServerCSR, cfg.APIServerCSR)
	add(ConfigKeyAPIServerEtcdClientCSR, cfg.APIServerEtcdClientCSR)
	add(ConfigKeyFrontProxyClientCSR, cfg.FrontProxyClientCSR)
	add(ConfigKeyAPIServerKubeletClientCSR, cfg.APIServerKubeletClientCSR)
	add(ConfigKeyFrontProxyCACSR, cfg.FrontProxyCACSR)
	add(ConfigKeyEtcdCACSR, cfg.EtcdCACSR)
	add(ConfigKeyEtcdServerCSR, cfg.EtcdServerCSR)
	add(ConfigKeyEtcdHealthcheckClientCSR, cfg.EtcdHealthcheckClientCSR)
	add(ConfigKeyEtcdPeerCSR, cfg.EtcdPeerCSR)
	add(ConfigKeyAdminKubeConfigCSR, cfg.AdminKubeConfigCSR)
	add(ConfigKeyKubeletKubeConfigCSR, cfg.KubeletKubeConfigCSR)
	add(ConfigKeyControllerManagerCSR, cfg.ControllerManagerCSR)
	add(ConfigKeySchedulerCSR, cfg.SchedulerCSR)
	return data
}

// upsertCertConfigMap creates or updates the certificate configuration ConfigMap
func (l *CertConfigLoader) upsertCertConfigMap(data map[string]string) error {
	cm := &corev1.ConfigMap{}
	var getErr bool
	if err := phaseutil.RetryOnConflict(func() error {
		if err := l.client.Get(l.ctx, types.NamespacedName{Namespace: CertConfigMapNamespace, Name: CertConfigMapName}, cm); err != nil {
			getErr = true
			return err
		}
		getErr = false
		if cm.Data == nil {
			cm.Data = map[string]string{}
		}
		for k, v := range data {
			cm.Data[k] = v
		}
		return l.client.Update(l.ctx, cm)
	}); err != nil {
		if apierrors.IsNotFound(err) {
			return l.createCertConfigMap(data)
		}
		if getErr {
			return fmt.Errorf("failed to get ConfigMap %s/%s: %w", CertConfigMapNamespace, CertConfigMapName, err)
		}
		return fmt.Errorf("failed to update ConfigMap %s/%s: %w", CertConfigMapNamespace, CertConfigMapName, err)
	}
	l.log.Infof("Updated certificate configuration ConfigMap %s/%s", CertConfigMapNamespace, CertConfigMapName)
	return nil
}

// createCertConfigMap creates a new certificate configuration ConfigMap
func (l *CertConfigLoader) createCertConfigMap(data map[string]string) error {
	cm := &corev1.ConfigMap{}
	cm.Namespace = CertConfigMapNamespace
	cm.Name = CertConfigMapName
	cm.Data = data
	// Only set owner reference when the BKECluster exists in the same namespace
	// as the ConfigMap. Setting an ownerRef across namespaces causes
	// OwnerRefInvalidNamespace and the ConfigMap may be garbage-collected.
	if l.bkeCluster != nil {
		if l.bkeCluster.Namespace == CertConfigMapNamespace {
			controllerRef := metav1.NewControllerRef(l.bkeCluster, l.bkeCluster.GroupVersionKind())
			cm.SetOwnerReferences([]metav1.OwnerReference{*controllerRef})
		} else {
			l.log.Infof("Skipping ownerRef for ConfigMap %s/%s because BKECluster is in namespace %s", CertConfigMapNamespace, CertConfigMapName, l.bkeCluster.Namespace)
		}
	}
	if err := l.client.Create(l.ctx, cm); err != nil {
		return fmt.Errorf("failed to create ConfigMap %s/%s: %w", CertConfigMapNamespace, CertConfigMapName, err)
	}
	l.log.Infof("Created certificate configuration ConfigMap %s/%s", CertConfigMapNamespace, CertConfigMapName)
	return nil
}
