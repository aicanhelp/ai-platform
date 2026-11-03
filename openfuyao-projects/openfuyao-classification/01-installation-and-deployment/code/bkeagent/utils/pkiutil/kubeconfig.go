package pkiutil

import (
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

// KubeConfigOptions contains options for creating a KubeConfigGenerater
type KubeConfigOptions struct {
	PkiPath     string
	ClusterName string
	FileName    string
	ServerPort  string
	HostIP      string
	BKEConfig   *bkev1beta1.BKEConfig // Optional: used for adding master node IPs to AltNames
}

type KubeConfigGenerater struct {
	hostIP      string
	serverPort  string
	clusterName string
	userName    string
	fileName    string
	pkiPath     string
	bkeConfig   *bkev1beta1.BKEConfig
	BKECertSpec *BKECert
}

type clientCertAndKeyResult struct {
	Cert *x509.Certificate
	Key  *rsa.PrivateKey
}

const (
	// certConfigDir is the local certificate configuration directory
	certConfigDir = "/etc/openFuyao/certs/cert_config"
	// signPolicyConfigFile
	signPolicyConfigFile = "sign-policy.json"
	// defaultCertValidity represents the default certificate validity period (1 year)
	defaultCertValidity = 365 * 24 * time.Hour
	// MasterHADomain defines the domain name for HA cluster load balancer
	MasterHADomain = "master.bocloud.com"
	// KubeProxyBaseName defines the base name for kube-proxy
	KubeProxyKubeConfigFileName = "kube-proxy"
	// KubeProxyCommonName
	KubeProxyCommonName = "system:kube-proxy"
	// KubeProxyNodeGroup
	KubeProxyNodeGroup = "system:node-proxier"
)

// CertCSR represents a certificate signing request template
type CertCSR struct {
	CN  string `json:"CN"`
	O   string `json:"O"`
	C   string `json:"C"`
	ST  string `json:"ST"`
	L   string `json:"L"`
	OU  string `json:"OU"`
	Key struct {
		Algo string `json:"algo"`
		Size int    `json:"size"`
	} `json:"key"`
	Hosts []string `json:"hosts"`
}

// CertPolicy represents a certificate signing policy
type CertPolicy struct {
	Signing struct {
		Default struct {
			Usages       []string `json:"usages,omitempty"`
			Expiry       string   `json:"expiry,omitempty"`
			CAConstraint *struct {
				IsCA           bool `json:"is_ca,omitempty"`
				MaxPathLen     int  `json:"max_path_len,omitempty"`
				MaxPathLenZero bool `json:"max_path_len_zero,omitempty"`
			} `json:"ca_constraint,omitempty"`
		} `json:"default"`
		Profiles map[string]struct {
			Usages       []string `json:"usages,omitempty"`
			Expiry       string   `json:"expiry,omitempty"`
			CAConstraint *struct {
				IsCA           bool `json:"is_ca,omitempty"`
				MaxPathLen     int  `json:"max_path_len,omitempty"`
				MaxPathLenZero bool `json:"max_path_len_zero,omitempty"`
			} `json:"ca_constraint,omitempty"`
		} `json:"profiles,omitempty"`
	} `json:"signing"`
}

// kubeconfigCSRFileMap maps kubeconfig BaseName to CSR file name
var kubeconfigCSRFileMap = map[string]string{
	AdminKubeConfigFileName:             "admin-kubeconfig-csr.json",
	KubeletKubeConfigFileName:           "kubelet-kubeconfig-csr.json",
	ControllerManagerKubeConfigFileName: "controller-manager-csr.json",
	SchedulerKubeConfigFileName:         "scheduler-csr.json",
	KubeProxyKubeConfigFileName:         "kube-proxy-csr.json",
}

// NewKubeConfigGenerater creates a new KubeConfigGenerater with the provided options
func NewKubeConfigGenerater(opts KubeConfigOptions) *KubeConfigGenerater {
	return &KubeConfigGenerater{
		hostIP:      opts.HostIP,
		serverPort:  opts.ServerPort,
		clusterName: opts.ClusterName,
		fileName:    opts.FileName,
		pkiPath:     opts.PkiPath,
		bkeConfig:   opts.BKEConfig,
	}
}

func (k KubeConfigGenerater) server() string {
	return fmt.Sprintf("https://%s:%s", k.hostIP, k.serverPort)
}

func (k KubeConfigGenerater) contextName() string {
	return fmt.Sprintf("%s@%s", k.userName, k.clusterName)
}

func (k *KubeConfigGenerater) Generate() error {
	// step 1 find BKECert for this kubeconfig
	kubeConfigs := GetKubeConfigs()
	kubeConfigs = append(kubeConfigs, BKEKubeProxyKubeConfig())
	for _, kubeConfigSpec := range kubeConfigs {
		if kubeConfigSpec.Name == k.fileName {
			k.BKECertSpec = kubeConfigSpec
			break
		}
	}
	if k.BKECertSpec == nil {
		return fmt.Errorf("not found BKE kubeconfig spec for %s", k.fileName)
	}
	k.userName = fmt.Sprintf("%s-%s", k.clusterName, k.BKECertSpec.Config.CommonName)

	//step 2 load ca cert and key
	caCertSpec := BKECertRootCA()
	caCertSpec.PkiPath = k.pkiPath
	if err := CertExists(caCertSpec); err != nil {
		return err
	}
	caCert, caKey, err := loadCACertificateAuthority(caCertSpec)
	if err != nil {
		return err
	}
	// step 3 apply certificate configuration from local CSR and policy files
	if err := k.applyCertConfig(); err != nil {
		log.Warnf("Failed to apply certificate configuration, will use default: %v", err)
	}

	// step 4 load or generate client cert and key
	result, err := k.GenerateClientCertAndKey(caCert, caKey)
	if err != nil {
		return err
	}

	clientCertByte := EncodeCertToPEM(result.Cert)
	clientKeyByte := EncodeKeyToPEM(result.Key)
	certPath := filepath.Join(KubernetesDir, fmt.Sprintf("%s.crt", k.BKECertSpec.BaseName))
	keyPath := filepath.Join(KubernetesDir, fmt.Sprintf("%s.key", k.BKECertSpec.BaseName))

	if err = os.MkdirAll(KubernetesDir, utils.RwxRxRx); err != nil {
		return err
	}
	if err = os.WriteFile(certPath, clientCertByte, utils.RwRR); err != nil {
		return err
	}
	if err = os.WriteFile(keyPath, clientKeyByte, utils.RwRR); err != nil {
		return err
	}

	// step 5 generate kubeconfig
	cfg := k.newKubeConfigCfg(caCert, result.Cert, result.Key)
	return clientcmd.WriteToFile(cfg, pathForKubeConfig(k.BKECertSpec))
}

func (k *KubeConfigGenerater) newKubeConfigCfg(caCert, clientCert *x509.Certificate, clientKey *rsa.PrivateKey) api.Config {
	return api.Config{
		Clusters: map[string]*api.Cluster{
			k.clusterName: {
				Server:                   k.server(),
				CertificateAuthorityData: EncodeCertToPEM(caCert),
			},
		},
		Contexts: map[string]*api.Context{
			k.contextName(): {
				Cluster:  k.clusterName,
				AuthInfo: k.userName,
			},
		},
		AuthInfos: map[string]*api.AuthInfo{
			k.userName: {
				ClientCertificateData: EncodeCertToPEM(clientCert),
				ClientKeyData:         EncodeKeyToPEM(clientKey),
			},
		},
		CurrentContext: k.contextName(),
	}
}

func writeKubeConfig(certSpec *BKECert, data []byte) error {
	kubeConfigPath := pathForKubeConfig(certSpec)
	if err := os.MkdirAll(certSpec.PkiPath, 0755); err != nil {
		return err
	}
	return ioutil.WriteFile(kubeConfigPath, data, 0644)
}

func pathForKubeConfig(certSpec *BKECert) string {
	if certSpec.PkiPath == "" {
		certSpec.PkiPath = GetDefaultPkiPath()
	}
	return filepath.Join(certSpec.PkiPath, certSpec.BaseName+".conf")
}

// loadOrGenerateClientCertAndKey attempts to load client certificate and key from local .crt and .key files.
// If the files don't exist, it generates new certificate and key using the CA.
func (k *KubeConfigGenerater) GenerateClientCertAndKey(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*clientCertAndKeyResult, error) {
	certSpec := k.BKECertSpec

	clientCert, clientKey, err := NewCertAndKey(certSpec, caCert, caKey)
	if err != nil {
		log.Infof("failed to generate new cert and key for %q: %v", certSpec.Name, err)
		return nil, err
	}
	log.Infof("generated and saved new client cert and key for %q", certSpec.Name)
	return &clientCertAndKeyResult{
		Cert: clientCert,
		Key:  clientKey,
	}, nil
}

// applyCertConfig loads and applies certificate configuration from local CSR and policy files
func (k *KubeConfigGenerater) applyCertConfig() error {
	if k.BKECertSpec == nil {
		return fmt.Errorf("BKECertSpec is nil")
	}

	// Get CSR file name for this kubeconfig
	csrFileName, exists := kubeconfigCSRFileMap[k.BKECertSpec.BaseName]
	if !exists {
		log.Debugf("No CSR file mapping found for kubeconfig %s, skipping config application", k.BKECertSpec.BaseName)
		return nil
	}

	// Load CSR configuration
	csr, err := k.loadCSRFromFile(csrFileName)
	if err != nil {
		return fmt.Errorf("failed to load CSR from file %s: %w", csrFileName, err)
	}
	if csr == nil {
		log.Debugf("CSR file %s not found or empty, skipping config application", csrFileName)
		return nil
	}

	// Load sign policy
	policy, err := k.loadSignPolicy()
	if err != nil {
		return fmt.Errorf("failed to load sign policy: %w", err)
	}
	if policy == nil {
		log.Debugf("Sign policy file not found or empty, skipping policy application")
	}

	// Apply CSR configuration to certificate
	k.applyCSRToCert(csr)

	// Apply policy configuration if available
	if policy != nil {
		k.applyPolicyToCert(policy)
	}

	return nil
}

// loadCSRFromFile loads CSR configuration from a local JSON file
func (k *KubeConfigGenerater) loadCSRFromFile(filename string) (*CertCSR, error) {
	filePath := filepath.Join(certConfigDir, filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // File doesn't exist, return nil without error
		}
		return nil, err
	}

	if len(data) == 0 {
		return nil, nil
	}

	var csr CertCSR
	if err := json.Unmarshal(data, &csr); err != nil {
		return nil, fmt.Errorf("failed to parse CSR JSON: %w", err)
	}

	log.Infof("Loaded CSR configuration from %s", filePath)
	return &csr, nil
}

// loadSignPolicy loads sign policy from local JSON file
func (k *KubeConfigGenerater) loadSignPolicy() (*CertPolicy, error) {
	filePath := filepath.Join(certConfigDir, signPolicyConfigFile)
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // File doesn't exist, return nil without error
		}
		return nil, err
	}

	if len(data) == 0 {
		log.Infof("No sign policy configuration file found in %s", certConfigDir)
		return nil, nil
	}

	var policy CertPolicy
	if err := json.Unmarshal(data, &policy); err != nil {
		return nil, fmt.Errorf("failed to parse sign policy JSON: %w", err)
	}

	log.Infof("Loaded sign policy from %s", filePath)
	return &policy, nil
}

// applyCSRToCert applies CSR configuration to the certificate
func (k *KubeConfigGenerater) applyCSRToCert(csr *CertCSR) {
	cert := k.BKECertSpec

	// Apply basic CSR fields
	if cert.BaseName != KubeletKubeConfigFileName {
		cert.Config.CommonName = csr.CN
	}
	cert.Config.Organization = []string{csr.O}
	cert.Config.Country = []string{csr.C}
	cert.Config.Province = []string{csr.ST}
	cert.Config.Locality = []string{csr.L}
	cert.Config.OrganizationalUnit = []string{csr.OU}

	// Apply key configuration
	cert.Config.KeySize = csr.Key.Size
	cert.Config.PublicKeyAlgorithm = parsePublicKeyAlgorithm(csr.Key.Algo)

	// Apply hosts to AltNames (for kubeconfig, hosts are usually empty, but we handle it anyway)
	// This mimics the applyAltNamesToCert logic in generator.go
	if err := k.applyHostsToAltNames(csr.Hosts); err != nil {
		log.Warnf("Failed to apply hosts to alt names: %v", err)
		// Continue with default behavior if applying hosts fails
	}
}

// applyPolicyToCert applies policy configuration to the certificate
func (k *KubeConfigGenerater) applyPolicyToCert(policy *CertPolicy) {
	cert := k.BKECertSpec
	baseName := cert.BaseName

	// Try to find profile by BaseName, fall back to default
	var usages []string
	var expiry string
	var isCA bool

	if baseName != "" && policy.Signing.Profiles != nil {
		if profile, exists := policy.Signing.Profiles[baseName]; exists {
			log.Debugf("Using profile '%s' for certificate '%s'", baseName, baseName)
			usages = profile.Usages
			expiry = profile.Expiry
			if profile.CAConstraint != nil {
				isCA = profile.CAConstraint.IsCA
			}
			k.applyProfileConfig(usages, expiry, isCA)
			return
		}
	}

	// Use default
	log.Debugf("Using default profile for certificate '%s'", baseName)
	usages = policy.Signing.Default.Usages
	expiry = policy.Signing.Default.Expiry
	if policy.Signing.Default.CAConstraint != nil {
		isCA = policy.Signing.Default.CAConstraint.IsCA
	}
	k.applyProfileConfig(usages, expiry, isCA)
}

// applyProfileConfig applies a profile's configuration to the certificate
func (k *KubeConfigGenerater) applyProfileConfig(usages []string, expiry string, isCA bool) {
	cert := k.BKECertSpec

	// Apply usages
	if len(usages) > 0 {
		cert.Config.Usages = parseExtKeyUsages(usages)
		cert.Config.BaseUsages = parseKeyUsages(usages)
	}

	// Apply expiry
	if expiry != "" {
		cert.Config.Validity = parseDuration(expiry)
	}

	// Apply CA constraint
	cert.IsCA = isCA
}

// applyHostsToAltNames applies hosts to AltNames, mimicking applyAltNamesToCert in generator.go
// It merges both master node altNames (if bkeConfig is available) and CSR hosts together
// This ensures both sources of IPs/DNS names are preserved and combined
func (k *KubeConfigGenerater) applyHostsToAltNames(hosts []string) error {
	cert := k.BKECertSpec

	// Step 1: Get master node altNames from BKEConfig if available (similar to prepareBkeCerts in generator.go)
	// This is similar to how generator.go gets altNames for controller-manager and scheduler certificates
	var masterAltNames *certutil.AltNames
	if k.bkeConfig != nil {
		var err error
		masterAltNames, err = GetMasterNodeAltNames(k.bkeConfig)
		if err != nil {
			return fmt.Errorf("failed to get master node alt names: %w", err)
		}
	}

	// Step 2: Apply master node altNames first (similar to applyAltNamesToCert in generator.go)
	// This adds all master node IPs and DNS names to the certificate
	if masterAltNames != nil {
		cert.Config.AltNames.DNSNames = append(cert.Config.AltNames.DNSNames, masterAltNames.DNSNames...)
		cert.Config.AltNames.IPs = append(cert.Config.AltNames.IPs, masterAltNames.IPs...)
	}

	// Step 3: Append CSR hosts as extraAltNames using AppendSANsToAltNames (similar to generator.go)
	// This merges CSR hosts with master node altNames, and handles validation and deduplication automatically
	// Even if hosts is empty, we still want to ensure master node IPs are added (handled in step 2)
	if len(hosts) > 0 {
		if err := AppendSANsToAltNames(&cert.Config.AltNames, hosts, cert.BaseName); err != nil {
			return fmt.Errorf("failed to append CSR hosts to alt names for %q: %w", cert.BaseName, err)
		}
	}

	// Step 4: Add F5 load balancer IP from CustomExtra (similar to addF5IPToSet in config.go)
	// This adds the extraLoadBalanceIP from cluster configuration to the certificate
	k.addF5IPToAltNames(&cert.Config.AltNames)

	// Both CSR hosts, master node IPs, and F5 IP are now merged together in cert.Config.AltNames
	return nil
}

// addF5IPToAltNames adds F5 load balancer IP from cluster configuration to AltNames
// This mimics the addF5IPToSet function in config.go
func (k *KubeConfigGenerater) addF5IPToAltNames(altNames *certutil.AltNames) {
	// Guard clauses to reduce nesting depth
	if altNames == nil {
		return
	}
	if k.bkeConfig == nil || k.bkeConfig.CustomExtra == nil {
		return
	}
	v, ok := k.bkeConfig.CustomExtra["extraLoadBalanceIP"]
	if !ok || v == "" {
		return
	}
	ip := net.ParseIP(v)
	if ip == nil {
		return
	}
	// Check if IP already exists to avoid duplicates
	ipStr := ip.String()
	for _, existingIP := range altNames.IPs {
		if existingIP.String() == ipStr {
			return
		}
	}
	altNames.IPs = append(altNames.IPs, ip)
}

// parsePublicKeyAlgorithm converts string to x509.PublicKeyAlgorithm
func parsePublicKeyAlgorithm(algo string) x509.PublicKeyAlgorithm {
	switch strings.ToLower(algo) {
	case "rsa":
		return x509.RSA
	case "ecdsa", "ec":
		return x509.ECDSA
	case "ed25519":
		return x509.Ed25519
	default:
		return x509.RSA // Default to RSA
	}
}

// parseKeyUsages converts string slice to []x509.KeyUsage (basic key usage)
func parseKeyUsages(usages []string) []x509.KeyUsage {
	result := make([]x509.KeyUsage, 0, len(usages))

	usageMap := map[string]x509.KeyUsage{
		"digital signature": x509.KeyUsageDigitalSignature,
		"key encipherment":  x509.KeyUsageKeyEncipherment,
		"key agreement":     x509.KeyUsageKeyAgreement,
		"cert sign":         x509.KeyUsageCertSign,
		"crl sign":          x509.KeyUsageCRLSign,
		"encipher only":     x509.KeyUsageEncipherOnly,
		"decipher only":     x509.KeyUsageDecipherOnly,
	}

	for _, usage := range usages {
		if ku, exists := usageMap[strings.ToLower(usage)]; exists {
			result = append(result, ku)
		}
	}

	return result
}

// parseExtKeyUsages converts string slice to []x509.ExtKeyUsage (extended key usage)
func parseExtKeyUsages(usages []string) []x509.ExtKeyUsage {
	result := make([]x509.ExtKeyUsage, 0, len(usages))

	extUsageMap := map[string]x509.ExtKeyUsage{
		"any":          x509.ExtKeyUsageAny,
		"server auth":  x509.ExtKeyUsageServerAuth,
		"client auth":  x509.ExtKeyUsageClientAuth,
		"code signing": x509.ExtKeyUsageCodeSigning,
	}

	for _, usage := range usages {
		if extKu, exists := extUsageMap[strings.ToLower(usage)]; exists {
			result = append(result, extKu)
		}
	}

	return result
}

// parseDuration converts duration string to time.Duration
func parseDuration(durationStr string) time.Duration {
	duration, err := time.ParseDuration(durationStr)
	if err != nil {
		log.Warnf("Failed to parse duration %s: %v, using default", durationStr, err)
		return defaultCertValidity
	}
	return duration
}

// BKEKubeletKubeConfig returns the kubeconfig certificate for kubelet.
func BKEKubeProxyKubeConfig() *BKECert {
	return &BKECert{
		Name:     "kube-proxy",
		LongName: "kubeconfig for kube-proxy",
		BaseName: KubeProxyKubeConfigFileName,
		PkiPath:  KubernetesDir,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   KubeProxyCommonName,
				Organization: []string{KubeProxyNodeGroup},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}
