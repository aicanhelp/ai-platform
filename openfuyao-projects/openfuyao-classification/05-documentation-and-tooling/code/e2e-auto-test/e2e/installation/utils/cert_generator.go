// Package utils 提供证书配置文件生成工具
//
// 使用示例（在 BeforeAll 中调用）:
//
//	var certGenerator *utils.CertificateConfigGenerator
//
//	BeforeAll(func() {
//		// 方式1: 使用默认配置
//		certDir := "/tmp/cert-configs"
//		certGenerator = utils.NewCertificateConfigGenerator(certDir)
//		err := certGenerator.GenerateAllCertificateConfigs()
//		Expect(err).NotTo(HaveOccurred())
//
//		// 方式2: 使用自定义配置选项（可选）
//		options := &utils.CertificateConfigOptions{
//			NodeName: "node-1",
//		}
//		certGenerator = utils.NewCertificateConfigGeneratorWithOptions(certDir, options)
//		err = certGenerator.GenerateAllCertificateConfigs()
//		Expect(err).NotTo(HaveOccurred())
//
//		// 获取生成的配置文件目录
//		configDir := certGenerator.GetBaseDir()
//		// 现在可以使用 configDir 下的所有 JSON 文件进行证书验证测试
//	})
//
//	AfterAll(func() {
//		// 清理生成的配置文件
//		if certGenerator != nil {
//			certGenerator.CleanupCertificateConfigs()
//		}
//	})
package utils

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

// CSRConfig CSR (Certificate Signing Request) 配置
type CSRConfig struct {
	CN    string    `json:"CN"`
	O     string    `json:"O"`
	C     string    `json:"C"`
	ST    string    `json:"ST"`
	L     string    `json:"L"`
	OU    string    `json:"OU"`
	Key   KeyConfig `json:"key"`
	Hosts []string  `json:"hosts"`
}

// KeyConfig 密钥配置
type KeyConfig struct {
	Algo string `json:"algo"`
	Size int    `json:"size"`
}

// SignPolicyConfig 签名策略配置
type SignPolicyConfig struct {
	Signing SigningConfig `json:"signing"`
}

// SigningConfig 签名配置
type SigningConfig struct {
	Default  *ProfileConfig            `json:"default,omitempty"`
	Profiles map[string]*ProfileConfig `json:"profiles,omitempty"`
}

// ProfileConfig 配置文件
type ProfileConfig struct {
	Usages       []string      `json:"usages,omitempty"`
	Expiry       string        `json:"expiry,omitempty"`
	CAConstraint *CAConstraint `json:"ca_constraint,omitempty"`
}

// CAConstraint CA 约束配置
type CAConstraint struct {
	IsCA           bool `json:"is_ca,omitempty"`
	MaxPathLen     *int `json:"max_path_len,omitempty"`
	MaxPathLenZero bool `json:"max_path_len_zero,omitempty"`
}

// CertificateConfigOptions 证书配置选项
type CertificateConfigOptions struct {
	NodeName string // 节点名称，用于 kubelet-kubeconfig-csr.json（可选，默认 "test-node"）
}

// CertificateConfigGenerator 证书配置生成器
type CertificateConfigGenerator struct {
	baseDir string
	options *CertificateConfigOptions
}

// NewCertificateConfigGenerator 创建证书配置生成器（仅本地生成，不上传）
func NewCertificateConfigGenerator(baseDir string) *CertificateConfigGenerator {
	return &CertificateConfigGenerator{
		baseDir: baseDir,
		options: &CertificateConfigOptions{
			NodeName: "test-node",
		},
	}
}

// NewCertificateConfigGeneratorWithOptions 创建带选项的证书配置生成器
func NewCertificateConfigGeneratorWithOptions(baseDir string, options *CertificateConfigOptions) *CertificateConfigGenerator {
	if options == nil {
		options = &CertificateConfigOptions{
			NodeName: "test-node",
		}
	}
	return &CertificateConfigGenerator{
		baseDir: baseDir,
		options: options,
	}
}

// GenerateAllCertificateConfigs 生成所有证书配置文件
func (g *CertificateConfigGenerator) GenerateAllCertificateConfigs() error {
	// 确保基础目录存在
	if err := os.MkdirAll(g.baseDir, 0755); err != nil {
		return fmt.Errorf("创建证书配置目录失败: %w", err)
	}

	// 生成所有 CSR 配置文件
	csrConfigs := g.getAllCSRConfigs()
	for filename, config := range csrConfigs {
		if err := g.generateCSRConfig(filename, config); err != nil {
			return fmt.Errorf("生成 %s 失败: %w", filename, err)
		}
	}

	// 生成签名策略配置文件
	if err := g.generateSignPolicyConfig(); err != nil {
		return fmt.Errorf("生成 sign-policy.json 失败: %w", err)
	}

	// 生成集群 CA 策略配置文件
	if err := g.generateClusterCAPolicyConfig(); err != nil {
		return fmt.Errorf("生成 cluster-ca-policy.json 失败: %w", err)
	}

	return nil
}

// generateCSRConfig 生成单个 CSR 配置文件
func (g *CertificateConfigGenerator) generateCSRConfig(filename string, config *CSRConfig) error {
	filePath := filepath.Join(g.baseDir, filename)
	return g.writeJSONFile(filePath, config)
}

// generateSignPolicyConfig 生成签名策略配置文件
func (g *CertificateConfigGenerator) generateSignPolicyConfig() error {
	config := &SignPolicyConfig{
		Signing: SigningConfig{
			Default: &ProfileConfig{
				Usages: []string{"digital signature", "key encipherment"},
				Expiry: "8760h",
			},
			Profiles: map[string]*ProfileConfig{
				"ca": {
					Usages: []string{"cert sign", "crl sign"},
					Expiry: "87600h",
					CAConstraint: &CAConstraint{
						IsCA:       true,
						MaxPathLen: intPtr(0),
					},
				},
				"apiserver": {
					Usages: []string{"digital signature", "key encipherment", "server auth"},
					Expiry: "8760h",
				},
				"apiserver-etcd-client": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"apiserver-kubelet-client": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"front-proxy-client": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"front-proxy-ca": {
					Usages: []string{"cert sign", "crl sign"},
					Expiry: "87600h",
					CAConstraint: &CAConstraint{
						IsCA:       true,
						MaxPathLen: intPtr(1),
					},
				},
				"etcd/ca": {
					Usages: []string{"cert sign", "crl sign"},
					Expiry: "87600h",
					CAConstraint: &CAConstraint{
						IsCA:       true,
						MaxPathLen: intPtr(1),
					},
				},
				"etcd/server": {
					Usages: []string{"digital signature", "key encipherment", "server auth", "client auth"},
					Expiry: "8760h",
				},
				"etcd/peer": {
					Usages: []string{"digital signature", "key encipherment", "server auth", "client auth"},
					Expiry: "8760h",
				},
				"etcd/healthcheck-client": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"controller-manager": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"scheduler": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"kubelet": {
					Usages: []string{"digital signature", "key encipherment", "server auth", "client auth"},
					Expiry: "8760h",
				},
				"admin": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
				"kube-proxy": {
					Usages: []string{"digital signature", "key encipherment", "client auth"},
					Expiry: "8760h",
				},
			},
		},
	}

	filePath := filepath.Join(g.baseDir, "sign-policy.json")
	return g.writeJSONFile(filePath, config)
}

// generateClusterCAPolicyConfig 生成集群 CA 策略配置文件
func (g *CertificateConfigGenerator) generateClusterCAPolicyConfig() error {
	config := &SignPolicyConfig{
		Signing: SigningConfig{
			Default: &ProfileConfig{
				Usages: []string{"cert sign", "crl sign"},
				Expiry: "87600h",
				CAConstraint: &CAConstraint{
					IsCA:       true,
					MaxPathLen: intPtr(0),
				},
			},
			Profiles: map[string]*ProfileConfig{
				"ca": {
					Usages: []string{"cert sign", "crl sign"},
					Expiry: "87600h",
					CAConstraint: &CAConstraint{
						IsCA:           true,
						MaxPathLenZero: true,
					},
				},
			},
		},
	}

	filePath := filepath.Join(g.baseDir, "cluster-ca-policy.json")
	return g.writeJSONFile(filePath, config)
}

// getAllCSRConfigs 获取所有 CSR 配置
func (g *CertificateConfigGenerator) getAllCSRConfigs() map[string]*CSRConfig {
	return map[string]*CSRConfig{
		"cluster-ca-csr.json": {
			CN: "kubernetes",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-cluster-ca",
			L:  "Beijing-cluster-ca",
			OU: "Kubernetes-cluster-ca",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{"kubernetes", "kubernetes.default"},
		},
		"apiserver-csr.json": g.buildAPIServerCSRConfig(),
		"apiserver-kubelet-client-csr.json": {
			CN: "apiserver-kubelet-client",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-apiserver-kubelet-client",
			L:  "Beijing-apiserver-kubelet-client",
			OU: "Kubernetes-apiserver-kubelet-client",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"apiserver-etcd-client-csr.json": {
			CN: "apiserver-etcd-client",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-apiserver-etcd-client",
			L:  "Beijing-apiserver-etcd-client",
			OU: "Kubernetes-apiserver-etcd-client",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"controller-manager-csr.json": {
			CN: "system:kube-controller-manager",
			O:  "system:kube-controller-manager",
			C:  "CN",
			ST: "Beijing-controller-manager",
			L:  "Beijing-controller-manager",
			OU: "Kubernetes-controller-manager",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"scheduler-csr.json": {
			CN: "system:kube-scheduler",
			O:  "system:kube-scheduler",
			C:  "CN",
			ST: "Beijing-scheduler",
			L:  "Beijing-scheduler",
			OU: "Kubernetes-scheduler",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"kube-proxy-csr.json": {
			CN: "system:kube-proxy",
			O:  "system:node-proxier",
			C:  "CN",
			ST: "Beijing-kube-proxy",
			L:  "Beijing-kube-proxy",
			OU: "Kubernetes-kube-proxy",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"admin-kubeconfig-csr.json": {
			CN: "kubernetes-admin",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-admin",
			L:  "Beijing-admin",
			OU: "kubernetes-admin",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"kubelet-kubeconfig-csr.json": g.buildKubeletCSRConfig(),
		"etcd-ca-csr.json": {
			CN: "etcd-ca",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-etcd-ca",
			L:  "Beijing-etcd-ca",
			OU: "Kubernetes-etcd-ca",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"etcd-server-csr.json": {
			CN: "etcd-server",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-etcd-server",
			L:  "Beijing-etcd-server",
			OU: "Kubernetes-etcd-server",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{"localhost", "127.0.0.1"},
		},
		"etcd-peer-csr.json": {
			CN: "etcd-peer",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-etcd-peer",
			L:  "Beijing-etcd-peer",
			OU: "Kubernetes-etcd-peer",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"etcd-healthcheck-client-csr.json": {
			CN: "etcd-healthcheck-client",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-etcd-healthcheck-client",
			L:  "Beijing-etcd-healthcheck-client",
			OU: "Kubernetes-etcd-healthcheck-client",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"front-proxy-ca-csr.json": {
			CN: "front-proxy-ca",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-front-proxy-ca",
			L:  "Beijing-front-proxy-ca",
			OU: "Kubernetes-front-proxy-ca",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
		"front-proxy-client-csr.json": {
			CN: "front-proxy-client",
			O:  "system:masters",
			C:  "CN",
			ST: "Beijing-front-proxy-client",
			L:  "Beijing-front-proxy-client",
			OU: "Kubernetes-front-proxy-client",
			Key: KeyConfig{
				Algo: "rsa",
				Size: 2048,
			},
			Hosts: []string{},
		},
	}
}

// writeJSONFile 将配置写入 JSON 文件
func (g *CertificateConfigGenerator) writeJSONFile(filePath string, config interface{}) error {
	// 使用 MarshalIndent 生成格式化的 JSON
	jsonData, err := json.MarshalIndent(config, "", "\t")
	if err != nil {
		return fmt.Errorf("序列化 JSON 失败: %w", err)
	}

	// 写入文件
	if err := os.WriteFile(filePath, jsonData, 0644); err != nil {
		return fmt.Errorf("写入文件失败: %w", err)
	}

	return nil
}

// CleanupCertificateConfigs 清理证书配置文件
func (g *CertificateConfigGenerator) CleanupCertificateConfigs() error {
	if err := os.RemoveAll(g.baseDir); err != nil {
		return fmt.Errorf("清理证书配置目录失败: %w", err)
	}
	return nil
}

// GetBaseDir 获取基础目录路径
func (g *CertificateConfigGenerator) GetBaseDir() string {
	return g.baseDir
}

// buildAPIServerCSRConfig 构建 API Server CSR 配置
func (g *CertificateConfigGenerator) buildAPIServerCSRConfig() *CSRConfig {
	hosts := []string{
		"kubernetes",
		"kubernetes.default",
		"kubernetes.default.svc",
		"kubernetes.default.svc.cluster.local",
		"10.0.0.1",
		"{{.ClusterName}}",
		"{{.AdvertiseAddress}}",
	}

	return &CSRConfig{
		CN: "kube-apiserver",
		O:  "system:masters",
		C:  "CN",
		ST: "Beijing-apiserver",
		L:  "Beijing-apiserver",
		OU: "Kubernetes-apiserver",
		Key: KeyConfig{
			Algo: "rsa",
			Size: 2048,
		},
		Hosts: hosts,
	}
}

// buildKubeletCSRConfig 构建 Kubelet CSR 配置
func (g *CertificateConfigGenerator) buildKubeletCSRConfig() *CSRConfig {
	nodeName := g.options.NodeName
	if nodeName == "" {
		nodeName = "test-node"
	}

	return &CSRConfig{
		CN: fmt.Sprintf("system:node:%s", nodeName),
		O:  "system:nodes",
		C:  "CN",
		ST: "Beijing-kubelet",
		L:  "Beijing-kubelet",
		OU: "Kubernetes-kubelet",
		Key: KeyConfig{
			Algo: "rsa",
			Size: 2048,
		},
		Hosts: []string{},
	}
}

// intPtr 返回 int 指针
func intPtr(i int) *int {
	return &i
}

// GenerateCertificateChain 生成证书链文件
// 生成 global-ca.key、global-ca.crt 和 trust-chain.crt 文件
// outputDir: 输出目录路径
//
// 使用示例（在 BeforeAll 中调用）:
//
//	BeforeAll(func() {
//		certChainDir := "/tmp/cert-chain"
//		err := utils.GenerateCertificateChain(certChainDir)
//		Expect(err).NotTo(HaveOccurred())
//		// 生成的文件：
//		//   - global-ca.key: 签发集群CA的CA私钥
//		//   - global-ca.crt: 签发集群CA的CA证书
//		//   - trust-chain.crt: 完整的用户CA证书链
//	})
func GenerateCertificateChain(outputDir string) error {
	// 确保输出目录存在
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("创建输出目录失败: %w", err)
	}

	// 1. 生成根CA
	rootCAKey, rootCACert, err := generateRootCA()
	if err != nil {
		return fmt.Errorf("生成根CA失败: %w", err)
	}

	// 2. 生成中间CA（由根CA签发）
	intermediateCAKey, intermediateCACert, err := generateIntermediateCA(rootCAKey, rootCACert)
	if err != nil {
		return fmt.Errorf("生成中间CA失败: %w", err)
	}

	// 3. 生成global-ca（由中间CA签发）
	globalCAKey, globalCACert, err := generateGlobalCA(intermediateCAKey, intermediateCACert)
	if err != nil {
		return fmt.Errorf("生成global-ca失败: %w", err)
	}

	// 4. 保存 global-ca.key
	globalCAKeyPath := filepath.Join(outputDir, "global-ca.key")
	if err := savePrivateKey(globalCAKeyPath, globalCAKey); err != nil {
		return fmt.Errorf("保存global-ca.key失败: %w", err)
	}

	// 5. 保存 global-ca.crt
	globalCACertPath := filepath.Join(outputDir, "global-ca.crt")
	if err := saveCertificate(globalCACertPath, globalCACert); err != nil {
		return fmt.Errorf("保存global-ca.crt失败: %w", err)
	}

	// 6. 构造并保存证书链（global-ca -> intermediate-ca -> root-ca）
	trustChainPath := filepath.Join(outputDir, "trust-chain.crt")
	if err := saveCertificateChain(trustChainPath, globalCACert, intermediateCACert, rootCACert); err != nil {
		return fmt.Errorf("保存trust-chain.crt失败: %w", err)
	}

	return nil
}

// GenerateCertificateChainOnRemote 在远程节点上生成证书链文件
// 生成 global-ca.key、global-ca.crt 和 trust-chain.crt 文件到远程节点的指定目录
// exec: SSH执行器用于在远程节点上执行命令
// remoteDir: 远程节点上的输出目录路径（例如：/etc/openFuyao/certs）
func GenerateCertificateChainOnRemote(exec *executor.SSHExecutor, remoteDir string) error {
	// 1. 生成根CA
	rootCAKey, rootCACert, err := generateRootCA()
	if err != nil {
		return fmt.Errorf("生成根CA失败: %w", err)
	}

	// 2. 生成中间CA（由根CA签发）
	intermediateCAKey, intermediateCACert, err := generateIntermediateCA(rootCAKey, rootCACert)
	if err != nil {
		return fmt.Errorf("生成中间CA失败: %w", err)
	}

	// 3. 生成global-ca（由中间CA签发）
	globalCAKey, globalCACert, err := generateGlobalCA(intermediateCAKey, intermediateCACert)
	if err != nil {
		return fmt.Errorf("生成global-ca失败: %w", err)
	}

	// 4. 确保远程目录存在
	mkdirCmd := fmt.Sprintf("mkdir -p %s", remoteDir)
	result, err := exec.Exec(mkdirCmd)
	if err != nil {
		return fmt.Errorf("创建远程目录失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("创建远程目录失败: %s", result.Stderr)
	}

	// 5. 保存 global-ca.key 到远程节点
	globalCAKeyPath := filepath.Join(remoteDir, "global-ca.key")
	if err := savePrivateKeyToRemote(exec, globalCAKeyPath, globalCAKey); err != nil {
		return fmt.Errorf("保存global-ca.key到远程节点失败: %w", err)
	}

	// 6. 保存 global-ca.crt 到远程节点
	globalCACertPath := filepath.Join(remoteDir, "global-ca.crt")
	if err := saveCertificateToRemote(exec, globalCACertPath, globalCACert); err != nil {
		return fmt.Errorf("保存global-ca.crt到远程节点失败: %w", err)
	}

	// 7. 构造并保存证书链（global-ca -> intermediate-ca -> root-ca）到远程节点
	trustChainPath := filepath.Join(remoteDir, "trust-chain.crt")
	if err := saveCertificateChainToRemote(exec, trustChainPath, globalCACert, intermediateCACert, rootCACert); err != nil {
		return fmt.Errorf("保存trust-chain.crt到远程节点失败: %w", err)
	}

	return nil
}

// savePrivateKeyToRemote 保存私钥到远程节点
func savePrivateKeyToRemote(exec *executor.SSHExecutor, remotePath string, key *rsa.PrivateKey) error {
	keyBytes := x509.MarshalPKCS1PrivateKey(key)
	keyBlock := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: keyBytes,
	}

	keyPEM := pem.EncodeToMemory(keyBlock)
	if keyPEM == nil {
		return fmt.Errorf("编码私钥失败")
	}

	// 使用 base64 编码传输，避免特殊字符问题
	encoded := base64.StdEncoding.EncodeToString(keyPEM)
	// 使用 cat <<EOF 方式写入，比 echo 更安全
	cmd := fmt.Sprintf("cat <<EOF | base64 -d > %s\n%s\nEOF", remotePath, encoded)

	result, err := exec.Exec(cmd)
	if err != nil {
		return fmt.Errorf("写入私钥到远程节点失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("写入私钥到远程节点失败: %s", result.Stderr)
	}

	return nil
}

// saveCertificateToRemote 保存证书到远程节点
func saveCertificateToRemote(exec *executor.SSHExecutor, remotePath string, cert *x509.Certificate) error {
	certBlock := &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert.Raw,
	}

	certPEM := pem.EncodeToMemory(certBlock)
	if certPEM == nil {
		return fmt.Errorf("编码证书失败")
	}

	// 使用 base64 编码传输，避免特殊字符问题
	encoded := base64.StdEncoding.EncodeToString(certPEM)
	// 使用 cat <<EOF 方式写入，比 echo 更安全
	cmd := fmt.Sprintf("cat <<EOF | base64 -d > %s\n%s\nEOF", remotePath, encoded)

	result, err := exec.Exec(cmd)
	if err != nil {
		return fmt.Errorf("写入证书到远程节点失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("写入证书到远程节点失败: %s", result.Stderr)
	}

	return nil
}

// saveCertificateChainToRemote 保存证书链到远程节点（按顺序拼接）
func saveCertificateChainToRemote(exec *executor.SSHExecutor, remotePath string, certs ...*x509.Certificate) error {
	var chainPEM []byte

	// 按顺序写入证书
	for _, cert := range certs {
		certBlock := &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		}
		certPEM := pem.EncodeToMemory(certBlock)
		if certPEM == nil {
			return fmt.Errorf("编码证书失败")
		}
		chainPEM = append(chainPEM, certPEM...)
	}

	// 使用 base64 编码传输，避免特殊字符问题
	encoded := base64.StdEncoding.EncodeToString(chainPEM)
	// 使用 cat <<EOF 方式写入，比 echo 更安全
	cmd := fmt.Sprintf("cat <<EOF | base64 -d > %s\n%s\nEOF", remotePath, encoded)

	result, err := exec.Exec(cmd)
	if err != nil {
		return fmt.Errorf("写入证书链到远程节点失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("写入证书链到远程节点失败: %s", result.Stderr)
	}

	return nil
}

// generateRootCA 生成根CA证书和私钥
func generateRootCA() (*rsa.PrivateKey, *x509.Certificate, error) {
	// 生成私钥
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("生成私钥失败: %w", err)
	}

	// 创建证书模板
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Country:            []string{"CN"},
			Province:           []string{"Guangdong"},
			Locality:           []string{"Guangdong"},
			Organization:       []string{"Test Company"},
			OrganizationalUnit: []string{"Root CA"},
			CommonName:         "Test Root CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(10, 0, 0), // 10年
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
		MaxPathLenZero:        true,
	}

	// 自签名证书
	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return nil, nil, fmt.Errorf("创建证书失败: %w", err)
	}

	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, nil, fmt.Errorf("解析证书失败: %w", err)
	}

	return key, cert, nil
}

// generateIntermediateCA 生成中间CA证书和私钥（由根CA签发）
func generateIntermediateCA(parentKey *rsa.PrivateKey, parentCert *x509.Certificate) (*rsa.PrivateKey, *x509.Certificate, error) {
	// 生成私钥
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("生成私钥失败: %w", err)
	}

	// 创建证书模板
	template := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			Country:            []string{"CN"},
			Province:           []string{"Guangdong"},
			Locality:           []string{"Guangdong"},
			Organization:       []string{"Test Company"},
			OrganizationalUnit: []string{"Intermediate CA"},
			CommonName:         "Test Intermediate CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(10, 0, 0), // 10年
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            1,
		MaxPathLenZero:        false,
	}

	// 由根CA签发
	certDER, err := x509.CreateCertificate(rand.Reader, template, parentCert, &key.PublicKey, parentKey)
	if err != nil {
		return nil, nil, fmt.Errorf("创建证书失败: %w", err)
	}

	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, nil, fmt.Errorf("解析证书失败: %w", err)
	}

	return key, cert, nil
}

// generateGlobalCA 生成global-ca证书和私钥（由中间CA签发）
func generateGlobalCA(parentKey *rsa.PrivateKey, parentCert *x509.Certificate) (*rsa.PrivateKey, *x509.Certificate, error) {
	// 生成私钥
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("生成私钥失败: %w", err)
	}

	// 创建证书模板
	template := &x509.Certificate{
		SerialNumber: big.NewInt(3),
		Subject: pkix.Name{
			Country:            []string{"CN"},
			Province:           []string{"Guangdong"},
			Locality:           []string{"Guangdong"},
			Organization:       []string{"Test Company"},
			OrganizationalUnit: []string{"Cluster Issuer"},
			CommonName:         "Global CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(10, 0, 0), // 10年
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            1,
		MaxPathLenZero:        false,
	}

	// 由中间CA签发
	certDER, err := x509.CreateCertificate(rand.Reader, template, parentCert, &key.PublicKey, parentKey)
	if err != nil {
		return nil, nil, fmt.Errorf("创建证书失败: %w", err)
	}

	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, nil, fmt.Errorf("解析证书失败: %w", err)
	}

	return key, cert, nil
}

// savePrivateKey 保存私钥到文件
func savePrivateKey(filePath string, key *rsa.PrivateKey) error {
	keyFile, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("创建文件失败: %w", err)
	}
	defer keyFile.Close()

	keyBytes := x509.MarshalPKCS1PrivateKey(key)
	keyBlock := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: keyBytes,
	}

	if err := pem.Encode(keyFile, keyBlock); err != nil {
		return fmt.Errorf("编码私钥失败: %w", err)
	}

	return nil
}

// saveCertificate 保存证书到文件
func saveCertificate(filePath string, cert *x509.Certificate) error {
	certFile, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("创建文件失败: %w", err)
	}
	defer certFile.Close()

	certBlock := &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert.Raw,
	}

	if err := pem.Encode(certFile, certBlock); err != nil {
		return fmt.Errorf("编码证书失败: %w", err)
	}

	return nil
}

// saveCertificateChain 保存证书链到文件（按顺序拼接）
func saveCertificateChain(filePath string, certs ...*x509.Certificate) error {
	chainFile, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("创建文件失败: %w", err)
	}
	defer chainFile.Close()

	// 按顺序写入证书
	for _, cert := range certs {
		certBlock := &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		}
		if err := pem.Encode(chainFile, certBlock); err != nil {
			return fmt.Errorf("编码证书失败: %w", err)
		}
	}

	return nil
}
