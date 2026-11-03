// Package utils provides certificate verification tools
//
// Usage example (in test cases):
//
//	It("should verify certificate chain", func() {
//		// Create certificate verifier (default uses /etc/kubernetes/pki)
//		verifier := utils.NewCertificateVerifier("")
//
//		// Or specify custom PKI directory
//		verifier := utils.NewCertificateVerifier("/etc/kubernetes/pki")
//
//		// Execute verification
//		results, err := verifier.VerifyCertificateChain()
//		Expect(err).NotTo(HaveOccurred())
//
//		// Print verification results
//		verifier.PrintResults(results)
//
//		// Check if all verification items are successful
//		for _, item := range results.Items {
//			if !item.Success {
//				Fail(fmt.Sprintf("Certificate verification failed: %s - %v", item.Name, item.Error))
//			}
//		}
//	})
package utils

import (
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

// CertificateVerifier certificate verifier
type CertificateVerifier struct {
	pkiDir     string                  // PKI directory, default /etc/kubernetes/pki
	trustChain string                  // trust-chain.crt path
	executor   *executor.LocalExecutor // Local executor for executing commands on guide node (optional)
	nodeIP     string                  // Node IP for remote verification (optional)
	nodeUser   string                  // Node user for remote verification (optional)
	nodePass   string                  // Node password for remote verification (optional)
}

// NewCertificateVerifier creates a certificate verifier for local verification
// pkiDir: PKI directory path, defaults to /etc/kubernetes/pki
func NewCertificateVerifier(pkiDir string) *CertificateVerifier {
	if pkiDir == "" {
		pkiDir = "/etc/kubernetes/pki"
	}
	return &CertificateVerifier{
		pkiDir:     pkiDir,
		trustChain: filepath.Join(pkiDir, "trust-chain.crt"),
	}
}

// NewCertificateVerifierWithSSH creates a certificate verifier for remote verification via SSH
// pkiDir: PKI directory path on remote node, defaults to /etc/kubernetes/pki
// exec: Local executor for executing commands on guide node (tests run on guide node by default)
// nodeIP: Node IP address for remote verification
// nodeUser: Node user for SSH connection, defaults to "root"
// nodePass: Node password for SSH connection (optional, reserved for compatibility)
func NewCertificateVerifierWithSSH(pkiDir string, exec *executor.LocalExecutor, nodeIP, nodeUser, nodePass string) *CertificateVerifier {
	if pkiDir == "" {
		pkiDir = "/etc/kubernetes/pki"
	}
	if nodeUser == "" {
		nodeUser = "root"
	}
	return &CertificateVerifier{
		pkiDir:     pkiDir,
		trustChain: filepath.Join(pkiDir, "trust-chain.crt"),
		executor:   exec,
		nodeIP:     nodeIP,
		nodeUser:   nodeUser,
		nodePass:   nodePass,
	}
}

// VerificationResult verification result
type VerificationResult struct {
	Category string       // Verification category
	Items    []ResultItem // List of verification items
}

// ResultItem single verification item result
type ResultItem struct {
	Name        string // Certificate name
	Success     bool   // Whether verification succeeded
	Description string // Description
	Error       error  // Error message (if failed)
}

// VerifyCertificateChain verifies the complete certificate chain
// Returns verification result and error
func (v *CertificateVerifier) VerifyCertificateChain() (*VerificationResult, error) {
	results := &VerificationResult{
		Category: "Kubernetes Certificate Chain Verification",
		Items:    []ResultItem{},
	}

	// 1. Verify global-ca → cluster CA
	clusterCAItems := v.verifyGlobalCAToClusterCA()
	results.Items = append(results.Items, clusterCAItems...)

	// 2. Verify cluster CA → component certificates
	componentItems := v.verifyClusterCAToComponents()
	results.Items = append(results.Items, componentItems...)

	// 3. Verify front-proxy-ca → its client certificate
	frontProxyItems := v.verifyFrontProxyCA()
	results.Items = append(results.Items, frontProxyItems...)

	// 4. Verify etcd-ca → etcd component certificates
	etcdItems := v.verifyEtcdCA()
	results.Items = append(results.Items, etcdItems...)

	// 5. Verify ca-chain.crt existence and composition, and kube-apiserver.yaml configuration
	caChainItems := v.verifyCAChain()
	results.Items = append(results.Items, caChainItems...)

	// 6. Verify certificate Subject/SAN fields match expected CSR definitions
	csrMatchItems := v.verifyCertificateFieldsMatchCSR()
	results.Items = append(results.Items, csrMatchItems...)

	return results, nil
}

// VerifyWorkerCertificateChain verifies worker-node certificate items only.
// It intentionally skips control-plane and CA-chain related checks that do not exist on worker nodes.
func (v *CertificateVerifier) VerifyWorkerCertificateChain() (*VerificationResult, error) {
	results := &VerificationResult{
		Category: "Kubernetes Worker Certificate Verification",
		Items:    []ResultItem{},
	}

	// 1. Verify cluster CA -> worker component certificates
	workerChainItems := v.verifyWorkerClusterCAToComponents()
	results.Items = append(results.Items, workerChainItems...)

	// 2. Verify worker component certificate fields match expected CSR definitions
	workerCSRItems := v.verifyWorkerCertificateFieldsMatchCSR()
	results.Items = append(results.Items, workerCSRItems...)

	return results, nil
}

type certificateFieldExpectation struct {
	CN       string
	O        string
	C        string
	ST       string
	L        string
	OU       string
	KeyAlgo  string
	KeySize  int
	DNSNames []string
	IPAddrs  []string
}

// verifyGlobalCAToClusterCA verifies global-ca → cluster CA
func (v *CertificateVerifier) verifyGlobalCAToClusterCA() []ResultItem {
	items := []ResultItem{}

	// Load trust-chain.crt as root certificate
	rootCAs, err := v.loadTrustChain()
	if err != nil {
		return []ResultItem{{
			Name:        "Load root certificate",
			Success:     false,
			Description: "Failed to load trust-chain.crt",
			Error:       err,
		}}
	}

	// Verify ca.crt
	caCertPath := filepath.Join(v.pkiDir, "ca.crt")
	if err := v.verifyCertificate(caCertPath, rootCAs); err != nil {
		items = append(items, ResultItem{
			Name:        "Root CA → cluster CA",
			Success:     false,
			Description: "Verification failed",
			Error:       err,
		})
	} else {
		items = append(items, ResultItem{
			Name:        "Root CA → cluster CA",
			Success:     true,
			Description: "Verification succeeded",
		})
	}

	// Verify front-proxy-ca.crt
	frontProxyCAPath := filepath.Join(v.pkiDir, "front-proxy-ca.crt")
	if err := v.verifyCertificate(frontProxyCAPath, rootCAs); err != nil {
		items = append(items, ResultItem{
			Name:        "Root CA → front-proxy-ca",
			Success:     false,
			Description: "Verification failed",
			Error:       err,
		})
	} else {
		items = append(items, ResultItem{
			Name:        "Root CA → front-proxy-ca",
			Success:     true,
			Description: "Verification succeeded",
		})
	}

	// Verify etcd/ca.crt
	etcdCAPath := filepath.Join(v.pkiDir, "etcd", "ca.crt")
	if err := v.verifyCertificate(etcdCAPath, rootCAs); err != nil {
		items = append(items, ResultItem{
			Name:        "Root CA → etcd-ca",
			Success:     false,
			Description: "Verification failed",
			Error:       err,
		})
	} else {
		items = append(items, ResultItem{
			Name:        "Root CA → etcd-ca",
			Success:     true,
			Description: "Verification succeeded",
		})
	}

	return items
}

// verifyClusterCAToComponents verifies cluster CA → component certificates
func (v *CertificateVerifier) verifyClusterCAToComponents() []ResultItem {
	items := []ResultItem{}

	// Load trust-chain.crt + ca.crt as CA certificate pool
	rootCAs, err := v.loadTrustChain()
	if err != nil {
		return []ResultItem{{
			Name:        "Load root certificate",
			Success:     false,
			Description: "Failed to load trust-chain.crt",
			Error:       err,
		}}
	}

	// Load cluster CA
	caCertPath := filepath.Join(v.pkiDir, "ca.crt")
	caCert, err := v.loadCertificate(caCertPath)
	if err != nil {
		return []ResultItem{{
			Name:        "Load cluster CA",
			Success:     false,
			Description: "Failed to load ca.crt",
			Error:       err,
		}}
	}

	// Combine root CAs and cluster CA into a single list
	allCAs := make([]*x509.Certificate, 0, len(rootCAs)+1)
	allCAs = append(allCAs, rootCAs...)
	allCAs = append(allCAs, caCert)

	// Verify component certificates
	componentCerts := []struct {
		path string
		name string
	}{
		{filepath.Join(v.pkiDir, "apiserver.crt"), "Cluster CA → apiserver"},
		{filepath.Join(v.pkiDir, "apiserver-kubelet-client.crt"), "Cluster CA → apiserver-kubelet-client"},
		{"/etc/kubernetes/controller-manager.crt", "Cluster CA → controller-manager"},
		{"/etc/kubernetes/scheduler.crt", "Cluster CA → scheduler"},
		{"/etc/kubernetes/admin.crt", "Cluster CA → admin-kubeconfig"},
		{"/etc/kubernetes/kubelet.crt", "Cluster CA → kubelet-kubeconfig"},
		{"/etc/kubernetes/kube-proxy.crt", "Cluster CA → kube-proxy-kubeconfig"},
	}

	for _, cert := range componentCerts {
		if err := v.verifyCertificate(cert.path, allCAs); err != nil {
			items = append(items, ResultItem{
				Name:        cert.name,
				Success:     false,
				Description: "Verification failed",
				Error:       err,
			})
		} else {
			items = append(items, ResultItem{
				Name:        cert.name,
				Success:     true,
				Description: "Verification succeeded",
			})
		}
	}

	return items
}

// verifyWorkerClusterCAToComponents verifies cluster CA -> worker component certificates.
func (v *CertificateVerifier) verifyWorkerClusterCAToComponents() []ResultItem {
	items := []ResultItem{}

	// Worker nodes should have cluster CA locally.
	caCertPath := filepath.Join(v.pkiDir, "ca.crt")
	caCert, err := v.loadCertificate(caCertPath)
	if err != nil {
		return []ResultItem{{
			Name:        "Load cluster CA",
			Success:     false,
			Description: "Failed to load ca.crt",
			Error:       err,
		}}
	}

	workerCerts := []struct {
		path string
		name string
	}{
		{"/etc/kubernetes/kubelet.crt", "Cluster CA → kubelet-kubeconfig"},
		{"/etc/kubernetes/kube-proxy.crt", "Cluster CA → kube-proxy-kubeconfig"},
	}

	for _, cert := range workerCerts {
		if err := v.verifyCertificate(cert.path, []*x509.Certificate{caCert}); err != nil {
			items = append(items, ResultItem{
				Name:        cert.name,
				Success:     false,
				Description: "Verification failed",
				Error:       err,
			})
		} else {
			items = append(items, ResultItem{
				Name:        cert.name,
				Success:     true,
				Description: "Verification succeeded",
			})
		}
	}

	return items
}

// verifyFrontProxyCA verifies front-proxy-ca → its client certificate
func (v *CertificateVerifier) verifyFrontProxyCA() []ResultItem {
	items := []ResultItem{}

	// Load trust-chain.crt + front-proxy-ca.crt as CA certificate pool
	rootCAs, err := v.loadTrustChain()
	if err != nil {
		return []ResultItem{{
			Name:        "Load root certificate",
			Success:     false,
			Description: "Failed to load trust-chain.crt",
			Error:       err,
		}}
	}

	// Load front-proxy-ca
	frontProxyCAPath := filepath.Join(v.pkiDir, "front-proxy-ca.crt")
	frontProxyCA, err := v.loadCertificate(frontProxyCAPath)
	if err != nil {
		return []ResultItem{{
			Name:        "Load front-proxy-ca",
			Success:     false,
			Description: "Failed to load front-proxy-ca.crt",
			Error:       err,
		}}
	}

	// Combine root CAs and front-proxy CA into a single list
	allCAs := make([]*x509.Certificate, 0, len(rootCAs)+1)
	allCAs = append(allCAs, rootCAs...)
	allCAs = append(allCAs, frontProxyCA)

	// Verify front-proxy-client.crt
	frontProxyClientPath := filepath.Join(v.pkiDir, "front-proxy-client.crt")
	if err := v.verifyCertificate(frontProxyClientPath, allCAs); err != nil {
		items = append(items, ResultItem{
			Name:        "front-proxy-ca → front-proxy-client",
			Success:     false,
			Description: "Verification failed",
			Error:       err,
		})
	} else {
		items = append(items, ResultItem{
			Name:        "front-proxy-ca → front-proxy-client",
			Success:     true,
			Description: "Verification succeeded",
		})
	}

	return items
}

// verifyEtcdCA verifies etcd-ca → etcd component certificates
func (v *CertificateVerifier) verifyEtcdCA() []ResultItem {
	items := []ResultItem{}

	// Load trust-chain.crt + etcd/ca.crt as CA certificate pool
	rootCAs, err := v.loadTrustChain()
	if err != nil {
		return []ResultItem{{
			Name:        "Load root certificate",
			Success:     false,
			Description: "Failed to load trust-chain.crt",
			Error:       err,
		}}
	}

	// Load etcd-ca
	etcdCAPath := filepath.Join(v.pkiDir, "etcd", "ca.crt")
	etcdCA, err := v.loadCertificate(etcdCAPath)
	if err != nil {
		return []ResultItem{{
			Name:        "Load etcd-ca",
			Success:     false,
			Description: "Failed to load etcd/ca.crt",
			Error:       err,
		}}
	}

	// Combine root CAs and etcd CA into a single list
	allCAs := make([]*x509.Certificate, 0, len(rootCAs)+1)
	allCAs = append(allCAs, rootCAs...)
	allCAs = append(allCAs, etcdCA)

	// Verify etcd component certificates
	etcdCerts := []struct {
		path string
		name string
	}{
		{filepath.Join(v.pkiDir, "etcd", "healthcheck-client.crt"), "etcd-ca → healthcheck-client"},
		{filepath.Join(v.pkiDir, "etcd", "peer.crt"), "etcd-ca → peer"},
		{filepath.Join(v.pkiDir, "etcd", "server.crt"), "etcd-ca → server"},
		{filepath.Join(v.pkiDir, "apiserver-etcd-client.crt"), "etcd-ca → apiserver-etcd-client"},
	}

	for _, cert := range etcdCerts {
		if err := v.verifyCertificate(cert.path, allCAs); err != nil {
			items = append(items, ResultItem{
				Name:        cert.name,
				Success:     false,
				Description: "Verification failed",
				Error:       err,
			})
		} else {
			items = append(items, ResultItem{
				Name:        cert.name,
				Success:     true,
				Description: "Verification succeeded",
			})
		}
	}

	return items
}

// loadTrustChain loads all certificates from trust-chain.crt
func (v *CertificateVerifier) loadTrustChain() ([]*x509.Certificate, error) {
	var data []byte
	var err error

	if v.nodeIP != "" {
		// Remote verification: read file from target node
		result, err := v.executeOnRemoteNode(fmt.Sprintf("cat %s", v.trustChain))
		if err != nil {
			return nil, fmt.Errorf("failed to read trust-chain.crt from remote node: %w, stderr: %s", err, result.Stderr)
		}
		if result.ExitCode != 0 {
			return nil, fmt.Errorf("failed to read trust-chain.crt from remote node (exit code %d): %s", result.ExitCode, result.Stderr)
		}
		data = []byte(result.Stdout)
	} else {
		// Local verification: read file directly
		data, err = os.ReadFile(v.trustChain)
		if err != nil {
			return nil, fmt.Errorf("failed to read trust-chain.crt: %w", err)
		}
	}

	var certs []*x509.Certificate
	var block *pem.Block
	rest := data

	for {
		block, rest = pem.Decode(rest)
		if block == nil {
			break
		}
		if block.Type == "CERTIFICATE" {
			cert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return nil, fmt.Errorf("failed to parse certificate: %w", err)
			}
			certs = append(certs, cert)
		}
	}

	if len(certs) == 0 {
		return nil, fmt.Errorf("no certificates found in trust-chain.crt")
	}

	return certs, nil
}

// loadCertificate loads a single certificate file
func (v *CertificateVerifier) loadCertificate(certPath string) (*x509.Certificate, error) {
	var data []byte
	var err error

	if v.nodeIP != "" {
		// Remote verification: read file from target node
		result, err := v.executeOnRemoteNode(fmt.Sprintf("cat %s", certPath))
		if err != nil {
			return nil, fmt.Errorf("failed to read certificate file from remote node: %w, stderr: %s", err, result.Stderr)
		}
		if result.ExitCode != 0 {
			return nil, fmt.Errorf("failed to read certificate file from remote node (exit code %d): %s", result.ExitCode, result.Stderr)
		}
		data = []byte(result.Stdout)
	} else {
		// Local verification: read file directly
		data, err = os.ReadFile(certPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read certificate file: %w", err)
		}
	}

	block, _ := pem.Decode(data)
	if block == nil {
		return nil, fmt.Errorf("failed to decode PEM format")
	}

	if block.Type != "CERTIFICATE" {
		return nil, fmt.Errorf("not a certificate file")
	}

	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse certificate: %w", err)
	}

	return cert, nil
}

// verifyCertificate verifies certificate using root certificate pool
// Only checks signature chain and validity period, skips Key Usage checks
func (v *CertificateVerifier) verifyCertificate(certPath string, rootCAs []*x509.Certificate) error {
	v.logVerify("chain-check start: %s", certPath)
	cert, err := v.loadCertificate(certPath)
	if err != nil {
		v.logVerify("chain-check load-failed: %s, err=%v", certPath, err)
		return err
	}

	// Find the signing CA by checking signature
	for _, ca := range rootCAs {
		if err := cert.CheckSignatureFrom(ca); err == nil {
			// Found signing CA, verify validity period
			now := time.Now()
			if now.Before(cert.NotBefore) || now.After(cert.NotAfter) {
				v.logVerify("chain-check validity-failed: %s, notBefore=%v, notAfter=%v, now=%v", certPath, cert.NotBefore, cert.NotAfter, now)
				return fmt.Errorf("certificate is not valid: notBefore=%v, notAfter=%v, now=%v", cert.NotBefore, cert.NotAfter, now)
			}
			v.logVerify("chain-check success: %s", certPath)
			return nil // Signature and validity OK
		}
	}

	v.logVerify("chain-check signer-not-found: %s", certPath)
	return fmt.Errorf("certificate is not signed by any root CA")
}

// verifyCAChain verifies ca-chain.crt existence, composition, and kube-apiserver.yaml configuration
func (v *CertificateVerifier) verifyCAChain() []ResultItem {
	items := []ResultItem{}

	// 1. Check if ca-chain.crt exists
	caChainPath := filepath.Join(v.pkiDir, "ca-chain.crt")
	caChainExists, err := v.fileExists(caChainPath)
	if err != nil {
		items = append(items, ResultItem{
			Name:        "ca-chain.crt existence check",
			Success:     false,
			Description: "Failed to check if ca-chain.crt exists",
			Error:       err,
		})
		return items
	}
	if !caChainExists {
		items = append(items, ResultItem{
			Name:        "ca-chain.crt existence check",
			Success:     false,
			Description: "ca-chain.crt does not exist",
			Error:       fmt.Errorf("ca-chain.crt not found at %s", caChainPath),
		})
		return items
	}
	items = append(items, ResultItem{
		Name:        "ca-chain.crt existence check",
		Success:     true,
		Description: "ca-chain.crt exists",
	})

	// 2. Verify ca-chain.crt is composed of ca.crt and trust-chain.crt (in that order)
	compositionErr := v.verifyCAChainComposition(caChainPath)
	if compositionErr != nil {
		items = append(items, ResultItem{
			Name:        "ca-chain.crt composition verification",
			Success:     false,
			Description: "ca-chain.crt is not correctly composed of ca.crt and trust-chain.crt",
			Error:       compositionErr,
		})
	} else {
		items = append(items, ResultItem{
			Name:        "ca-chain.crt composition verification",
			Success:     true,
			Description: "ca-chain.crt is correctly composed of ca.crt and trust-chain.crt",
		})
	}

	// 3. Check kube-apiserver.yaml --client-ca-file parameter
	apiServerYAMLPath := "/etc/kubernetes/manifests/kube-apiserver.yaml"
	apiServerCheckErr := v.verifyAPIServerClientCAFile(apiServerYAMLPath, caChainPath)
	if apiServerCheckErr != nil {
		items = append(items, ResultItem{
			Name:        "kube-apiserver.yaml --client-ca-file verification",
			Success:     false,
			Description: fmt.Sprintf("--client-ca-file in kube-apiserver.yaml does not point to %s", caChainPath),
			Error:       apiServerCheckErr,
		})
	} else {
		items = append(items, ResultItem{
			Name:        "kube-apiserver.yaml --client-ca-file verification",
			Success:     true,
			Description: fmt.Sprintf("--client-ca-file in kube-apiserver.yaml correctly points to %s", caChainPath),
		})
	}

	// 4. Check kube-controller-manager.yaml kubeconfig parameters
	controllerManagerYAMLPath := "/etc/kubernetes/manifests/kube-controller-manager.yaml"
	controllerManagerKubeconfigPath := "/etc/kubernetes/controller-manager.conf"
	controllerManagerItems := v.verifyControllerManagerKubeconfig(controllerManagerYAMLPath, controllerManagerKubeconfigPath)
	items = append(items, controllerManagerItems...)

	// 5. Check kube-scheduler.yaml kubeconfig parameters
	schedulerYAMLPath := "/etc/kubernetes/manifests/kube-scheduler.yaml"
	schedulerKubeconfigPath := "/etc/kubernetes/scheduler.conf"
	schedulerItems := v.verifySchedulerKubeconfig(schedulerYAMLPath, schedulerKubeconfigPath)
	items = append(items, schedulerItems...)

	// 6. Check kubelet kubeconfig parameter (via systemctl status)
	kubeletKubeconfigPath := "/etc/kubernetes/kubelet.conf"
	kubeletItems := v.verifyKubeletKubeconfig(kubeletKubeconfigPath)
	items = append(items, kubeletItems...)

	return items
}

// verifyCertificateFieldsMatchCSR verifies that issued certificate fields match expected CSR fields.
// Focuses on CN/O/OU and deterministic SAN entries used by automation.
func (v *CertificateVerifier) verifyCertificateFieldsMatchCSR() []ResultItem {
	type certCheck struct {
		certPath string
		name     string
		exp      certificateFieldExpectation
	}

	checks := []certCheck{
		{
			certPath: filepath.Join(v.pkiDir, "apiserver.crt"),
			name:     "CSR field match → apiserver",
			exp: certificateFieldExpectation{
				CN:       "kube-apiserver",
				O:        "system:masters",
				C:        "CN",
				ST:       "Beijing-apiserver",
				L:        "Beijing-apiserver",
				OU:       "Kubernetes-apiserver",
				KeyAlgo:  "rsa",
				KeySize:  2048,
				DNSNames: []string{"kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local"},
				IPAddrs:  []string{"10.0.0.1"},
			},
		},
		{
			certPath: filepath.Join(v.pkiDir, "apiserver-kubelet-client.crt"),
			name:     "CSR field match → apiserver-kubelet-client",
			exp: certificateFieldExpectation{
				CN:      "apiserver-kubelet-client",
				O:       "system:masters",
				C:       "CN",
				ST:      "Beijing-apiserver-kubelet-client",
				L:       "Beijing-apiserver-kubelet-client",
				OU:      "Kubernetes-apiserver-kubelet-client",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: "/etc/kubernetes/controller-manager.crt",
			name:     "CSR field match → controller-manager",
			exp: certificateFieldExpectation{
				CN:      "system:kube-controller-manager",
				O:       "system:kube-controller-manager",
				C:       "CN",
				ST:      "Beijing-controller-manager",
				L:       "Beijing-controller-manager",
				OU:      "Kubernetes-controller-manager",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: "/etc/kubernetes/scheduler.crt",
			name:     "CSR field match → scheduler",
			exp: certificateFieldExpectation{
				CN:      "system:kube-scheduler",
				O:       "system:kube-scheduler",
				C:       "CN",
				ST:      "Beijing-scheduler",
				L:       "Beijing-scheduler",
				OU:      "Kubernetes-scheduler",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: "/etc/kubernetes/admin.crt",
			name:     "CSR field match → admin-kubeconfig",
			exp: certificateFieldExpectation{
				CN:      "kubernetes-admin",
				O:       "system:masters",
				C:       "CN",
				ST:      "Beijing-admin",
				L:       "Beijing-admin",
				OU:      "kubernetes-admin",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: "/etc/kubernetes/kube-proxy.crt",
			name:     "CSR field match → kube-proxy-kubeconfig",
			exp: certificateFieldExpectation{
				CN:      "system:kube-proxy",
				O:       "system:node-proxier",
				C:       "CN",
				ST:      "Beijing-kube-proxy",
				L:       "Beijing-kube-proxy",
				OU:      "Kubernetes-kube-proxy",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: "/etc/kubernetes/kubelet.crt",
			name:     "CSR field match → kubelet-kubeconfig",
			exp: certificateFieldExpectation{
				O:       "system:nodes",
				C:       "CN",
				ST:      "Beijing-kubelet",
				L:       "Beijing-kubelet",
				OU:      "Kubernetes-kubelet",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: filepath.Join(v.pkiDir, "apiserver-etcd-client.crt"),
			name:     "CSR field match → apiserver-etcd-client",
			exp: certificateFieldExpectation{
				CN:      "apiserver-etcd-client",
				O:       "system:masters",
				C:       "CN",
				ST:      "Beijing-apiserver-etcd-client",
				L:       "Beijing-apiserver-etcd-client",
				OU:      "Kubernetes-apiserver-etcd-client",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: filepath.Join(v.pkiDir, "etcd", "healthcheck-client.crt"),
			name:     "CSR field match → etcd-healthcheck-client",
			exp: certificateFieldExpectation{
				CN:      "etcd-healthcheck-client",
				O:       "system:masters",
				C:       "CN",
				ST:      "Beijing-etcd-healthcheck-client",
				L:       "Beijing-etcd-healthcheck-client",
				OU:      "Kubernetes-etcd-healthcheck-client",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: filepath.Join(v.pkiDir, "etcd", "peer.crt"),
			name:     "CSR field match → etcd-peer",
			exp: certificateFieldExpectation{
				CN:      "etcd-peer",
				O:       "system:masters",
				C:       "CN",
				ST:      "Beijing-etcd-peer",
				L:       "Beijing-etcd-peer",
				OU:      "Kubernetes-etcd-peer",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: filepath.Join(v.pkiDir, "etcd", "server.crt"),
			name:     "CSR field match → etcd-server",
			exp: certificateFieldExpectation{
				CN:       "etcd-server",
				O:        "system:masters",
				C:        "CN",
				ST:       "Beijing-etcd-server",
				L:        "Beijing-etcd-server",
				OU:       "Kubernetes-etcd-server",
				KeyAlgo:  "rsa",
				KeySize:  2048,
				DNSNames: []string{"localhost"},
				IPAddrs:  []string{"127.0.0.1"},
			},
		},
		{
			certPath: filepath.Join(v.pkiDir, "front-proxy-client.crt"),
			name:     "CSR field match → front-proxy-client",
			exp: certificateFieldExpectation{
				CN:      "front-proxy-client",
				O:       "system:masters",
				C:       "CN",
				ST:      "Beijing-front-proxy-client",
				L:       "Beijing-front-proxy-client",
				OU:      "Kubernetes-front-proxy-client",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
	}

	items := make([]ResultItem, 0, len(checks))
	for _, check := range checks {
		v.logVerify("csr-field-check start: %s (%s)", check.name, check.certPath)
		cert, err := v.loadCertificate(check.certPath)
		if err != nil {
			v.logVerify("csr-field-check load-failed: %s, err=%v", check.name, err)
			items = append(items, ResultItem{
				Name:        check.name,
				Success:     false,
				Description: "Failed to load certificate for CSR field comparison",
				Error:       err,
			})
			continue
		}

		if err := verifyCertificateFields(cert, check.exp); err != nil {
			v.logVerify("csr-field-check failed: %s, err=%v", check.name, err)
			items = append(items, ResultItem{
				Name:        check.name,
				Success:     false,
				Description: "Issued certificate fields do not match expected CSR fields",
				Error:       err,
			})
		} else {
			v.logVerify("csr-field-check success: %s", check.name)
			items = append(items, ResultItem{
				Name:        check.name,
				Success:     true,
				Description: "Issued certificate fields match expected CSR fields",
			})
		}
	}

	return items
}

func (v *CertificateVerifier) verifyWorkerCertificateFieldsMatchCSR() []ResultItem {
	type certCheck struct {
		certPath string
		name     string
		exp      certificateFieldExpectation
	}

	checks := []certCheck{
		{
			certPath: "/etc/kubernetes/kube-proxy.crt",
			name:     "CSR field match → kube-proxy-kubeconfig",
			exp: certificateFieldExpectation{
				CN:      "system:kube-proxy",
				O:       "system:node-proxier",
				C:       "CN",
				ST:      "Beijing-kube-proxy",
				L:       "Beijing-kube-proxy",
				OU:      "Kubernetes-kube-proxy",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
		{
			certPath: "/etc/kubernetes/kubelet.crt",
			name:     "CSR field match → kubelet-kubeconfig",
			exp: certificateFieldExpectation{
				O:       "system:nodes",
				C:       "CN",
				ST:      "Beijing-kubelet",
				L:       "Beijing-kubelet",
				OU:      "Kubernetes-kubelet",
				KeyAlgo: "rsa",
				KeySize: 2048,
			},
		},
	}

	items := make([]ResultItem, 0, len(checks))
	for _, check := range checks {
		v.logVerify("worker-csr-field-check start: %s (%s)", check.name, check.certPath)
		cert, err := v.loadCertificate(check.certPath)
		if err != nil {
			v.logVerify("worker-csr-field-check load-failed: %s, err=%v", check.name, err)
			items = append(items, ResultItem{
				Name:        check.name,
				Success:     false,
				Description: "Failed to load certificate for CSR field comparison",
				Error:       err,
			})
			continue
		}

		if err := verifyCertificateFields(cert, check.exp); err != nil {
			v.logVerify("worker-csr-field-check failed: %s, err=%v", check.name, err)
			items = append(items, ResultItem{
				Name:        check.name,
				Success:     false,
				Description: "Issued certificate fields do not match expected CSR fields",
				Error:       err,
			})
		} else {
			v.logVerify("worker-csr-field-check success: %s", check.name)
			items = append(items, ResultItem{
				Name:        check.name,
				Success:     true,
				Description: "Issued certificate fields match expected CSR fields",
			})
		}
	}

	return items
}

func verifyCertificateFields(cert *x509.Certificate, exp certificateFieldExpectation) error {
	if exp.CN != "" && cert.Subject.CommonName != exp.CN {
		return fmt.Errorf("CN mismatch: expected %q, got %q", exp.CN, cert.Subject.CommonName)
	}
	if exp.CN == "" && !strings.HasPrefix(cert.Subject.CommonName, "system:node:") && exp.O == "system:nodes" {
		return fmt.Errorf("CN mismatch: expected prefix %q, got %q", "system:node:", cert.Subject.CommonName)
	}

	if exp.O != "" && !stringInSlice(exp.O, cert.Subject.Organization) {
		return fmt.Errorf("O mismatch: expected %q in %v", exp.O, cert.Subject.Organization)
	}
	if exp.C != "" && !stringInSlice(exp.C, cert.Subject.Country) {
		return fmt.Errorf("C mismatch: expected %q in %v", exp.C, cert.Subject.Country)
	}
	if exp.ST != "" && !stringInSlice(exp.ST, cert.Subject.Province) {
		return fmt.Errorf("ST mismatch: expected %q in %v", exp.ST, cert.Subject.Province)
	}
	if exp.L != "" && !stringInSlice(exp.L, cert.Subject.Locality) {
		return fmt.Errorf("L mismatch: expected %q in %v", exp.L, cert.Subject.Locality)
	}

	if exp.OU != "" && !stringInSlice(exp.OU, cert.Subject.OrganizationalUnit) {
		return fmt.Errorf("OU mismatch: expected %q in %v", exp.OU, cert.Subject.OrganizationalUnit)
	}

	if len(exp.DNSNames) > 0 {
		missingDNS := missingStrings(exp.DNSNames, cert.DNSNames)
		if len(missingDNS) > 0 {
			return fmt.Errorf("DNS SAN mismatch: missing %v, actual %v", missingDNS, cert.DNSNames)
		}
	}

	if len(exp.IPAddrs) > 0 {
		actualIPs := make([]string, 0, len(cert.IPAddresses))
		for _, ip := range cert.IPAddresses {
			actualIPs = append(actualIPs, ip.String())
		}
		missingIPs := missingStrings(exp.IPAddrs, actualIPs)
		if len(missingIPs) > 0 {
			return fmt.Errorf("IP SAN mismatch: missing %v, actual %v", missingIPs, actualIPs)
		}
	}

	if exp.KeyAlgo != "" || exp.KeySize > 0 {
		algo, size := getPublicKeyAlgoAndSize(cert.PublicKey)
		if exp.KeyAlgo != "" && !strings.EqualFold(algo, exp.KeyAlgo) {
			return fmt.Errorf("key algo mismatch: expected %q, got %q", exp.KeyAlgo, algo)
		}
		if exp.KeySize > 0 && size != exp.KeySize {
			return fmt.Errorf("key size mismatch: expected %d, got %d", exp.KeySize, size)
		}
	}

	return nil
}

func getPublicKeyAlgoAndSize(publicKey interface{}) (string, int) {
	switch pk := publicKey.(type) {
	case *rsa.PublicKey:
		return "rsa", pk.Size() * 8
	case rsa.PublicKey:
		return "rsa", pk.Size() * 8
	case *ecdsa.PublicKey:
		return "ecdsa", pk.Params().BitSize
	case ecdsa.PublicKey:
		return "ecdsa", pk.Params().BitSize
	case ed25519.PublicKey:
		return "ed25519", len(pk) * 8
	case *ed25519.PublicKey:
		return "ed25519", len(*pk) * 8
	default:
		return "unknown", 0
	}
}

func (v *CertificateVerifier) logVerify(format string, args ...interface{}) {
	fmt.Printf("[cert-verify] "+format+"\n", args...)
}

func stringInSlice(target string, values []string) bool {
	for _, v := range values {
		if v == target {
			return true
		}
	}
	return false
}

func missingStrings(expected, actual []string) []string {
	set := make(map[string]struct{}, len(actual))
	for _, s := range actual {
		// Normalize IP-like values to canonical format where possible.
		if ip := net.ParseIP(s); ip != nil {
			set[ip.String()] = struct{}{}
			continue
		}
		set[s] = struct{}{}
	}

	missing := make([]string, 0)
	for _, s := range expected {
		key := s
		if ip := net.ParseIP(s); ip != nil {
			key = ip.String()
		}
		if _, ok := set[key]; !ok {
			missing = append(missing, s)
		}
	}
	sort.Strings(missing)
	return missing
}

// fileExists checks if a file exists (works for both local and remote)
func (v *CertificateVerifier) fileExists(filePath string) (bool, error) {
	if v.nodeIP != "" {
		// Remote verification: check file existence on target node
		result, err := v.executeOnRemoteNode(fmt.Sprintf("test -f %s && echo exists || echo notfound", filePath))
		if err != nil {
			return false, fmt.Errorf("failed to check file existence on remote node: %w, stderr: %s", err, result.Stderr)
		}
		if result.ExitCode != 0 {
			return false, fmt.Errorf("failed to check file existence on remote node (exit code %d): %s", result.ExitCode, result.Stderr)
		}
		return strings.Contains(result.Stdout, "exists"), nil
	} else {
		// Local verification: check file existence directly
		_, err := os.Stat(filePath)
		if os.IsNotExist(err) {
			return false, nil
		}
		if err != nil {
			return false, fmt.Errorf("failed to check file existence: %w", err)
		}
		return true, nil
	}
}

// readFileContent reads file content (works for both local and remote)
func (v *CertificateVerifier) readFileContent(filePath string) ([]byte, error) {
	if v.nodeIP != "" {
		// Remote verification: read file from target node
		result, err := v.executeOnRemoteNode(fmt.Sprintf("cat %s", filePath))
		if err != nil {
			return nil, fmt.Errorf("failed to read file from remote node: %w, stderr: %s", err, result.Stderr)
		}
		if result.ExitCode != 0 {
			return nil, fmt.Errorf("failed to read file from remote node (exit code %d): %s", result.ExitCode, result.Stderr)
		}
		return []byte(result.Stdout), nil
	} else {
		// Local verification: read file directly
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %w", err)
		}
		return data, nil
	}
}

// verifyCAChainComposition verifies that ca-chain.crt is composed of ca.crt and trust-chain.crt (in that order)
func (v *CertificateVerifier) verifyCAChainComposition(caChainPath string) error {
	// Load trust-chain.crt certificates
	trustChainCerts, err := v.loadTrustChain()
	if err != nil {
		return fmt.Errorf("failed to load trust-chain.crt: %w", err)
	}

	// Load ca.crt certificate
	caCertPath := filepath.Join(v.pkiDir, "ca.crt")
	caCert, err := v.loadCertificate(caCertPath)
	if err != nil {
		return fmt.Errorf("failed to load ca.crt: %w", err)
	}

	// Load ca-chain.crt certificates
	caChainData, err := v.readFileContent(caChainPath)
	if err != nil {
		return fmt.Errorf("failed to read ca-chain.crt: %w", err)
	}

	var caChainCerts []*x509.Certificate
	var block *pem.Block
	rest := caChainData

	for {
		block, rest = pem.Decode(rest)
		if block == nil {
			break
		}
		if block.Type == "CERTIFICATE" {
			cert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return fmt.Errorf("failed to parse certificate in ca-chain.crt: %w", err)
			}
			caChainCerts = append(caChainCerts, cert)
		}
	}

	if len(caChainCerts) == 0 {
		return fmt.Errorf("no certificates found in ca-chain.crt")
	}

	// Expected: ca-chain.crt should contain all certificates from ca.crt + trust-chain.crt
	// Order: ca.crt (intermediate CA) first, then trust-chain.crt (root CA)
	expectedCount := 1 + len(trustChainCerts)
	if len(caChainCerts) != expectedCount {
		return fmt.Errorf("ca-chain.crt contains %d certificates, expected %d (ca.crt: 1 + trust-chain.crt: %d)", len(caChainCerts), expectedCount, len(trustChainCerts))
	}

	// Verify that ca.crt is the first certificate in ca-chain.crt
	if !caCert.Equal(caChainCerts[0]) {
		return fmt.Errorf("ca-chain.crt first certificate does not match ca.crt")
	}

	// Verify that all trust-chain.crt certificates are after ca.crt in ca-chain.crt
	for i, trustCert := range trustChainCerts {
		chainIndex := i + 1 // Start from index 1 (after ca.crt)
		if chainIndex >= len(caChainCerts) {
			return fmt.Errorf("ca-chain.crt is missing certificate %d from trust-chain.crt", i)
		}
		// Compare certificates by their raw bytes (DER encoding)
		if !trustCert.Equal(caChainCerts[chainIndex]) {
			return fmt.Errorf("ca-chain.crt certificate %d does not match trust-chain.crt certificate %d", chainIndex, i)
		}
	}

	return nil
}

// verifyAPIServerClientCAFile verifies that kube-apiserver.yaml --client-ca-file points to ca-chain.crt
func (v *CertificateVerifier) verifyAPIServerClientCAFile(yamlPath, expectedCAChainPath string) error {
	// Read kube-apiserver.yaml content
	yamlContent, err := v.readFileContent(yamlPath)
	if err != nil {
		return fmt.Errorf("failed to read kube-apiserver.yaml: %w", err)
	}

	// Parse YAML content to find --client-ca-file parameter
	// Simple string search for --client-ca-file parameter
	lines := strings.Split(string(yamlContent), "\n")
	found := false
	for _, line := range lines {
		// Look for --client-ca-file parameter
		if strings.Contains(line, "--client-ca-file") {
			found = true
			// Extract the path value
			// Format: - --client-ca-file=/etc/kubernetes/pki/ca-chain.crt
			parts := strings.Split(line, "=")
			if len(parts) >= 2 {
				actualPath := strings.TrimSpace(parts[1])
				// Remove quotes if present
				actualPath = strings.Trim(actualPath, `"'`)
				if actualPath != expectedCAChainPath {
					return fmt.Errorf("--client-ca-file points to %s, expected %s", actualPath, expectedCAChainPath)
				}
				return nil // Found and matches
			}
		}
	}

	if !found {
		return fmt.Errorf("--client-ca-file parameter not found in kube-apiserver.yaml")
	}

	return nil
}

// verifyControllerManagerKubeconfig verifies that kube-controller-manager.yaml kubeconfig parameters point to controller-manager.conf
func (v *CertificateVerifier) verifyControllerManagerKubeconfig(yamlPath, expectedKubeconfigPath string) []ResultItem {
	items := []ResultItem{}

	// Read kube-controller-manager.yaml content
	yamlContent, err := v.readFileContent(yamlPath)
	if err != nil {
		return []ResultItem{{
			Name:        "kube-controller-manager.yaml kubeconfig verification",
			Success:     false,
			Description: "Failed to read kube-controller-manager.yaml",
			Error:       err,
		}}
	}

	// Parse YAML content to find kubeconfig parameters
	lines := strings.Split(string(yamlContent), "\n")
	params := map[string]string{
		"--kubeconfig":                "",
		"--authentication-kubeconfig": "",
		"--authorization-kubeconfig":  "",
	}

	for _, line := range lines {
		for param := range params {
			if strings.Contains(line, param) {
				// Extract the path value
				// Format: - --kubeconfig=/etc/kubernetes/controller-manager.conf
				parts := strings.Split(line, "=")
				if len(parts) >= 2 {
					actualPath := strings.TrimSpace(parts[1])
					// Remove quotes if present
					actualPath = strings.Trim(actualPath, `"'`)
					params[param] = actualPath
					break // Found this parameter, move to next line
				}
			}
		}
	}

	// Check if all parameters are found and correct
	for param, actualPath := range params {
		if actualPath == "" {
			items = append(items, ResultItem{
				Name:        fmt.Sprintf("kube-controller-manager.yaml %s verification", param),
				Success:     false,
				Description: fmt.Sprintf("%s parameter not found", param),
				Error:       fmt.Errorf("%s parameter not found", param),
			})
		} else if actualPath != expectedKubeconfigPath {
			items = append(items, ResultItem{
				Name:        fmt.Sprintf("kube-controller-manager.yaml %s verification", param),
				Success:     false,
				Description: fmt.Sprintf("%s points to %s, expected %s", param, actualPath, expectedKubeconfigPath),
				Error:       fmt.Errorf("%s points to %s, expected %s", param, actualPath, expectedKubeconfigPath),
			})
		} else {
			items = append(items, ResultItem{
				Name:        fmt.Sprintf("kube-controller-manager.yaml %s verification", param),
				Success:     true,
				Description: fmt.Sprintf("%s correctly points to %s", param, expectedKubeconfigPath),
			})
		}
	}

	return items
}

// verifySchedulerKubeconfig verifies that kube-scheduler.yaml kubeconfig parameters point to scheduler.conf
func (v *CertificateVerifier) verifySchedulerKubeconfig(yamlPath, expectedKubeconfigPath string) []ResultItem {
	items := []ResultItem{}

	// Read kube-scheduler.yaml content
	yamlContent, err := v.readFileContent(yamlPath)
	if err != nil {
		return []ResultItem{{
			Name:        "kube-scheduler.yaml kubeconfig verification",
			Success:     false,
			Description: "Failed to read kube-scheduler.yaml",
			Error:       err,
		}}
	}

	// Parse YAML content to find kubeconfig parameters
	lines := strings.Split(string(yamlContent), "\n")
	params := map[string]string{
		"--kubeconfig":                "",
		"--authentication-kubeconfig": "",
		"--authorization-kubeconfig":  "",
	}

	for _, line := range lines {
		for param := range params {
			if strings.Contains(line, param) {
				// Extract the path value
				// Format: - --kubeconfig=/etc/kubernetes/scheduler.conf
				parts := strings.Split(line, "=")
				if len(parts) >= 2 {
					actualPath := strings.TrimSpace(parts[1])
					// Remove quotes if present
					actualPath = strings.Trim(actualPath, `"'`)
					params[param] = actualPath
					break // Found this parameter, move to next line
				}
			}
		}
	}

	// Check if all parameters are found and correct
	for param, actualPath := range params {
		if actualPath == "" {
			items = append(items, ResultItem{
				Name:        fmt.Sprintf("kube-scheduler.yaml %s verification", param),
				Success:     false,
				Description: fmt.Sprintf("%s parameter not found", param),
				Error:       fmt.Errorf("%s parameter not found", param),
			})
		} else if actualPath != expectedKubeconfigPath {
			items = append(items, ResultItem{
				Name:        fmt.Sprintf("kube-scheduler.yaml %s verification", param),
				Success:     false,
				Description: fmt.Sprintf("%s points to %s, expected %s", param, actualPath, expectedKubeconfigPath),
				Error:       fmt.Errorf("%s points to %s, expected %s", param, actualPath, expectedKubeconfigPath),
			})
		} else {
			items = append(items, ResultItem{
				Name:        fmt.Sprintf("kube-scheduler.yaml %s verification", param),
				Success:     true,
				Description: fmt.Sprintf("%s correctly points to %s", param, expectedKubeconfigPath),
			})
		}
	}

	return items
}

// verifyKubeletKubeconfig verifies that kubelet --kubeconfig parameter points to kubelet.conf (via systemctl status)
func (v *CertificateVerifier) verifyKubeletKubeconfig(expectedKubeconfigPath string) []ResultItem {
	items := []ResultItem{}

	if v.nodeIP == "" {
		// Local verification: try to read systemctl status directly (may not work)
		return []ResultItem{{
			Name:        "kubelet --kubeconfig verification",
			Success:     false,
			Description: "Remote verification required for kubelet systemctl status",
			Error:       fmt.Errorf("remote verification required"),
		}}
	}

	// Get kubelet status via systemctl
	result, err := v.executeOnRemoteNode("systemctl status kubelet 2>&1")
	if err != nil {
		return []ResultItem{{
			Name:        "kubelet --kubeconfig verification",
			Success:     false,
			Description: "Failed to get kubelet status",
			Error:       fmt.Errorf("failed to get kubelet status: %w, stderr: %s", err, result.Stderr),
		}}
	}

	if result.ExitCode != 0 {
		return []ResultItem{{
			Name:        "kubelet --kubeconfig verification",
			Success:     false,
			Description: "Failed to get kubelet status",
			Error:       fmt.Errorf("systemctl status kubelet failed (exit code %d): %s", result.ExitCode, result.Stderr),
		}}
	}

	// Search for --kubeconfig parameter in the output
	output := result.Stdout
	lines := strings.Split(output, "\n")
	found := false
	for _, line := range lines {
		if strings.Contains(line, "--kubeconfig=") {
			found = true
			// Extract the path value
			// Format: --kubeconfig=/etc/kubernetes/kubelet.conf
			parts := strings.Split(line, "--kubeconfig=")
			if len(parts) >= 2 {
				// Get the path (may be followed by spaces or other arguments)
				actualPath := strings.Fields(parts[1])[0] // Get first word after = (may have spaces or other args)
				actualPath = strings.Trim(actualPath, `"'`)
				if actualPath == expectedKubeconfigPath {
					items = append(items, ResultItem{
						Name:        "kubelet --kubeconfig verification",
						Success:     true,
						Description: fmt.Sprintf("--kubeconfig correctly points to %s", expectedKubeconfigPath),
					})
				} else {
					items = append(items, ResultItem{
						Name:        "kubelet --kubeconfig verification",
						Success:     false,
						Description: fmt.Sprintf("--kubeconfig points to %s, expected %s", actualPath, expectedKubeconfigPath),
						Error:       fmt.Errorf("--kubeconfig points to %s, expected %s", actualPath, expectedKubeconfigPath),
					})
				}
				break
			}
		}
	}
	if !found {
		items = append(items, ResultItem{
			Name:        "kubelet --kubeconfig verification",
			Success:     false,
			Description: "--kubeconfig parameter not found in kubelet status",
			Error:       fmt.Errorf("--kubeconfig parameter not found in systemctl status kubelet output"),
		})
	}

	return items
}

func (v *CertificateVerifier) executeOnRemoteNode(command string) (*executor.ExecResult, error) {
	node := config.NodeInfo{
		IP:       v.nodeIP,
		Port:     "22",
		Username: v.nodeUser,
		Password: v.nodePass,
	}
	return ExecuteCommandOnNode(node, command)
}

// VerifyKubeconfigMount 只验证 kubeconfig 挂载（不验证证书）
// 验证 kubelet、controller-manager 和 scheduler 是否挂载了对应的 kubeconfig
func (v *CertificateVerifier) VerifyKubeconfigMount() (*VerificationResult, error) {
	items := []ResultItem{}

	// 1. Check kube-controller-manager.yaml kubeconfig parameters
	controllerManagerYAMLPath := "/etc/kubernetes/manifests/kube-controller-manager.yaml"
	controllerManagerKubeconfigPath := "/etc/kubernetes/controller-manager.conf"
	controllerManagerItems := v.verifyControllerManagerKubeconfig(controllerManagerYAMLPath, controllerManagerKubeconfigPath)
	items = append(items, controllerManagerItems...)

	// 2. Check kube-scheduler.yaml kubeconfig parameters
	schedulerYAMLPath := "/etc/kubernetes/manifests/kube-scheduler.yaml"
	schedulerKubeconfigPath := "/etc/kubernetes/scheduler.conf"
	schedulerItems := v.verifySchedulerKubeconfig(schedulerYAMLPath, schedulerKubeconfigPath)
	items = append(items, schedulerItems...)

	// 3. Check kubelet kubeconfig parameter (via systemctl status)
	kubeletKubeconfigPath := "/etc/kubernetes/kubelet.conf"
	kubeletItems := v.verifyKubeletKubeconfig(kubeletKubeconfigPath)
	items = append(items, kubeletItems...)

	return &VerificationResult{
		Category: "Kubeconfig Mount Verification",
		Items:    items,
	}, nil
}

// PrintResults prints verification results
func (v *CertificateVerifier) PrintResults(results *VerificationResult) {
	fmt.Printf("=== %s ===\n", results.Category)
	fmt.Println()

	// Print grouped by category
	currentCategory := ""
	for _, item := range results.Items {
		// Extract category from name (e.g., "Root CA → cluster CA" category is "Global CA Issuance Verification")
		category := v.extractCategory(item.Name)
		if category != currentCategory {
			if currentCategory != "" {
				fmt.Println()
			}
			currentCategory = category
			fmt.Printf("%s:\n", category)
		}

		if item.Success {
			fmt.Printf("   ✓ %s\n", item.Name)
		} else {
			fmt.Printf("   ✗ %s", item.Name)
			if item.Error != nil {
				fmt.Printf(" (%s)", item.Error.Error())
			}
			fmt.Println()
		}
	}
	fmt.Println()
}

// extractCategory extracts category from verification item name
func (v *CertificateVerifier) extractCategory(name string) string {
	if strings.Contains(name, "Root CA →") {
		return "1. Global CA Issuance Verification"
	}
	if strings.Contains(name, "Cluster CA →") {
		return "2. Cluster CA Issues Component Certificates"
	}
	if strings.Contains(name, "front-proxy-ca →") {
		return "3. FrontProxy CA Issues Client Certificate"
	}
	if strings.Contains(name, "etcd-ca →") {
		return "4. ETCD CA Issues ETCD Component Certificates"
	}
	if strings.Contains(name, "CSR field match") {
		return "6. CSR Field Consistency Verification"
	}
	if strings.Contains(name, "ca-chain.crt") || strings.Contains(name, "kube-apiserver.yaml") || strings.Contains(name, "kube-controller-manager.yaml") || strings.Contains(name, "kube-scheduler.yaml") || strings.Contains(name, "kubelet") {
		return "5. CA Chain Certificate Verification"
	}
	return "Other Verification"
}
