package pkiutil

import (
	"crypto/x509"
	"net"
	"time"

	certutil "k8s.io/client-go/util/cert"

	"gopkg.openfuyao.cn/bkeagent/utils"
)

var (
	localhost = "127.0.0.1"
)

// BKECert represents a TLS certificate managed by bkeagent for secure communications.
type BKECert struct {
	Name     string // Short identifier for the certificate.
	LongName string // Detailed description for human readability.
	BaseName string // Base filename for certificate and key files.
	CAName   string // Name of the Certificate Authority used for signing.
	PkiPath  string // Directory path to store the certificate.
	IsCA     bool   // Indicates whether this certificate is a CA.
	Config   CertConfig
}

// CertConfig extends certutil.Config by adding key algorithm, validity, and key usage fields.
type CertConfig struct {
	certutil.Config
	BaseUsages         []x509.KeyUsage
	Validity           time.Duration
	PublicKeyAlgorithm x509.PublicKeyAlgorithm

	KeySize            int
	Country            []string // Country (C)
	Province           []string // State or Province (ST)
	Locality           []string // Locality or City (L)
	OrganizationalUnit []string // Organizational Unit (OU)
}

// Certificates is a slice of BKECert.
type Certificates []*BKECert

// SetPkiPath sets the PKI path for all certificates in the collection.
func (c *Certificates) SetPkiPath(pkiPath string) {
	for _, cert := range *c {
		cert.PkiPath = pkiPath
	}
}

// Export returns a list of certificate and key file paths that should be generated.
func (c *Certificates) Export(pkiDirPath string) []string {
	c.SetPkiPath(pkiDirPath)
	var files []string
	for _, cert := range *c {
		if cert.BaseName == ServiceAccountKeyBaseName {
			files = append(files, pathForPublicKey(cert))
			files = append(files, pathForKey(cert))
			continue
		}
		files = append(files, pathForCert(cert))
		files = append(files, pathForKey(cert))
	}
	return files
}

// getCommonCoreCerts returns the common set of certificates used by multiple configurations.
// This includes root CA, API server, kubelet client, front proxy, and etcd certificates.
func getCommonCoreCerts() Certificates {
	return Certificates{
		BKECertRootCA(),
		BKECertAPIServer(),
		BKECertKubeletClient(),
		BKECertFrontProxyCA(),
		BKECertFrontProxyClient(),
		BKECertEtcdCA(),
		BKECertEtcdServer(),
		BKECertEtcdPeer(),
		BKECertEtcdHealthcheck(),
		BKECertEtcdAPIClient(),
	}
}

// GetDefaultCertList returns the default list of certificates to create (excluding SA and kubeconfig).
func GetDefaultCertList() Certificates {
	return getCommonCoreCerts()
}

// GetUserCustomCerts returns the list of certificates for user-defined configurations.
func GetUserCustomCerts() Certificates {
	return getCommonCoreCerts()
}

// GetTargetClusterCertList returns all certificates required for a target cluster deployment.
func GetTargetClusterCertList() Certificates {
	return Certificates{
		BKECertRootCA(),
		BKECertEtcdCA(),
		BKECertFrontProxyCA(),
		BKECertServiceAccount(),
		BKECertAPIServer(),
		BKECertKubeletClient(),
		BKECertFrontProxyClient(),
		BKECertEtcdServer(),
		BKECertEtcdPeer(),
		BKECertEtcdHealthcheck(),
		BKECertEtcdAPIClient(),
	}
}

// GetClusterAPICertList returns all certificates required by the Cluster API.
func GetClusterAPICertList() Certificates {
	return Certificates{
		BKECertRootCA(),
		BKECertEtcdCA(),
		BKECertFrontProxyCA(),
		BKECertServiceAccount(),
		BKEAdminKubeConfig(),
		BKECertAPIServer(),
		BKECertKubeletClient(),
		BKECertFrontProxyClient(),
		BKECertEtcdServer(),
		BKECertEtcdPeer(),
		BKECertEtcdHealthcheck(),
		BKECertEtcdAPIClient(),
	}
}

// GetCertsWithoutCA returns all non-CA certificates.
func GetCertsWithoutCA() Certificates {
	return Certificates{
		BKECertAPIServer(),
		BKECertKubeletClient(),
		BKECertFrontProxyClient(),
		BKECertEtcdServer(),
		BKECertEtcdPeer(),
		BKECertEtcdHealthcheck(),
		BKECertEtcdAPIClient(),
	}
}

// GetCACerts returns all CA certificates.
func GetCACerts() Certificates {
	return Certificates{
		BKECertRootCA(),
		BKECertEtcdCA(),
		BKECertFrontProxyCA(),
	}
}

// GetCertsWithoutEtcd returns all certificates excluding etcd-related ones (for external etcd setups).
func GetCertsWithoutEtcd() Certificates {
	return Certificates{
		BKECertRootCA(),
		BKECertAPIServer(),
		BKECertKubeletClient(),
		BKECertServiceAccount(),
		BKECertFrontProxyCA(),
		BKECertFrontProxyClient(),
	}
}

// GetEtcdCerts returns all etcd-related certificates.
func GetEtcdCerts() Certificates {
	return Certificates{
		BKECertEtcdCA(),
		BKECertEtcdServer(),
		BKECertEtcdPeer(),
		BKECertEtcdHealthcheck(),
		BKECertEtcdAPIClient(),
	}
}

// GetKubeConfigs returns all kubeconfig-related certificates.
func GetKubeConfigs() Certificates {
	return Certificates{
		BKEAdminKubeConfig(),
		BKEKubeletKubeConfig(),
		BKEControllerKubeConfig(),
		BKESchedulerKubeConfig(),
	}
}

// BKEAdminKubeConfig returns the admin kubeconfig certificate.
func BKEAdminKubeConfig() *BKECert {
	return &BKECert{
		Name:     "kubeconfig",
		LongName: "admin kubeconfig used by the superuser/admin of the cluster",
		BaseName: AdminKubeConfigFileName,
		PkiPath:  KubernetesDir,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   "kubernetes-admin",
				Organization: []string{SystemPrivilegedGroup},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKEKubeletKubeConfig returns the kubeconfig certificate for kubelet.
func BKEKubeletKubeConfig() *BKECert {
	nodeName := utils.HostName()
	return &BKECert{
		Name:     "kubelet",
		LongName: "kubeconfig for kubelet",
		BaseName: KubeletKubeConfigFileName,
		PkiPath:  KubernetesDir,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   "system:node:" + nodeName,
				Organization: []string{NodesGroup},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKEControllerKubeConfig returns the kubeconfig certificate for the controller-manager.
func BKEControllerKubeConfig() *BKECert {
	return &BKECert{
		Name:     "controller-manager",
		LongName: "kubeconfig for controller-manager",
		BaseName: ControllerManagerKubeConfigFileName,
		PkiPath:  KubernetesDir,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: ControllerManagerUser,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKESchedulerKubeConfig returns the kubeconfig certificate for the scheduler.
func BKESchedulerKubeConfig() *BKECert {
	return &BKECert{
		Name:     "scheduler",
		LongName: "kubeconfig for scheduler",
		BaseName: SchedulerKubeConfigFileName,
		PkiPath:  KubernetesDir,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: SchedulerUser,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKECertServiceAccount returns the certificate used for service account token signing.
func BKECertServiceAccount() *BKECert {
	return &BKECert{
		Name:     "sa",
		LongName: "certificate for the service account token issuer",
		BaseName: ServiceAccountKeyBaseName,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: ServiceAccountKeyBaseName,
			},
		},
	}
}

// BKECertRootCA returns the Root CA certificate.
func BKECertRootCA() *BKECert {
	return &BKECert{
		Name:     "ca",
		LongName: "Root Certificate Authority",
		BaseName: CACertAndKeyBaseName,
		Config: CertConfig{
			Validity: CertificateValidity,
			Config: certutil.Config{
				CommonName: CACertCommonName,
			},
		},
	}
}

// BKECertAPIServer returns the certificate used by the Kubernetes API server.
func BKECertAPIServer() *BKECert {
	return &BKECert{
		Name:     "apiserver",
		LongName: "certificate for serving the Kubernetes API",
		BaseName: APIServerCertAndKeyBaseName,
		CAName:   "ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: APIServerCertCommonName,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				AltNames: certutil.AltNames{
					DNSNames: []string{
						"master.bocloud.com",
						"kubernetes",
						"kubernetes.default",
						"kubernetes.default.svc",
						"kubernetes.default.svc.cluster",
					},
					IPs: []net.IP{
						net.IPv4(127, 0, 0, 1),
						net.IPv6loopback,
						net.IPv6zero,
					},
				},
			},
		},
	}
}

// BKECertKubeletClient returns the certificate used by the API server to connect to kubelet.
func BKECertKubeletClient() *BKECert {
	return &BKECert{
		Name:     "apiserver-kubelet-client",
		LongName: "certificate for the API server to connect to kubelet",
		BaseName: APIServerKubeletClientCertAndKeyBaseName,
		CAName:   "ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   APIServerKubeletClientCertCommonName,
				Organization: []string{SystemPrivilegedGroup},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKECertFrontProxyCA returns the CA certificate for the front proxy.
func BKECertFrontProxyCA() *BKECert {
	return &BKECert{
		Name:     "proxy",
		LongName: "self-signed CA to provision identities for front proxy",
		BaseName: FrontProxyCACertAndKeyBaseName,
		Config: CertConfig{
			Validity: CertificateValidity,
			Config: certutil.Config{
				CommonName: "front-proxy-ca",
			},
		},
	}
}

// BKECertFrontProxyClient returns the client certificate for the front proxy.
func BKECertFrontProxyClient() *BKECert {
	return &BKECert{
		Name:     "front-proxy-client",
		BaseName: FrontProxyClientCertAndKeyBaseName,
		LongName: "certificate for the front proxy client",
		CAName:   "front-proxy-ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: FrontProxyClientCertCommonName,
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKECertEtcdCA returns the self-signed CA certificate used by etcd.
func BKECertEtcdCA() *BKECert {
	return &BKECert{
		Name:     "etcd",
		LongName: "self-signed CA to provision identities for etcd",
		BaseName: EtcdCACertAndKeyBaseName,
		Config: CertConfig{
			Validity: CertificateValidity,
			Config: certutil.Config{
				CommonName: "etcd-ca",
			},
		},
	}
}

// BKECertEtcdServer returns the server certificate for etcd.
func BKECertEtcdServer() *BKECert {
	return &BKECert{
		Name:     "etcd-server",
		LongName: "certificate for serving etcd",
		BaseName: EtcdServerCertAndKeyBaseName,
		CAName:   "etcd-ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "etcd-server",
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
				AltNames:   getEtcdAltNames(),
			},
		},
	}
}

// BKECertEtcdPeer returns the certificate for etcd peer communication.
func BKECertEtcdPeer() *BKECert {
	return &BKECert{
		Name:     "etcd-peer",
		LongName: "certificate for etcd nodes to communicate with each other",
		BaseName: EtcdPeerCertAndKeyBaseName,
		CAName:   "etcd-ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "etcd-peer",
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
				AltNames:   getEtcdAltNames(),
			},
		},
	}
}

// BKECertEtcdHealthcheck returns the certificate used by Kubernetes to perform etcd health checks.
func BKECertEtcdHealthcheck() *BKECert {
	return &BKECert{
		Name:     "etcd-healthcheck-client",
		LongName: "certificate for liveness probes to healthcheck etcd",
		BaseName: EtcdHealthcheckClientCertAndKeyBaseName,
		CAName:   "etcd-ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   EtcdHealthcheckClientCertCommonName,
				Organization: []string{SystemPrivilegedGroup},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// BKECertEtcdAPIClient returns the certificate used by the API server to access etcd.
func BKECertEtcdAPIClient() *BKECert {
	return &BKECert{
		Name:     "apiserver-etcd-client",
		LongName: "certificate for the apiserver to access etcd",
		BaseName: APIServerEtcdClientCertAndKeyBaseName,
		CAName:   "etcd-ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   APIServerEtcdClientCertCommonName,
				Organization: []string{SystemPrivilegedGroup},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}
}

// getEtcdAltNames returns common alternative names used by etcd certificates.
func getEtcdAltNames() certutil.AltNames {
	return certutil.AltNames{
		DNSNames: []string{"localhost"},
		IPs: []net.IP{
			net.ParseIP(localhost),
			net.IPv6loopback,
			net.IPv6zero,
		},
	}
}
