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
	"crypto/x509"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
)

const (
	// testHost defines an IP for test
	testHost = "192." + "168." + "1." + "100"
	// testPort defines a port for test
	testPort = 6443

	signPolicyJsonTestData = `{
	  "signing": {
		"default": {
		  "usages": ["digital signature", "key encipherment"],
		  "expiry": "8760h"
		},
		"profiles": {
		  "ca": {
			"usages": ["cert sign", "crl sign"],
			"expiry": "87600h",
			"ca_constraint": {
			  "is_ca": true,
			  "max_path_len": 0
			}
		  },
		  "apiserver": {
			"usages": ["digital signature", "key encipherment", "server auth"],
			"expiry": "8760h"
		  },
		  "apiserver-etcd-client": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "apiserver-kubelet-client": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "front-proxy-client": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "front-proxy-ca": {
			"usages": ["cert sign", "crl sign"],
			"expiry": "87600h",
			"ca_constraint": {
			  "is_ca": true,
			  "max_path_len": 1
			}
		  },
		  "etcd/ca": {
			"usages": ["cert sign", "crl sign"],
			"expiry": "87600h",
			"ca_constraint": {
			  "is_ca": true,
			  "max_path_len": 1
			}
		  },
		  "etcd/server": {
			"usages": ["digital signature", "key encipherment", "server auth", "client auth"],
			"expiry": "8760h"
		  },
		  "etcd/peer": {
			"usages": ["digital signature", "key encipherment", "server auth", "client auth"],
			"expiry": "8760h"
		  },
		  "etcd/healthcheck-client": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "controller-manager": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "scheduler": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "kubelet": {
			"usages": ["digital signature", "key encipherment", "server auth", "client auth"],
			"expiry": "8760h"
		  },
		  "admin": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  },
		  "kubeconfig": {
			"usages": ["digital signature", "key encipherment", "client auth"],
			"expiry": "8760h"
		  }
		}
	  }
	}`
)

// createTestConfigMap creates a ConfigMap with certificate configurations for testing
func createTestConfigMap() *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      CertConfigMapName,
			Namespace: CertConfigMapNamespace,
		},
		Data: map[string]string{
			ConfigKeyClusterCAPolicy: getClusterCAPolicyJSON(),
			ConfigKeyClusterCACSR:    getClusterCACSRJSON(),
			ConfigKeySignPolicy:      getSignPolicyJSON(),

			ConfigKeyAPIServerCSR:              getAPIServerCSRJSON(),
			ConfigKeyAPIServerEtcdClientCSR:    getAPIServerEtcdClientCSRJSON(),
			ConfigKeyFrontProxyClientCSR:       getFrontProxyClientCSRJSON(),
			ConfigKeyAPIServerKubeletClientCSR: getAPIServerKubeletClientCSRJSON(),

			ConfigKeyFrontProxyCACSR: getFrontProxyCACSRJSON(),

			ConfigKeyEtcdCACSR:                getEtcdCACSRJSON(),
			ConfigKeyEtcdServerCSR:            getEtcdServerCSRJSON(),
			ConfigKeyEtcdHealthcheckClientCSR: getEtcdHealthcheckClientCSRJSON(),
			ConfigKeyEtcdPeerCSR:              getEtcdPeerCSRJSON(),

			ConfigKeyAdminKubeConfigCSR:   getAdminKubeConfigCSRJSON(),
			ConfigKeyKubeletKubeConfigCSR: getKubeletKubeConfigCSRJSON(),
			ConfigKeyControllerManagerCSR: getControllerManagerCSRJSON(),
			ConfigKeySchedulerCSR:         getSchedulerCSRJSON(),
		},
	}
}

// createTestBKECluster creates a BKECluster for testing
func createTestBKECluster() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testHost,
				Port: testPort,
			},
		},
	}
}

// createTestLoader creates a test loader with fake client (without loading config data)
func createTestLoader(t *testing.T) *CertConfigLoader {

	ctx := context.Background()

	bkeCluster := createTestBKECluster()
	configMap := createTestConfigMap()

	client := fake.NewClientBuilder().
		WithObjects(configMap).
		Build()

	return NewCertConfigLoader(ctx, client, bkeCluster, nil)
}

// TestLoadConfigMapData tests loading configuration from ConfigMap
func TestLoadConfigMapData(t *testing.T) {
	// Setup
	loader := createTestLoader(t)

	// Test loading
	configData, err := loader.LoadConfigMapData()
	assert.NoError(t, err)
	assert.NotNil(t, configData)

	// Verify loaded configurations
	assert.True(t, configData.AvailableKeys[ConfigKeyClusterCAPolicy])
	assert.True(t, configData.AvailableKeys[ConfigKeyClusterCACSR])
	assert.True(t, configData.AvailableKeys[ConfigKeySignPolicy])
	assert.True(t, configData.AvailableKeys[ConfigKeyAPIServerCSR])
	assert.True(t, configData.AvailableKeys[ConfigKeyAdminKubeConfigCSR])

	// Verify policy structure
	assert.NotEmpty(t, configData.ClusterCAPolicy.Signing.Default.Usages)
	assert.Equal(t, "87600h", configData.ClusterCAPolicy.Signing.Default.Expiry)
	assert.True(t, configData.ClusterCAPolicy.Signing.Default.CAConstraint.IsCA)

	assert.NotEmpty(t, configData.SignPolicy.Signing.Default.Usages)
	assert.Equal(t, "8760h", configData.SignPolicy.Signing.Default.Expiry)

	// Verify profiles
	apiserverProfile, exists := configData.SignPolicy.Signing.Profiles["apiserver"]
	assert.True(t, exists)
	assert.Contains(t, apiserverProfile.Usages, "server auth")
}

// setupTestLoader creates a test loader with fake client
func setupTestLoader(t *testing.T) (*CertConfigLoader, *CertConfigData) {
	loader := createTestLoader(t)

	configData, err := loader.LoadConfigMapData()
	assert.NoError(t, err)
	assert.NotNil(t, configData)

	return loader, configData
}

// getTestCertificates returns all test certificates
func getTestCertificates() pkiutil.Certificates {
	bkeCerts := pkiutil.GetUserCustomCerts()
	bkeCerts = append(bkeCerts, pkiutil.BKEAdminKubeConfig())
	return bkeCerts
}

// TestApplyConfigToCerts tests applying configuration to certificates
func TestApplyConfigToCerts(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	printAllCertificates(t, bkeCerts)
}

// printAllCertificates prints all certificate configurations
func printAllCertificates(t *testing.T, certs pkiutil.Certificates) {
	t.Log("\n=== All Certificate Configurations ===")
	for i := range certs {
		cert := certs[i]
		t.Logf("\nCertificate #%d: %s (%s)", i+1, cert.Name, cert.BaseName)
		t.Logf("  CN: %s", cert.Config.Config.CommonName)
		t.Logf("  O: %v", cert.Config.Config.Organization)
		t.Logf("  C: %v", cert.Config.Country)
		t.Logf("  ST: %v", cert.Config.Province)
		t.Logf("  L: %v", cert.Config.Locality)
		t.Logf("  OU: %v", cert.Config.OrganizationalUnit)
		t.Logf("  IsCA: %v", cert.IsCA)
		t.Logf("  KeySize: %d", cert.Config.KeySize)
		t.Logf("  PublicKeyAlgorithm: %v", cert.Config.PublicKeyAlgorithm)
		t.Logf("  Validity: %v", cert.Config.Validity)
		t.Logf("  KeyUsages: %v", cert.Config.BaseUsages)
		t.Logf("  ExtKeyUsage: %v", cert.Config.Config.Usages)
		t.Logf("  DNSNames: %v", cert.Config.AltNames.DNSNames)
		t.Logf("  IPs: %v", cert.Config.AltNames.IPs)
	}
	t.Log("=== End of Certificate Configurations ===\n")
}

// findCertByName finds a certificate by BaseName
func findCertByName(certs pkiutil.Certificates, name string) *pkiutil.BKECert {
	for i := range certs {
		if certs[i].BaseName == name {
			return certs[i]
		}
	}
	return nil
}

// TestCACertificate tests Cluster CA certificate configuration
func TestCACertificate(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	caCert := findCertByName(bkeCerts, pkiutil.CACertAndKeyBaseName)
	if caCert != nil {
		assert.Equal(t, "kubernetes", caCert.Config.Config.CommonName)
		assert.Equal(t, []string{"system:masters"}, caCert.Config.Config.Organization)
		assert.Contains(t, caCert.Config.AltNames.DNSNames, "kubernetes")
	}
}

// TestAPIServerCertificate tests API Server certificate configuration
func TestAPIServerCertificate(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	apiserverCert := findCertByName(bkeCerts, pkiutil.APIServerCertAndKeyBaseName)
	if apiserverCert != nil {
		assert.Equal(t, "kube-apiserver", apiserverCert.Config.Config.CommonName)
		assert.Contains(t, apiserverCert.Config.AltNames.DNSNames, "test-cluster")
		assert.Contains(t, apiserverCert.Config.Config.Usages, x509.ExtKeyUsageServerAuth)
	}
}

// TestComponentCertificates tests component certificates
func TestComponentCertificates(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	// Controller Manager
	cmCert := findCertByName(bkeCerts, pkiutil.ControllerManagerCertAndKeyBaseName)
	if cmCert != nil {
		assert.Equal(t, "system:kube-controller-manager", cmCert.Config.Config.CommonName)
	}

	// Scheduler
	schedulerCert := findCertByName(bkeCerts, pkiutil.SchedulerCertAndKeyBaseName)
	if schedulerCert != nil {
		assert.Equal(t, "system:kube-scheduler", schedulerCert.Config.Config.CommonName)
	}
}

// TestClientCertificates tests client certificates
func TestClientCertificates(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	// API Server etcd client
	etcdClientCert := findCertByName(bkeCerts, pkiutil.APIServerEtcdClientCertAndKeyBaseName)
	if etcdClientCert != nil {
		assert.Equal(t, "kube-apiserver-etcd-client", etcdClientCert.Config.Config.CommonName)
		assert.Contains(t, etcdClientCert.Config.Config.Usages, x509.ExtKeyUsageClientAuth)
	}

	// Front Proxy Client
	frontProxyCert := findCertByName(bkeCerts, pkiutil.FrontProxyClientCertAndKeyBaseName)
	if frontProxyCert != nil {
		assert.Equal(t, "front-proxy-client", frontProxyCert.Config.Config.CommonName)
		assert.Contains(t, frontProxyCert.Config.Config.Usages, x509.ExtKeyUsageClientAuth)
	}
}

// TestKubeConfigCertificates tests kubeconfig certificates
func TestKubeConfigCertificates(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	// Admin KubeConfig
	adminKubeConfigCert := findCertByName(bkeCerts, pkiutil.AdminKubeConfigFileName)
	if adminKubeConfigCert != nil {
		assert.Equal(t, "kubernetes-admin", adminKubeConfigCert.Config.Config.CommonName)
		assert.Contains(t, adminKubeConfigCert.Config.Config.Usages, x509.ExtKeyUsageClientAuth)
	}
}

// TestEtcdCertificates tests etcd certificates
func TestEtcdCertificates(t *testing.T) {
	loader, configData := setupTestLoader(t)
	bkeCerts := getTestCertificates()

	err := loader.ApplyConfigToCerts(bkeCerts, configData, "test-cluster")
	assert.NoError(t, err)

	// etcd Server
	etcdServerCert := findCertByName(bkeCerts, pkiutil.EtcdServerCertAndKeyBaseName)
	if etcdServerCert != nil {
		assert.Equal(t, "etcd-server", etcdServerCert.Config.Config.CommonName)
		assert.Contains(t, etcdServerCert.Config.Config.Usages, x509.ExtKeyUsageServerAuth)
	}

	// etcd Healthcheck Client
	etcdHealthcheckCert := findCertByName(bkeCerts, pkiutil.EtcdHealthcheckClientCertAndKeyBaseName)
	if etcdHealthcheckCert != nil {
		assert.Equal(t, "kube-etcd-healthcheck-client", etcdHealthcheckCert.Config.Config.CommonName)
	}

	// etcd Peer
	etcdPeerCert := findCertByName(bkeCerts, pkiutil.EtcdPeerCertAndKeyBaseName)
	if etcdPeerCert != nil {
		assert.Equal(t, "etcd-peer", etcdPeerCert.Config.Config.CommonName)
	}
}

// getClusterCAPolicyJSON returns cluster CA policy configuration
func getClusterCAPolicyJSON() string {
	return `{
		"signing": {
			"default": {
				"usages": ["cert sign", "crl sign"],
				"expiry": "87600h",
				"ca_constraint": {
					"is_ca": true,
					"max_path_len": 0
				}
			},
			"profiles": {
				"ca": {
					"usages": ["cert sign", "crl sign"],
					"expiry": "87600h",
					"ca_constraint": {
						"is_ca": true,
						"max_path_len_zero": true
					}
				}
			}
		}
	}`
}

// getSignPolicyJSON returns signing policy configuration
func getSignPolicyJSON() string {
	return signPolicyJsonTestData
}

// getClusterCACSRJSON returns cluster CA CSR configuration
func getClusterCACSRJSON() string {
	return `{
		"CN": "kubernetes",
		"O": "system:masters",
		"C": "CN-cluster-ca",
		"ST": "Beijing-cluster-ca",
		"L": "Beijing-cluster-ca",
		"OU": "Kubernetes-cluster-ca",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": ["kubernetes", "kubernetes.default"]
	}`
}

// getAPIServerCSRJSON returns API server CSR configuration
func getAPIServerCSRJSON() string {
	return `{
		"CN": "kube-apiserver",
		"O": "system:masters",
		"C": "CN",
		"ST": "Beijing",
		"L": "Beijing",
		"OU": "Kubernetes",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": [
			"kubernetes",
			"kubernetes.default",
			"kubernetes.default.svc",
			"kubernetes.default.svc.cluster.local",
			"10.0.0.1",
			"{{.ClusterName}}",
			"{{.AdvertiseAddress}}"
		]
	}`
}

// getAPIServerEtcdClientCSRJSON returns API server etcd client CSR configuration
func getAPIServerEtcdClientCSRJSON() string {
	return `{
		"CN": "kube-apiserver-etcd-client",
		"O": "system:masters",
		"C": "CN-api-server-etcd-client",
		"ST": "Beijing-api-server-etcd-client",
		"L": "Beijing-api-server-etcd-client",
		"OU": "Kubernetes-api-server-etcd-client",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getFrontProxyClientCSRJSON returns front proxy client CSR configuration
func getFrontProxyClientCSRJSON() string {
	return `{
		"CN": "front-proxy-client",
		"O": "system:masters",
		"C": "CN-front-proxy-client",
		"ST": "Beijing-front-proxy-client",
		"L": "Beijing-front-proxy-client",
		"OU": "Kubernetes-front-proxy-client",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getAPIServerKubeletClientCSRJSON returns API server kubelet client CSR configuration
func getAPIServerKubeletClientCSRJSON() string {
	return `{
		"CN": "kube-apiserver-kubelet-client",
		"O": "system:masters",
		"C": "CN-api-server-kubelet-client",
		"ST": "Beijing-api-server-kubelet-client",
		"L": "Beijing-api-server-kubelet-client",
		"OU": "Kubernetes-api-server-kubelet-client",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getControllerManagerCSRJSON returns controller manager CSR configuration
func getControllerManagerCSRJSON() string {
	return `{
		"CN": "system:kube-controller-manager",
		"O": "system:kube-controller-manager",
		"C": "CN-controller-manager",
		"ST": "Beijing-controller-manager",
		"L": "Beijing-controller-manager",
		"OU": "Kubernetes-controller-manager",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getSchedulerCSRJSON returns scheduler CSR configuration
func getSchedulerCSRJSON() string {
	return `{
		"CN": "system:kube-scheduler",
		"O": "system:kube-scheduler",
		"C": "CN-scheduler",
		"ST": "Beijing-scheduler",
		"L": "Beijing-scheduler",
		"OU": "Kubernetes-scheduler",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getFrontProxyCACSRJSON returns front proxy CA CSR configuration
func getFrontProxyCACSRJSON() string {
	return `{
		"CN": "front-proxy-ca",
		"O": "system:masters",
		"C": "CN-front-proxy-ca",
		"ST": "Beijing-front-proxy-ca",
		"L": "Beijing-front-proxy-ca",
		"OU": "Kubernetes-front-proxy-ca",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getEtcdCACSRJSON returns etcd CA CSR configuration
func getEtcdCACSRJSON() string {
	return `{
		"CN": "etcd-ca",
		"O": "system:masters",
		"C": "CN-etcd-ca",
		"ST": "Beijing-etcd-ca",
		"L": "Beijing-etcd-ca",
		"OU": "Kubernetes-etcd-ca",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getEtcdServerCSRJSON returns etcd server CSR configuration
func getEtcdServerCSRJSON() string {
	return `{
		"CN": "etcd-server",
		"O": "system:masters",
		"C": "CN-etcd-server",
		"ST": "Beijing-etcd-server",
		"L": "Beijing-etcd-server",
		"OU": "Kubernetes-etcd-server",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": ["localhost", "127.0.0.1"]
	}`
}

// getEtcdHealthcheckClientCSRJSON returns etcd healthcheck client CSR configuration
func getEtcdHealthcheckClientCSRJSON() string {
	return `{
		"CN": "kube-etcd-healthcheck-client",
		"O": "system:masters",
		"C": "CN-etcd-healthcheck-client",
		"ST": "Beijing-etcd-healthcheck-client",
		"L": "Beijing-etcd-healthcheck-client",
		"OU": "Kubernetes-etcd-healthcheck-client",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getEtcdPeerCSRJSON returns etcd peer CSR configuration
func getEtcdPeerCSRJSON() string {
	return `{
		"CN": "etcd-peer",
		"O": "system:masters",
		"C": "CN-etcd-peer",
		"ST": "Beijing-etcd-peer",
		"L": "Beijing-etcd-peer",
		"OU": "Kubernetes-etcd-peer",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getAdminKubeConfigCSRJSON returns admin kubeconfig CSR configuration
func getAdminKubeConfigCSRJSON() string {
	return `{
		"CN": "kubernetes-admin",
		"O": "system:masters",
		"C": "CN-admin",
		"ST": "Beijing-admin",
		"L": "Beijing-admin",
		"OU": "Kubernetes-admin",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// getKubeletKubeConfigCSRJSON returns kubelet kubeconfig CSR configuration
func getKubeletKubeConfigCSRJSON() string {
	return `{
		"CN": "system:node:test-node",
		"O": "system:nodes",
		"C": "CN-Kubelet",
		"ST": "Beijing-Kubelet",
		"L": "Beijing-Kubelet",
		"OU": "Kubernetes-Kubelet",
		"key": {
			"algo": "rsa",
			"size": 2048
		},
		"hosts": []
	}`
}

// TestLoadConfigMapData_ConfigMapNotFound tests LoadConfigMapData when ConfigMap does not exist
func TestLoadConfigMapData_ConfigMapNotFound(t *testing.T) {

	ctx := context.Background()

	bkeCluster := createTestBKECluster()
	// Create client without ConfigMap to simulate NotFound error
	client := fake.NewClientBuilder().Build()

	loader := NewCertConfigLoader(ctx, client, bkeCluster, nil)

	configData, err := loader.LoadConfigMapData()
	assert.NoError(t, err)
	assert.NotNil(t, configData)
	assert.NotNil(t, configData.AvailableKeys)
	assert.Empty(t, configData.AvailableKeys)
}

// TestGetCertConfigMap_NotFoundError tests getCertConfigMap when ConfigMap is not found
func TestGetCertConfigMap_NotFoundError(t *testing.T) {

	ctx := context.Background()

	bkeCluster := createTestBKECluster()
	// Create client without ConfigMap
	client := fake.NewClientBuilder().Build()

	loader := NewCertConfigLoader(ctx, client, bkeCluster, nil)

	configMap, err := loader.getCertConfigMap()
	assert.Error(t, err)
	assert.Nil(t, configMap)
	assert.Contains(t, err.Error(), "not found")
}

// TestGetCertConfigMap_OtherError tests getCertConfigMap when client returns non-NotFound error
func TestGetCertConfigMap_OtherError(t *testing.T) {

	ctx := context.Background()

	bkeCluster := createTestBKECluster()
	// Create error client that returns error on Get
	errorClient := &errorClient{
		Client: fake.NewClientBuilder().Build(),
	}

	loader := NewCertConfigLoader(ctx, errorClient, bkeCluster, nil)

	configMap, err := loader.getCertConfigMap()
	assert.Error(t, err)
	assert.Nil(t, configMap)
	assert.Contains(t, err.Error(), "failed to get ConfigMap")
}

// errorClient is a test client that returns error on Get operation
type errorClient struct {
	client.Client
}

// Get always returns an error for testing error paths
func (c *errorClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	return errors.New("simulated client error")
}

// TestBuildConfigMapData tests the buildConfigMapData function
func TestBuildConfigMapData(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)
	tests := getBuildConfigMapDataTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := loader.buildConfigMapData(tt.configData)
			verifyBuildConfigMapDataResult(t, result, tt.expectedKeys)
		})
	}
}

// getBuildConfigMapDataTestCases returns test cases for buildConfigMapData
func getBuildConfigMapDataTestCases() []buildConfigMapDataTestCase {
	return []buildConfigMapDataTestCase{
		createAllKeysAvailableTestCase(),
		createPartialKeysAvailableTestCase(),
		createNoKeysAvailableTestCase(),
		createNilConfigDataTestCase(),
	}
}

// buildConfigMapDataTestCase represents a test case for buildConfigMapData
type buildConfigMapDataTestCase struct {
	name         string
	configData   *CertConfigData
	expectedKeys []string
}

// createAllKeysAvailableTestCase creates a test case with all keys available
func createAllKeysAvailableTestCase() buildConfigMapDataTestCase {
	configData := createConfigDataWithAllKeys()
	expectedKeys := getAllConfigKeys()
	return buildConfigMapDataTestCase{
		name:         "all keys available",
		configData:   configData,
		expectedKeys: expectedKeys,
	}
}

// createPartialKeysAvailableTestCase creates a test case with partial keys available
func createPartialKeysAvailableTestCase() buildConfigMapDataTestCase {
	configData := createConfigDataWithPartialKeys()
	expectedKeys := []string{ConfigKeyClusterCAPolicy, ConfigKeySignPolicy}
	return buildConfigMapDataTestCase{
		name:         "partial keys available",
		configData:   configData,
		expectedKeys: expectedKeys,
	}
}

// createNoKeysAvailableTestCase creates a test case with no keys available
func createNoKeysAvailableTestCase() buildConfigMapDataTestCase {
	configData := createConfigDataWithNoKeys()
	return buildConfigMapDataTestCase{
		name:         "no keys available",
		configData:   configData,
		expectedKeys: []string{},
	}
}

// createNilConfigDataTestCase creates a test case with nil config data
func createNilConfigDataTestCase() buildConfigMapDataTestCase {
	return buildConfigMapDataTestCase{
		name:         "nil config data",
		configData:   &CertConfigData{AvailableKeys: nil},
		expectedKeys: []string{},
	}
}

// createConfigDataWithAllKeys creates a CertConfigData with all keys available
func createConfigDataWithAllKeys() *CertConfigData {
	configData := createBaseConfigData()
	markAllKeysAsAvailable(configData)
	return configData
}

// createConfigDataWithPartialKeys creates a CertConfigData with partial keys available
func createConfigDataWithPartialKeys() *CertConfigData {
	configData := createBaseConfigData()
	configData.AvailableKeys[ConfigKeyClusterCAPolicy] = true
	configData.AvailableKeys[ConfigKeySignPolicy] = true
	return configData
}

// createConfigDataWithNoKeys creates a CertConfigData with no keys available
func createConfigDataWithNoKeys() *CertConfigData {
	configData := createBaseConfigData()
	markAllKeysAsUnavailable(configData)
	return configData
}

// createBaseConfigData creates a base CertConfigData with sample data
func createBaseConfigData() *CertConfigData {
	return &CertConfigData{
		ClusterCAPolicy: createTestCertPolicy(),
		ClusterCACSR:    createTestCertCSR(),
		SignPolicy:      createTestCertPolicy(),
		APIServerCSR:    createTestCertCSR(),
		AvailableKeys:   make(map[string]bool),
	}
}

// createTestCertPolicy creates a test CertPolicy
func createTestCertPolicy() pkiutil.CertPolicy {
	policy := pkiutil.CertPolicy{}
	policy.Signing.Default.Usages = []string{"digital signature"}
	policy.Signing.Default.Expiry = "8760h"
	return policy
}

// createTestCertCSR creates a test CertCSR
func createTestCertCSR() pkiutil.CertCSR {
	return pkiutil.CertCSR{
		CN: "test-cn",
		Key: struct {
			Algo string `json:"algo"`
			Size int    `json:"size"`
		}{
			Algo: "rsa",
			Size: 2048,
		},
	}
}

// markAllKeysAsAvailable marks all config keys as available
func markAllKeysAsAvailable(configData *CertConfigData) {
	allKeys := getAllConfigKeys()
	for _, key := range allKeys {
		configData.AvailableKeys[key] = true
	}
}

// markAllKeysAsUnavailable marks all config keys as unavailable
func markAllKeysAsUnavailable(configData *CertConfigData) {
	allKeys := getAllConfigKeys()
	for _, key := range allKeys {
		configData.AvailableKeys[key] = false
	}
}

// getAllConfigKeys returns all configuration keys
func getAllConfigKeys() []string {
	return []string{
		ConfigKeyClusterCAPolicy,
		ConfigKeyClusterCACSR,
		ConfigKeySignPolicy,
		ConfigKeyAPIServerCSR,
		ConfigKeyAPIServerEtcdClientCSR,
		ConfigKeyFrontProxyClientCSR,
		ConfigKeyAPIServerKubeletClientCSR,
		ConfigKeyFrontProxyCACSR,
		ConfigKeyEtcdCACSR,
		ConfigKeyEtcdServerCSR,
		ConfigKeyEtcdHealthcheckClientCSR,
		ConfigKeyEtcdPeerCSR,
		ConfigKeyAdminKubeConfigCSR,
		ConfigKeyKubeletKubeConfigCSR,
		ConfigKeyControllerManagerCSR,
		ConfigKeySchedulerCSR,
	}
}

// verifyBuildConfigMapDataResult verifies the result of buildConfigMapData
func verifyBuildConfigMapDataResult(t *testing.T, result map[string]string, expectedKeys []string) {
	if len(expectedKeys) == 0 {
		assert.Empty(t, result, "result should be empty when no keys are available")
		return
	}
	assert.Equal(t, len(expectedKeys), len(result), "result should contain expected number of keys")
	for _, key := range expectedKeys {
		value, exists := result[key]
		assert.True(t, exists, "key %s should exist in result", key)
		assert.NotEmpty(t, value, "value for key %s should not be empty", key)
		assertValidJSON(t, value, "value for key %s should be valid JSON", key)
	}
}

// assertValidJSON checks if a string is valid JSON
func assertValidJSON(t *testing.T, jsonStr string, msgAndArgs ...interface{}) {
	var jsonObj interface{}
	err := json.Unmarshal([]byte(jsonStr), &jsonObj)
	assert.NoError(t, err, msgAndArgs...)
}

// TestLocalFileMappings tests the localFileMappings function
func TestLocalFileMappings(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)
	configData := createBaseConfigData()
	mappings := loader.localFileMappings(configData)

	verifyLocalFileMappings(t, mappings, configData)
}

// verifyLocalFileMappings verifies the localFileMappings result
func verifyLocalFileMappings(t *testing.T, mappings map[string]interface{}, configData *CertConfigData) {
	expectedKeys := getAllConfigKeys()
	assert.Equal(t, len(expectedKeys), len(mappings), "mappings should contain all config keys")

	verifyAllMappingPointers(t, mappings, configData)
}

// verifyAllMappingPointers verifies all mapping pointers
func verifyAllMappingPointers(t *testing.T, mappings map[string]interface{}, configData *CertConfigData) {
	expectedKeys := getAllConfigKeys()
	for _, key := range expectedKeys {
		verifyKeyExistsAndPointsToField(t, mappings, key, configData)
	}
}

// verifyKeyExistsAndPointsToField verifies a key exists and points to correct field
func verifyKeyExistsAndPointsToField(t *testing.T, mappings map[string]interface{}, key string, configData *CertConfigData) {
	ptr, exists := mappings[key]
	assert.True(t, exists, "key %s should exist in mappings", key)
	assert.NotNil(t, ptr, "key %s should have non-nil pointer", key)
	verifyPointerPointsToCorrectField(t, key, ptr, configData)
}

// verifyPointerPointsToCorrectField verifies pointer points to correct field
func verifyPointerPointsToCorrectField(t *testing.T, key string, ptr interface{}, configData *CertConfigData) {
	expectedPtr := getExpectedPointerForKey(key, configData)
	if expectedPtr != nil {
		assert.Equal(t, expectedPtr, ptr, "key %s should point to correct field", key)
	}
}

// getExpectedPointerForKey returns the expected pointer for a given key
func getExpectedPointerForKey(key string, configData *CertConfigData) interface{} {
	switch key {
	case ConfigKeyClusterCAPolicy:
		return &configData.ClusterCAPolicy
	case ConfigKeyClusterCACSR:
		return &configData.ClusterCACSR
	case ConfigKeySignPolicy:
		return &configData.SignPolicy
	case ConfigKeyAPIServerCSR:
		return &configData.APIServerCSR
	case ConfigKeyAPIServerEtcdClientCSR:
		return &configData.APIServerEtcdClientCSR
	case ConfigKeyFrontProxyClientCSR:
		return &configData.FrontProxyClientCSR
	case ConfigKeyAPIServerKubeletClientCSR:
		return &configData.APIServerKubeletClientCSR
	case ConfigKeyFrontProxyCACSR:
		return &configData.FrontProxyCACSR
	case ConfigKeyEtcdCACSR:
		return &configData.EtcdCACSR
	case ConfigKeyEtcdServerCSR:
		return &configData.EtcdServerCSR
	case ConfigKeyEtcdHealthcheckClientCSR:
		return &configData.EtcdHealthcheckClientCSR
	case ConfigKeyEtcdPeerCSR:
		return &configData.EtcdPeerCSR
	case ConfigKeyAdminKubeConfigCSR:
		return &configData.AdminKubeConfigCSR
	case ConfigKeyKubeletKubeConfigCSR:
		return &configData.KubeletKubeConfigCSR
	case ConfigKeyControllerManagerCSR:
		return &configData.ControllerManagerCSR
	case ConfigKeySchedulerCSR:
		return &configData.SchedulerCSR
	default:
		return nil
	}
}

// TestLoadLocalConfigData tests loading configuration from local directory
func TestLoadLocalConfigData(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	configData, err := loader.LoadLocalConfigData()
	assert.NoError(t, err)
	assert.Nil(t, configData)
}

// TestEnsureLocalConfigDir tests the ensureLocalConfigDir function
func TestEnsureLocalConfigDir(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	ok, err := loader.ensureLocalConfigDir()
	assert.NoError(t, err)
	assert.False(t, ok)
}

// TestSaveConfigMapData tests the SaveConfigMapData function
func TestSaveConfigMapData(t *testing.T) {
	tests := getSaveConfigMapDataTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runSaveConfigMapDataTest(t, tt)
		})
	}
}

// saveConfigMapDataTestCase represents a test case for SaveConfigMapData
type saveConfigMapDataTestCase struct {
	name        string
	cfg         *CertConfigData
	setupClient func() client.Client
	expectError bool
}

// getSaveConfigMapDataTestCases returns test cases for SaveConfigMapData
func getSaveConfigMapDataTestCases() []saveConfigMapDataTestCase {
	scheme := createSchemeForConfigTest()

	return []saveConfigMapDataTestCase{
		createNilConfigTestCase(),
		createEmptyConfigTestCase(scheme),
		createConfigWithAvailableKeysTestCase(scheme),
	}
}

// createNilConfigTestCase creates a test case for nil config
func createNilConfigTestCase() saveConfigMapDataTestCase {
	return saveConfigMapDataTestCase{
		name:        "nil config",
		cfg:         nil,
		setupClient: func() client.Client { return nil },
		expectError: false,
	}
}

// createEmptyConfigTestCase creates a test case for empty config
func createEmptyConfigTestCase(scheme *runtime.Scheme) saveConfigMapDataTestCase {
	return saveConfigMapDataTestCase{
		name: "empty config",
		cfg:  &CertConfigData{AvailableKeys: make(map[string]bool)},
		setupClient: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).Build()
		},
		expectError: false,
	}
}

// createConfigWithAvailableKeysTestCase creates a test case for config with available keys
func createConfigWithAvailableKeysTestCase(scheme *runtime.Scheme) saveConfigMapDataTestCase {
	cfg := createBaseConfigData()
	cfg.AvailableKeys[ConfigKeyClusterCAPolicy] = true
	cfg.AvailableKeys[ConfigKeySignPolicy] = true

	return saveConfigMapDataTestCase{
		name: "config with available keys",
		cfg:  cfg,
		setupClient: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).Build()
		},
		expectError: false,
	}
}

// runSaveConfigMapDataTest runs a single SaveConfigMapData test
func runSaveConfigMapDataTest(t *testing.T, tt saveConfigMapDataTestCase) {

	ctx := context.Background()

	var client client.Client
	if tt.setupClient != nil {
		client = tt.setupClient()
	}

	loader := NewCertConfigLoader(ctx, client, createTestBKECluster(), nil)

	err := loader.SaveConfigMapData(tt.cfg)

	if tt.expectError {
		assert.Error(t, err)
	} else {
		assert.NoError(t, err)
	}
}

// TestBuildConfigMapData_EdgeCases tests edge cases for buildConfigMapData
func TestBuildConfigMapData_EdgeCases(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	tests := []struct {
		name       string
		configData *CertConfigData
		expectLen  int
	}{
		{
			name:       "nil AvailableKeys",
			configData: &CertConfigData{AvailableKeys: nil, ClusterCAPolicy: createTestCertPolicy()},
			expectLen:  0,
		},
		{
			name: "all keys false",
			configData: &CertConfigData{
				AvailableKeys: map[string]bool{
					ConfigKeyClusterCAPolicy: false,
					ConfigKeySignPolicy:      false,
				},
				ClusterCAPolicy: createTestCertPolicy(),
			},
			expectLen: 0,
		},
		{
			name: "marshal error case",
			configData: &CertConfigData{
				AvailableKeys: map[string]bool{
					ConfigKeyClusterCAPolicy: true,
				},
				ClusterCAPolicy: pkiutil.CertPolicy{},
			},
			expectLen: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := loader.buildConfigMapData(tt.configData)
			assert.Equal(t, tt.expectLen, len(result))
		})
	}
}

// TestUpsertCertConfigMap tests the upsertCertConfigMap function
func TestUpsertCertConfigMap(t *testing.T) {
	scheme := createSchemeForConfigTest()
	data := map[string]string{
		ConfigKeyClusterCAPolicy: "{}",
	}

	tests := []struct {
		name        string
		setupClient func() client.Client
		expectError bool
	}{
		{
			name: "create new ConfigMap",
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectError: false,
		},
		{
			name: "update existing ConfigMap",
			setupClient: func() client.Client {
				existingCM := &corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      CertConfigMapName,
						Namespace: CertConfigMapNamespace,
					},
					Data: map[string]string{},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(existingCM).Build()
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			loader := NewCertConfigLoader(ctx, tt.setupClient(), createTestBKECluster(), nil)

			err := loader.upsertCertConfigMap(data)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestApplyConfigToCerts_EmptyCerts tests ApplyConfigToCerts with empty certs
func TestApplyConfigToCerts_EmptyCerts(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	configData := &CertConfigData{AvailableKeys: make(map[string]bool)}
	configData.AvailableKeys[ConfigKeyClusterCAPolicy] = true

	err := loader.ApplyConfigToCerts(pkiutil.Certificates{}, configData, "test-cluster")
	assert.NoError(t, err)
}

// TestProcessTemplateString tests the processTemplateString function
func TestProcessTemplateString(t *testing.T) {

	ctx := context.Background()
	bkeCluster := createTestBKECluster()
	loader := NewCertConfigLoader(ctx, nil, bkeCluster, nil)

	tests := []struct {
		name         string
		templateStr  string
		expectError  bool
		expectResult string
	}{
		{
			name:         "simple template",
			templateStr:  "{{.ClusterName}}",
			expectError:  false,
			expectResult: "test-cluster",
		},
		{
			name:         "template with no variables",
			templateStr:  "static-value",
			expectError:  false,
			expectResult: "static-value",
		},
		{
			name:         "invalid template",
			templateStr:  "{{.Invalid",
			expectError:  true,
			expectResult: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := loader.processTemplateString(tt.templateStr, "test-cluster")
			if tt.expectError {
				assert.Error(t, err)
				assert.Empty(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectResult, result)
			}
		})
	}
}

// TestGetTemplateData tests the getTemplateData function
func TestGetTemplateData(t *testing.T) {

	ctx := context.Background()

	tests := []struct {
		name       string
		bkeCluster *bkev1beta1.BKECluster
		expectData map[string]interface{}
	}{
		{
			name:       "cluster with endpoint",
			bkeCluster: createTestBKECluster(),
			expectData: map[string]interface{}{
				"ClusterName":      "test-cluster",
				"AdvertiseAddress": testHost,
			},
		},
		{
			name:       "cluster without endpoint",
			bkeCluster: &bkev1beta1.BKECluster{},
			expectData: map[string]interface{}{
				"ClusterName": "test-cluster",
			},
		},
		{
			name:       "nil bkeCluster",
			bkeCluster: nil,
			expectData: map[string]interface{}{
				"ClusterName": "test-cluster",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewCertConfigLoader(ctx, nil, tt.bkeCluster, nil)
			data := loader.getTemplateData("test-cluster")
			assert.Equal(t, tt.expectData["ClusterName"], data["ClusterName"])
			if addr, ok := tt.expectData["AdvertiseAddress"]; ok {
				assert.Equal(t, addr, data["AdvertiseAddress"])
			}
		})
	}
}

// TestParseDuration tests the parseDuration function
func TestParseDuration(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	tests := []struct {
		name        string
		durationStr string
		expectValid bool
	}{
		{
			name:        "valid duration hours",
			durationStr: "8760h",
			expectValid: true,
		},
		{
			name:        "valid duration mixed",
			durationStr: "24h30m",
			expectValid: true,
		},
		{
			name:        "invalid duration",
			durationStr: "invalid",
			expectValid: false,
		},
		{
			name:        "empty duration",
			durationStr: "",
			expectValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := loader.parseDuration(tt.durationStr)
			if tt.expectValid {
				assert.NotEqual(t, time.Duration(0), result)
			} else {
				assert.Equal(t, defaultCertValidity, result)
			}
		})
	}
}

// TestIsIPAddress tests the isIPAddress function
func TestIsIPAddress(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	tests := []struct {
		name   string
		host   string
		expect bool
	}{
		{
			name:   "valid IP",
			host:   "192.168.1.1",
			expect: true,
		},
		{
			name:   "localhost",
			host:   "localhost",
			expect: false,
		},
		{
			name:   "kubernetes",
			host:   "kubernetes",
			expect: false,
		},
		{
			name:   "127.0.0.1",
			host:   "127.0.0.1",
			expect: false,
		},
		{
			name:   "DNS name with dot",
			host:   "example.com",
			expect: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := loader.isIPAddress(tt.host)
			assert.Equal(t, tt.expect, result)
		})
	}
}

// TestParseIPAddress tests the parseIPAddress function
func TestParseIPAddress(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	tests := []struct {
		name   string
		host   string
		expect string
	}{
		{
			name:   "127.0.0.1",
			host:   "127.0.0.1",
			expect: "127.0.0.1",
		},
		{
			name:   "valid IPv4",
			host:   "192.168.1.1",
			expect: "192.168.1.1",
		},
		{
			name:   "invalid IP",
			host:   "invalid",
			expect: "",
		},
		{
			name:   "empty string",
			host:   "",
			expect: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := loader.parseIPAddress(tt.host)
			if tt.expect == "" {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				assert.Equal(t, tt.expect, result.String())
			}
		})
	}
}

// TestReadLocalJSON tests the readLocalJSON function
func TestReadLocalJSON(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	content, ok := loader.readLocalJSON(ConfigKeyClusterCAPolicy)
	assert.False(t, ok)
	assert.Empty(t, content)
}

// TestReadAndParseLocalFiles tests the readAndParseLocalFiles function
func TestReadAndParseLocalFiles(t *testing.T) {

	ctx := context.Background()
	loader := NewCertConfigLoader(ctx, nil, createTestBKECluster(), nil)

	mappings := loader.localFileMappings(&CertConfigData{AvailableKeys: make(map[string]bool)})
	hasData, count := loader.readAndParseLocalFiles(mappings, &CertConfigData{AvailableKeys: make(map[string]bool)})

	assert.False(t, hasData)
	assert.Equal(t, 0, count)
}

// createSchemeForConfigTest creates a scheme for config tests
func createSchemeForConfigTest() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	return scheme
}
