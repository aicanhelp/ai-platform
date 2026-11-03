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

// ConfigMap key names for certificate configuration files
const (
	// Cluster CA configuration keys
	// ConfigKeyClusterCAPolicy is the ConfigMap key for cluster CA policy
	ConfigKeyClusterCAPolicy = "cluster-ca-policy.json"
	// ConfigKeyClusterCACSR is the ConfigMap key for cluster CA certificate signing request
	ConfigKeyClusterCACSR = "cluster-ca-csr.json"
	// ConfigKeySignPolicy is the ConfigMap key for signing policy
	ConfigKeySignPolicy = "sign-policy.json"

	// API Server certificate configuration keys
	// ConfigKeyAPIServerCSR is the ConfigMap key for API server certificate signing request
	ConfigKeyAPIServerCSR = "apiserver-csr.json"
	// ConfigKeyAPIServerEtcdClientCSR is the ConfigMap key for API server's etcd client certificate signing request
	ConfigKeyAPIServerEtcdClientCSR = "apiserver-etcd-client-csr.json"
	// ConfigKeyFrontProxyClientCSR is the ConfigMap key for front-proxy client certificate signing request
	ConfigKeyFrontProxyClientCSR = "front-proxy-client-csr.json"
	// ConfigKeyAPIServerKubeletClientCSR is the ConfigMap key for API server's kubelet client certificate signing request
	ConfigKeyAPIServerKubeletClientCSR = "apiserver-kubelet-client-csr.json"

	// Front Proxy CA certificate configuration key
	// ConfigKeyFrontProxyCACSR is the ConfigMap key for front-proxy CA certificate signing request
	ConfigKeyFrontProxyCACSR = "front-proxy-ca-csr.json"

	// etcd certificate configuration keys
	// ConfigKeyEtcdCACSR is the ConfigMap key for etcd CA certificate signing request
	ConfigKeyEtcdCACSR = "etcd-ca-csr.json"
	// ConfigKeyEtcdServerCSR is the ConfigMap key for etcd server certificate signing request
	ConfigKeyEtcdServerCSR = "etcd-server-csr.json"
	// ConfigKeyEtcdHealthcheckClientCSR is the ConfigMap key for etcd healthcheck client certificate signing request
	ConfigKeyEtcdHealthcheckClientCSR = "etcd-healthcheck-client-csr.json"
	// ConfigKeyEtcdPeerCSR is the ConfigMap key for etcd peer certificate signing request
	ConfigKeyEtcdPeerCSR = "etcd-peer-csr.json"

	// Kubelet certificate configuration key
	// ConfigKeyKubeletCSR is the ConfigMap key for kubelet certificate signing request
	ConfigKeyKubeletCSR = "kubelet-csr.json"

	// KubeConfig certificate configuration keys
	// ConfigKeyAdminKubeConfigCSR is the ConfigMap key for admin kubeconfig certificate signing request
	ConfigKeyAdminKubeConfigCSR = "admin-kubeconfig-csr.json"
	// ConfigKeyKubeletKubeConfigCSR is the ConfigMap key for kubelet kubeconfig certificate signing request
	ConfigKeyKubeletKubeConfigCSR = "kubelet-kubeconfig-csr.json"
	// ConfigKeyControllerManagerCSR is the ConfigMap key for controller manager certificate signing request
	ConfigKeyControllerManagerCSR = "controller-manager-csr.json"
	// ConfigKeySchedulerCSR is the ConfigMap key for scheduler certificate signing request
	ConfigKeySchedulerCSR = "scheduler-csr.json"
	// ConfigKeyKubeProxyCSR is the ConfigMap key for kube-proxy certificate signing request
	ConfigKeyKubeProxyCSR = "kube-proxy-csr.json"
)

const (
	// GlobalCANamespace is the secret namespace for global-ca
	GlobalCANamespace = "kube-system"
	// GlobalCASecretName is the secret name for global-ca
	GlobalCASecretName = "global-ca"
	// CertConfigMapName is the ConfigMap name for certificate configuration
	CertConfigMapName = "cluster-cert-config"
	// CertConfigMapNamespace is the ConfigMap namespace for certificate configuration
	CertConfigMapNamespace = "kube-system"

	// GlobalCACertPath is the path for global-ca.crt
	GlobalCACertPath = "/etc/openFuyao/certs/global-ca.crt"
	// GlobalCAKeyPath is the path for global-ca.key
	GlobalCAKeyPath = "/etc/openFuyao/certs/global-ca.key"
	// CertChainPath is the path for the trust chain cetificate
	CertChainPath = "/etc/openFuyao/certs/trust-chain.crt"
	// KubeConfigCertName define kubeconfig certification name
	KubeConfigCertName = "kubeconfig"
)
