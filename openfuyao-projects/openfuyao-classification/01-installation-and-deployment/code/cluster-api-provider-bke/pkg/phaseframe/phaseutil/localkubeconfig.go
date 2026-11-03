/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"context"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	"github.com/wsva/lib_go/crypto"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	oneYearDay     = 365
	oneDayHour     = 24
	oneHundredYear = 100
)

// managementClusterCA contains the parsed CA certificate, private key, and certificate bytes
type managementClusterCA struct {
	cert      *x509.Certificate
	key       *rsa.PrivateKey
	certBytes []byte
}

func GetLocalKubeConfig(ctx context.Context, c client.Client) ([]byte, error) {
	// get kubeconfig
	localKubeConfigSecret := &corev1.Secret{}
	if err := c.Get(ctx, constant.GetLocalKubeConfigObjectKey(), localKubeConfigSecret); err != nil {
		return nil, err
	}
	localKubeConfig := localKubeConfigSecret.Data["config"]
	return localKubeConfig, nil
}

// GetLeastPrivilegeKubeConfig gets the least privilege kubeconfig from secret
func GetLeastPrivilegeKubeConfig(ctx context.Context, c client.Client) ([]byte, error) {
	// get least privilege kubeconfig from secret
	leastPrivilegeKubeConfigSecret := &corev1.Secret{}
	secretKey := client.ObjectKey{
		Namespace: metav1.NamespaceSystem,
		Name:      constant.LeastPrivilegeKubeConfigName,
	}
	if err := c.Get(ctx, secretKey, leastPrivilegeKubeConfigSecret); err != nil {
		return nil, err
	}
	leastPrivilegeKubeConfig := leastPrivilegeKubeConfigSecret.Data["config"]
	return leastPrivilegeKubeConfig, nil
}

// GetRemoteLocalKubeConfig gets the localkubeconfig from remote cluster (target cluster)
func GetRemoteLocalKubeConfig(ctx context.Context, remoteClient *kubernetes.Clientset) ([]byte, error) {
	secret, err := remoteClient.CoreV1().Secrets("kube-system").Get(ctx, constant.LocalKubeConfigName, metav1.GetOptions{})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get localkubeconfig from remote cluster")
	}

	localKubeConfig := secret.Data["config"]
	if len(localKubeConfig) == 0 {
		return nil, errors.New("localkubeconfig from remote cluster is empty")
	}

	return localKubeConfig, nil
}

// GenerateLowPrivilegeKubeConfig generates a low-privilege kubeconfig for bkeagent
func GenerateLowPrivilegeKubeConfig(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, remoteLocalKubeConfig []byte) ([]byte, error) {
	serverURL, err := extractServerURLFromKubeConfig(remoteLocalKubeConfig)
	if err != nil {
		return nil, errors.Wrap(err, "failed to extract server URL")
	}

	ca, err := parseManagementClusterCA(ctx, c, bkeCluster)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse management cluster CA")
	}

	kubeconfig, err := generateKubeConfigWithCert(serverURL, ca)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate kubeconfig")
	}

	return kubeconfig, nil
}

// extractServerURLFromKubeConfig extracts the server URL from kubeconfig bytes
func extractServerURLFromKubeConfig(kubeConfig []byte) (string, error) {
	config, err := clientcmd.Load(kubeConfig)
	if err != nil {
		return "", errors.Wrap(err, "failed to parse kubeconfig")
	}

	for _, cluster := range config.Clusters {
		if cluster.Server != "" {
			return cluster.Server, nil
		}
	}

	return "", errors.New("failed to extract server URL from kubeconfig")
}

// extractServerURLFromLocalKubeConfig extracts the server URL from local kubeconfig
func extractServerURLFromLocalKubeConfig(ctx context.Context, c client.Client) (string, error) {
	localKubeConfig, err := GetLocalKubeConfig(ctx, c)
	if err != nil {
		return "", errors.Wrap(err, "failed to get local kubeconfig")
	}

	config, err := clientcmd.Load(localKubeConfig)
	if err != nil {
		return "", errors.Wrap(err, "failed to parse local kubeconfig")
	}

	for _, cluster := range config.Clusters {
		if cluster.Server != "" {
			return cluster.Server, nil
		}
	}

	return "", errors.New("failed to extract server URL from local kubeconfig")
}

// parseManagementClusterCA parses the management cluster CA certificate and key
func parseManagementClusterCA(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*managementClusterCA, error) {
	caCertBytes, caKeyBytes, err := getManagementClusterCA(ctx, c, bkeCluster)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get management cluster CA")
	}

	caCerts, err := crypto.ParseCertsPEM(caCertBytes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse CA certificate")
	}
	if len(caCerts) == 0 {
		return nil, errors.New("CA certificate is empty")
	}

	caKey, err := crypto.ParsePrivateKeyPEM(caKeyBytes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to parse CA private key")
	}

	rsaKey, ok := caKey.(*rsa.PrivateKey)
	if !ok {
		return nil, errors.New("CA private key is not RSA private key")
	}

	return &managementClusterCA{
		cert:      caCerts[0],
		key:       rsaKey,
		certBytes: caCertBytes,
	}, nil
}

// generateKubeConfigWithCert generates kubeconfig with client certificate
func generateKubeConfigWithCert(serverURL string, ca *managementClusterCA) ([]byte, error) {
	clientCert, clientKey, err := generateBKEAgentClientCert(ca.cert, ca.key)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate client certificate")
	}

	kubeconfig, err := createKubeConfigWithClientCert(serverURL, clientCert, clientKey, ca.certBytes)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create kubeconfig")
	}

	return kubeconfig, nil
}

// getManagementClusterCA retrieves the management cluster CA certificate and key from a secret named bkeCluster.Name + "ca" in bkeCluster.Namespace
func getManagementClusterCA(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) ([]byte, []byte, error) {
	secretName := fmt.Sprintf("%s-%s", bkeCluster.Name, "ca")
	secretNamespace := bkeCluster.Namespace

	secret := &corev1.Secret{}
	secretKey := client.ObjectKey{
		Name:      secretName,
		Namespace: secretNamespace,
	}

	if err := c.Get(ctx, secretKey, secret); err != nil {
		return nil, nil, errors.Wrapf(err, "failed to get CA secret %s/%s", secretNamespace, secretName)
	}

	certBytes, ok := secret.Data["tls.crt"]
	if !ok {
		return nil, nil, errors.Errorf("CA secret %s/%s does not contain ca.crt key", secretNamespace, secretName)
	}
	if len(certBytes) == 0 {
		return nil, nil, errors.Errorf("CA certificate in secret %s/%s is empty", secretNamespace, secretName)
	}

	keyBytes, ok := secret.Data["tls.key"]
	if !ok {
		return nil, nil, errors.Errorf("CA secret %s/%s does not contain ca.key key", secretNamespace, secretName)
	}
	if len(keyBytes) == 0 {
		return nil, nil, errors.Errorf("CA private key in secret %s/%s is empty", secretNamespace, secretName)
	}

	return certBytes, keyBytes, nil
}

// generateBKEAgentClientCert generates a client certificate for bkeagent
func generateBKEAgentClientCert(caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey, error) {
	certConfig := &pkiutil.BKECert{
		BaseName: "bkeagent-client",
		Name:     "bkeagent-client",
		Config: pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: "bkeagent-cert-user",
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
			Validity: oneHundredYear * oneYearDay * oneDayHour * time.Hour,
		},
	}

	cert, key, err := pkiutil.NewCertAndKey(certConfig, caCert, caKey)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to generate certificate and key")
	}

	return cert, key, nil
}

// createKubeConfigWithClientCert creates a kubeconfig with client certificate authentication
func createKubeConfigWithClientCert(serverURL string, clientCert *x509.Certificate, clientKey *rsa.PrivateKey, caCertBytes []byte) ([]byte, error) {
	clientCertPEM := pkiutil.EncodeCertToPEM(clientCert)
	clientKeyPEM := pkiutil.EncodeKeyToPEM(clientKey)

	config := api.NewConfig()

	clusterName := "management-cluster"
	config.Clusters[clusterName] = &api.Cluster{
		Server:                   serverURL,
		CertificateAuthorityData: caCertBytes,
	}

	contextName := "bkeagent-context"
	config.Contexts[contextName] = &api.Context{
		Cluster:  clusterName,
		AuthInfo: "bkeagent-cert-user",
	}

	config.AuthInfos["bkeagent-cert-user"] = &api.AuthInfo{
		ClientCertificateData: clientCertPEM,
		ClientKeyData:         clientKeyPEM,
	}

	config.CurrentContext = contextName

	kubeconfigBytes, err := clientcmd.Write(*config)
	if err != nil {
		return nil, errors.Wrap(err, "failed to write kubeconfig")
	}

	return kubeconfigBytes, nil
}

// CreateBKEAgentRBACWithLocalKubeConfig creates all necessary RBAC resources for bkeagent using the provided local kubeconfig
func CreateBKEAgentRBACWithLocalKubeConfig(ctx context.Context, localKubeConfig []byte, bkeCluster *bkev1beta1.BKECluster) error {

	restConfig, err := clientcmd.RESTConfigFromKubeConfig(localKubeConfig)
	if err != nil {
		return errors.Wrap(err, "failed to create rest config from kubeconfig")
	}

	scheme := runtime.NewScheme()
	if err := rbacv1.AddToScheme(scheme); err != nil {
		return errors.Wrap(err, "failed to add rbac API to scheme")
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		return errors.Wrap(err, "failed to add core API to scheme")
	}

	highPrivilegeClient, err := client.New(restConfig, client.Options{Scheme: scheme})
	if err != nil {
		return errors.Wrap(err, "failed to create high-privilege client")
	}

	if err := createBKEAgentClusterRoles(ctx, highPrivilegeClient); err != nil {
		return errors.Wrap(err, "failed to create ClusterRoles")
	}

	// Grant read-only configmap permissions in namespaces where bkeagent needs to watch system/patch configmaps.
	configMapOnlyNamespaces := []string{
		"cluster-system",
		"user-system",
	}
	if err := createRoleBindingsForNamespaces(ctx, highPrivilegeClient, configMapOnlyNamespaces, "bkeagent-configmap-only"); err != nil {
		return errors.Wrap(err, "failed to create configmap-only RoleBindings")
	}

	if bkeCluster != nil {
		clusterNamespace := bkeCluster.Namespace
		if clusterNamespace != "" {
			if err := createRoleBindingForNamespace(ctx, highPrivilegeClient, clusterNamespace, "bkeagent-readwrite"); err != nil {
				return errors.Wrapf(err, "failed to create RoleBinding for namespace %s", clusterNamespace)
			}
		}
	}

	if err := createBKEAgentClusterAccessRoleBinding(ctx, highPrivilegeClient); err != nil {
		return errors.Wrap(err, "failed to create cluster access ClusterRoleBinding")
	}

	return nil
}

func createRoleBindingsForNamespaces(ctx context.Context, c client.Client, namespaces []string, roleName string) error {
	seen := make(map[string]struct{}, len(namespaces))
	for _, namespace := range namespaces {
		namespace = strings.TrimSpace(namespace)
		if namespace == "" {
			continue
		}
		if _, exists := seen[namespace]; exists {
			continue
		}
		seen[namespace] = struct{}{}
		if err := createRoleBindingForNamespace(ctx, c, namespace, roleName); err != nil {
			return errors.Wrapf(err, "failed to create RoleBinding for namespace %s", namespace)
		}
	}
	return nil
}

// createBKEAgentClusterRoles creates the ClusterRoles needed for bkeagent
func createBKEAgentClusterRoles(ctx context.Context, c client.Client) error {

	if err := createReadwriteClusterRole(ctx, c); err != nil {
		return errors.Wrap(err, "failed to create readwrite ClusterRole")
	}

	if err := createConfigMapOnlyClusterRole(ctx, c); err != nil {
		return errors.Wrap(err, "failed to create configmap-only ClusterRole")
	}

	if err := createClusterAccessClusterRole(ctx, c); err != nil {
		return errors.Wrap(err, "failed to create cluster access ClusterRole")
	}

	return nil
}

// createReadwriteClusterRole creates the bkeagent-readwrite ClusterRole
func createReadwriteClusterRole(ctx context.Context, c client.Client) error {
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bkeagent-readwrite",
		},
		Rules: []rbacv1.PolicyRule{
			// Core API resources
			{
				APIGroups: []string{""},
				Resources: []string{"secrets"},
				Verbs:     []string{"get"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"configmaps"},
				Verbs:     []string{"get", "list", "watch", "create", "update"},
			},
		},
	}

	return createOrUpdateClusterRole(ctx, c, role)
}

// createConfigMapOnlyClusterRole creates the bkeagent-configmap-only ClusterRole
// This role only provides configmap permissions, without secret permissions
func createConfigMapOnlyClusterRole(ctx context.Context, c client.Client) error {
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bkeagent-configmap-only",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"configmaps"},
				Verbs:     []string{"get", "list", "watch"},
			},
		},
	}

	return createOrUpdateClusterRole(ctx, c, role)
}

// createClusterAccessClusterRole creates the bkeagent-cluster-access ClusterRole
// This role provides permissions for accessing CRDs and cluster resources
func createClusterAccessClusterRole(ctx context.Context, c client.Client) error {
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bkeagent-cluster-access",
		},
		Rules: []rbacv1.PolicyRule{
			// BKE CRDs - read only
			{
				APIGroups: []string{"bke.bocloud.com"},
				Resources: []string{"bkeclusters"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"bke.bocloud.com"},
				Resources: []string{"bkenodes"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{confv1beta1.ContainerdConfigGVR().Group},
				Resources: []string{confv1beta1.ContainerdConfigGVR().Resource},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{confv1beta1.KubeletConfigGVR().Group},
				Resources: []string{confv1beta1.KubeletConfigGVR().Resource},
				Verbs:     []string{"get", "list", "watch"},
			},
			// Node bkeagent: read commands; write status only. Finalizers are set/removed by bke-controller.
			{
				APIGroups: []string{"bkeagent.bocloud.com"},
				Resources: []string{"commands"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"bkeagent.bocloud.com"},
				Resources: []string{"commands/status"},
				Verbs:     []string{"get", "patch", "update"},
			},
		},
	}

	return createOrUpdateClusterRole(ctx, c, role)
}

// createOrUpdateClusterRole creates or updates a ClusterRole
func createOrUpdateClusterRole(ctx context.Context, c client.Client, role *rbacv1.ClusterRole) error {
	if err := RetryOnConflict(func() error {
		existingClusterRole := &rbacv1.ClusterRole{}
		if err := c.Get(ctx, client.ObjectKey{Name: role.Name}, existingClusterRole); err != nil {
			return err
		}
		existingClusterRole.Rules = role.Rules
		return c.Update(ctx, existingClusterRole)
	}); err != nil {
		if apierrors.IsNotFound(err) {
			return c.Create(ctx, role)
		}
		return err
	}
	return nil
}

// getBKEAgentSubject returns the Subject for bkeagent-cert-user
func getBKEAgentSubject() []rbacv1.Subject {
	return []rbacv1.Subject{
		{
			Kind:     "User",
			Name:     "bkeagent-cert-user",
			APIGroup: "rbac.authorization.k8s.io",
		},
	}
}

// ensureNamespaceExists ensures the namespace exists, creating it if necessary
func ensureNamespaceExists(ctx context.Context, c client.Client, namespace string) error {
	ns := &corev1.Namespace{}
	err := c.Get(ctx, client.ObjectKey{Name: namespace}, ns)
	if err != nil {
		if apierrors.IsNotFound(err) {
			ns = &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: namespace,
				},
			}
			if err := c.Create(ctx, ns); err != nil {
				return errors.Wrapf(err, "failed to create namespace %s", namespace)
			}
		} else {
			return errors.Wrapf(err, "failed to get namespace %s", namespace)
		}
	}
	return nil
}

// createOrUpdateRoleBinding creates or updates a RoleBinding
func createOrUpdateRoleBinding(ctx context.Context, c client.Client, roleBinding *rbacv1.RoleBinding) error {
	if err := RetryOnConflict(func() error {
		existingRoleBinding := &rbacv1.RoleBinding{}
		if err := c.Get(ctx, client.ObjectKey{Name: roleBinding.Name, Namespace: roleBinding.Namespace}, existingRoleBinding); err != nil {
			if apierrors.IsNotFound(err) {
				return err
			}
			return errors.Wrapf(err, "failed to get RoleBinding '%s' in namespace %s", roleBinding.Name, roleBinding.Namespace)
		}
		existingRoleBinding.Subjects = roleBinding.Subjects
		existingRoleBinding.RoleRef = roleBinding.RoleRef
		if err := c.Update(ctx, existingRoleBinding); err != nil {
			return errors.Wrapf(err, "failed to update RoleBinding '%s' in namespace %s", roleBinding.Name, roleBinding.Namespace)
		}
		return nil
	}); err != nil {
		if apierrors.IsNotFound(err) {
			if err := c.Create(ctx, roleBinding); err != nil {
				return errors.Wrapf(err, "failed to create RoleBinding '%s' in namespace %s", roleBinding.Name, roleBinding.Namespace)
			}
			return nil
		}
		return err
	}
	return nil
}

// createOrUpdateClusterRoleBinding creates or updates a ClusterRoleBinding
func createOrUpdateClusterRoleBinding(ctx context.Context, c client.Client, clusterRoleBinding *rbacv1.ClusterRoleBinding) error {
	if err := RetryOnConflict(func() error {
		existingClusterRoleBinding := &rbacv1.ClusterRoleBinding{}
		if err := c.Get(ctx, client.ObjectKey{Name: clusterRoleBinding.Name}, existingClusterRoleBinding); err != nil {
			return err
		}
		existingClusterRoleBinding.Subjects = clusterRoleBinding.Subjects
		existingClusterRoleBinding.RoleRef = clusterRoleBinding.RoleRef
		return c.Update(ctx, existingClusterRoleBinding)
	}); err != nil {
		if apierrors.IsNotFound(err) {
			return c.Create(ctx, clusterRoleBinding)
		}
		return err
	}
	return nil
}

// createRoleBindingForNamespace creates a RoleBinding in the specified namespace
func createRoleBindingForNamespace(ctx context.Context, c client.Client, namespace, roleName string) error {
	if err := ensureNamespaceExists(ctx, c, namespace); err != nil {
		return err
	}

	roleBinding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bkeagent",
			Namespace: namespace,
		},
		Subjects: getBKEAgentSubject(),
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     roleName,
			APIGroup: "rbac.authorization.k8s.io",
		},
	}

	return createOrUpdateRoleBinding(ctx, c, roleBinding)
}

// createBKEAgentClusterAccessRoleBinding creates the ClusterRoleBinding for cluster access resources
func createBKEAgentClusterAccessRoleBinding(ctx context.Context, c client.Client) error {
	clusterRoleBinding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bkeagent-cluster-access",
		},
		Subjects: getBKEAgentSubject(),
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     "bkeagent-cluster-access",
			APIGroup: "rbac.authorization.k8s.io",
		},
	}

	return createOrUpdateClusterRoleBinding(ctx, c, clusterRoleBinding)
}
