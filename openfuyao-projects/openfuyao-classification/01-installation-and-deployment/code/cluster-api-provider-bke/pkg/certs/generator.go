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

package certs

import (
	"context"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/pkg/errors"
	"github.com/wsva/lib_go/crypto"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/cluster-api/util/kubeconfig"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	agentutils "gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// TLSKeyDataName is the key used to store a TLS private key in the secret's data field.
	TLSKeyDataName = "tls.key"

	// TLSCrtDataName is the key used to store a TLS certificate in the secret's data field.
	TLSCrtDataName = "tls.crt"

	// ChainDataName is the key used to store a chain certificate in the secret's data field.
	ChainDataName = "trust-chain.crt"
)

type BKEKubernetesCertGenerator struct {
	certNamespace   string
	certClusterName string

	bkeCluster *bkev1beta1.BKECluster
	client     client.Client
	ctx        context.Context

	bkeCerts              pkiutil.Certificates
	caCertificatesContent map[string]map[string][]byte
	certificatesContent   map[string]map[string][]byte
	log                   *log.Logger

	kubeConfigEndpoint   string
	needCreateKubeConfig bool
	isUserCustomCA       bool

	// nodes holds the BKENode resources for this cluster, used for cert generation
	nodes bkenode.Nodes
}

func NewKubernetesCertGenerator(ctx context.Context, client client.Client,
	bkeCluster *bkev1beta1.BKECluster) *BKEKubernetesCertGenerator {
	return NewKubernetesCertGeneratorWithCache(ctx, client, nil, bkeCluster)
}

func NewKubernetesCertGeneratorWithCache(ctx context.Context, c client.Client, cache cache.Cache,
	bkeCluster *bkev1beta1.BKECluster) *BKEKubernetesCertGenerator {
	gen := &BKEKubernetesCertGenerator{
		certNamespace:        bkeCluster.Namespace,
		certClusterName:      bkeCluster.Name,
		bkeCluster:           bkeCluster,
		client:               c,
		ctx:                  ctx,
		log:                  log.With("name", "certsGenerator").With("clientObjNS", utils.ClientObjNS(bkeCluster)),
		needCreateKubeConfig: true,
		kubeConfigEndpoint:   bkeCluster.Spec.ControlPlaneEndpoint.String(),
	}
	if cache != nil {
		gen.client = phaseutil.NewWriteThroughClient(c, cache)
	}
	return gen
}

func (k *BKEKubernetesCertGenerator) ConfigKubeConfig(endpoint string) {
	k.needCreateKubeConfig = true
	if endpoint == "" {
		k.kubeConfigEndpoint = k.bkeCluster.Spec.ControlPlaneEndpoint.String()
		return
	}
	k.kubeConfigEndpoint = endpoint
}

// SetNodes sets the nodes for cert generation
func (k *BKEKubernetesCertGenerator) SetNodes(nodes bkenode.Nodes) {
	k.nodes = nodes
}

// LookUpOrGenerate generate kubernetes certs,exclude sa cert and kubeconfig
func (k *BKEKubernetesCertGenerator) LookUpOrGenerate() error {
	if err := k.setupGlobalCA(); err != nil {
		return err
	}

	if err := k.prepareBkeCerts(false); err != nil {
		return err
	}

	// Apply any loaded certificate configuration after cert definitions
	// have been prepared. setupGlobalCA() only determines whether a user
	// provided CA exists; applying configuration must happen after
	// prepareBkeCerts so that k.bkeCerts is initialized.
	k.LoadConfigForCerts()

	needCreateSecret, err := k.generateCertificates()
	if err != nil {
		return err
	}

	if needCreateSecret {
		return k.createCertificateSecrets()
	}
	return nil
}

// setupGlobalCA sets up the global CA
func (k *BKEKubernetesCertGenerator) setupGlobalCA() error {
	globalCASecret, err := k.LoadGlobalCA()
	if err != nil {
		k.log.Warnf("User don't upload global CA: %v", err)
	}

	// 如果找到了用户提供的全局 CA，则使用它；否则使用自签名 CA
	if globalCASecret != nil {
		k.fillInCertificateContent(globalCASecret[TLSCrtDataName], globalCASecret[TLSKeyDataName],
			GlobalCASecretName, true)
	}

	// Do not apply configuration here — cert definitions are not prepared yet.
	// Applying configuration will be done after prepareBkeCerts in LookUpOrGenerate.
	return nil
}

// generateCertificates generates all necessary certificates
func (k *BKEKubernetesCertGenerator) generateCertificates() (bool, error) {
	var needCreateSecret = false
	regeneratedCAs := make(map[string]bool)

	// 生成ca证书，以及其他证书
	for _, cert := range k.bkeCerts {
		exit, err := k.lookup(cert)
		if err != nil {
			return false, err
		}
		// CA cascade: after a CA is regenerated, force regeneration of all leaf certs
		// signed by that CA to avoid a broken certificate chain (new CA + old leaf cert).
		if exit && cert.CAName != "" && regeneratedCAs[cert.CAName] {
			k.log.Infof("CA %q was regenerated, force regenerating dependent cert %q", cert.CAName, cert.Name)
			exit = false
		}
		if exit {
			continue
		}

		k.log.Infof("generate cert %q, and save to secret %s/%s ",
			cert.Name, k.certNamespace, NewCertSecretName(k.certClusterName, cert.Name))
		needCreateSecret = true
		if cert.CAName != "" {
			if err := k.generateCertAndKeyWithCA(cert, cert.IsCA); err != nil {
				return false, err
			}
			continue
		}
		// 要确保生成CA证书是第一步，原本写死所以第一个循环一定在这执行
		if err := k.generateCACertAndKey(cert); err != nil {
			return false, err
		}
		regeneratedCAs[cert.Name] = true
	}

	// 生成sa证书
	saCert := pkiutil.BKECertServiceAccount()
	exit, err := k.lookup(saCert)
	if err != nil {
		return false, err
	}
	if !exit {
		needCreateSecret = true
		if err := k.generateSAKeyAndPublicKey(pkiutil.BKECertServiceAccount()); err != nil {
			return false, err
		}
	}

	return needCreateSecret, nil
}

func (k *BKEKubernetesCertGenerator) NeedGenerate() (bool, error) {
	bkeCerts := pkiutil.GetDefaultCertList()
	bkeCerts = append(bkeCerts, pkiutil.BKEAdminKubeConfig())

	for _, cert := range bkeCerts {
		exit, err := k.lookup(cert)
		if err != nil {
			return false, err
		}
		if !exit {
			log.Infof("need generate cert again: %v", cert.Name)
			return true, nil
		}
	}
	return false, nil
}

// prepareBkeCerts appends alternative names and IPs to the API server and etcd server certificates.
func (k *BKEKubernetesCertGenerator) prepareBkeCerts(isVerify bool) error {

	if k.isUserCustomCA {
		k.bkeCerts = pkiutil.GetUserCustomCerts()
		k.SetCertsCAName()
	} else {
		if isVerify {
			k.bkeCerts = pkiutil.GetCertsWithoutCA()
		} else {
			k.bkeCerts = pkiutil.GetDefaultCertList()
		}
	}
	// Append the BKE admin kubeconfig certificate if needed.
	if k.needCreateKubeConfig {
		k.bkeCerts = append(k.bkeCerts, pkiutil.BKEAdminKubeConfig())
	}

	// If there is no BKE cluster, return early.
	if k.bkeCluster == nil {
		return nil
	}

	bkeConfig := k.bkeCluster.Spec.ClusterConfig
	extraAltNames := k.getExtraAltNames()

	for _, cert := range k.bkeCerts {
		var altNames *certutil.AltNames
		var err error

		switch cert.BaseName {
		case pkiutil.APIServerCertAndKeyBaseName:
			altNames, err = pkiutil.GetAPIServerCertAltNamesWithNodes(bkeConfig, k.nodes)
		case pkiutil.EtcdServerCertAndKeyBaseName:
			altNames, err = pkiutil.GetEtcdCertAltNamesWithNodes(bkeConfig, k.nodes, true)
		case pkiutil.EtcdPeerCertAndKeyBaseName:
			altNames, err = pkiutil.GetEtcdCertAltNamesWithNodes(bkeConfig, k.nodes, false)
		case pkiutil.ControllerManagerCertAndKeyBaseName, pkiutil.SchedulerCertAndKeyBaseName:
			altNames, err = pkiutil.GetMasterNodeAltNamesWithNodes(k.nodes)
		default:
			continue
		}

		if err != nil {
			return errors.Wrapf(err, "failed to get alt names from bke config for %q", cert.BaseName)
		}

		if err := k.applyAltNamesToCert(cert, altNames, extraAltNames); err != nil {
			return err
		}
	}
	return nil
}

func (k *BKEKubernetesCertGenerator) getExtraAltNames() []string {
	var extraAltNames []string
	if k.bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		extraAltNames = append(extraAltNames, k.bkeCluster.Spec.ControlPlaneEndpoint.Host)
	}

	// Add external load balancer IP if available
	if k.bkeCluster.Spec.ClusterConfig != nil {
		if loadBalanceIP := utils.GetExtraLoadBalanceIP(k.bkeCluster.Spec.ClusterConfig.CustomExtra); loadBalanceIP != "" {
			extraAltNames = append(extraAltNames, loadBalanceIP)
		}
	}
	return extraAltNames
}

// applyAltNamesToCert applies altNames to cert and appends extra SANs
func (k *BKEKubernetesCertGenerator) applyAltNamesToCert(cert *pkiutil.BKECert,
	altNames *certutil.AltNames, extraAltNames []string) error {
	cert.Config.AltNames.DNSNames = append(cert.Config.AltNames.DNSNames, altNames.DNSNames...)
	cert.Config.AltNames.IPs = append(cert.Config.AltNames.IPs, altNames.IPs...)
	if err := pkiutil.AppendSANsToAltNames(&cert.Config.AltNames, extraAltNames, cert.BaseName); err != nil {
		return errors.Wrapf(err, "failed to append alt names to %q", cert.BaseName)
	}
	return nil
}

// fillInCertificateContent fills in the certificate content
func (k *BKEKubernetesCertGenerator) fillInCertificateContent(crtBytes, keyBytes []byte, certName string, isCA bool) {
	data := map[string][]byte{
		TLSCrtDataName: crtBytes,
		TLSKeyDataName: keyBytes,
	}
	if isCA {
		if k.caCertificatesContent == nil {
			k.caCertificatesContent = make(map[string]map[string][]byte)
		}
		k.caCertificatesContent[certName] = data
		return
	}

	if k.certificatesContent == nil {
		k.certificatesContent = make(map[string]map[string][]byte)
	}
	k.certificatesContent[certName] = data
}

// lookup checks if a certificate secret already exists.
// If the certificate exists but is expiring soon, it is treated as needing regeneration.
func (k *BKEKubernetesCertGenerator) lookup(cert *pkiutil.BKECert) (bool, error) {
	secretName := NewCertSecretName(k.certClusterName, cert.Name)

	var exists bool
	var err error
	if cert.Name == KubeConfigCertName {
		exists, err = k.lookupKubeConfigCert(secretName)
	} else {
		exists, err = k.lookupRegularCert(secretName)
	}
	if err != nil || !exists {
		return exists, err
	}

	// Certificate exists - check if it is expiring soon.
	// Certificates expiring within CertExpireAlertDays are treated as needing regeneration.
	if k.isCertExpiringSoon(cert) {
		k.log.Warnf("certificate %q will expire within %d days, will regenerate",
			cert.Name, constant.CertExpireAlertDays)
		return false, nil
	}

	return true, nil
}

// lookupRegularCert looks up a regular certificate secret
func (k *BKEKubernetesCertGenerator) lookupRegularCert(secretName string) (bool, error) {
	secret := &corev1.Secret{}
	err := k.client.Get(k.ctx, types.NamespacedName{Namespace: k.certNamespace, Name: secretName}, secret)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func (k *BKEKubernetesCertGenerator) lookupKubeConfigCert(secretName string) (bool, error) {
	secretKey := types.NamespacedName{Namespace: k.certNamespace, Name: secretName}
	var ready bool

	err := phaseutil.RetryOnError(
		func() error {
			secret := &corev1.Secret{}
			getErr := k.client.Get(k.ctx, secretKey, secret)
			if getErr != nil {
				return getErr // predicate below decides whether to retry
			}
			if secret.Data["value"] == nil {
				return apierrors.NewNotFound(corev1.Resource("secret"), secretKey.Name)
			}
			if IsHACluster(k.bkeCluster, k.nodes) && secret.Data["ha"] == nil {
				return apierrors.NewNotFound(corev1.Resource("secret"), secretKey.Name)
			}
			ready = true
			return nil
		},
		phaseutil.WithBackoff(phaseutil.ReadBackoff()),
		phaseutil.WithPredicate(phaseutil.IsNotFound),
		phaseutil.WithOnRetry(func(attempt int, err error) {
			k.log.Debugf("lookupKubeConfigCert: retrying read on notfound (%d/%d): %v",
				attempt, phaseutil.ReadBackoff().Steps, err)
		}),
	)
	if err != nil && !apierrors.IsNotFound(err) {
		k.log.Errorf("LOOKUP: failed to validate secret %s/%s for cert %s: %v",
			k.certNamespace, secretName, KubeConfigCertName, err)
		return false, err
	}
	return ready, nil
}

func (k *BKEKubernetesCertGenerator) loadCaCertContent() error {
	var bkeCaCerts pkiutil.Certificates = []*pkiutil.BKECert{
		pkiutil.BKECertRootCA(),
		pkiutil.BKECertEtcdCA(),
		pkiutil.BKECertFrontProxyCA(),
	}
	for _, cert := range bkeCaCerts {
		secretName := NewCertSecretName(k.certClusterName, cert.Name)
		secret := &corev1.Secret{}
		if err := k.client.Get(k.ctx,
			types.NamespacedName{Namespace: k.certNamespace, Name: secretName},
			secret); err != nil {
			if apierrors.IsNotFound(err) {
				k.log.Debugf("secret %s/%s not found", k.certNamespace, secretName)
				continue
			}
			return err

		}
		k.fillInCertificateContent(secret.Data[TLSCrtDataName], secret.Data[TLSKeyDataName], cert.Name, true)
	}
	return nil
}

// createCertificateSecrets creates the certificate secrets
func (k *BKEKubernetesCertGenerator) createCertificateSecrets() error {
	k.log.Infof("CREATE_SECRETS: starting to create certificate secrets")
	k.log.Infof("CREATE_SECRETS: caCertificatesContent count: %d", len(k.caCertificatesContent))
	k.log.Infof("CREATE_SECRETS: certificatesContent count: %d", len(k.certificatesContent))

	// Transfer CA certificates to certificatesContent for unified creation
	if err := k.transferCACertificates(); err != nil {
		return err
	}

	// Create certificate secrets
	if err := k.createCertSecrets(); err != nil {
		return err
	}

	// Finally create kubeconfig
	return k.maybeCreateKubeConfig()
}

// transferCACertificates transfers CA certificate information to certificatesContent
func (k *BKEKubernetesCertGenerator) transferCACertificates() error {
	for certName, data := range k.caCertificatesContent {
		k.log.Infof("CREATE_SECRETS: transferring CA cert %s to certificatesContent", certName)
		k.fillInCertificateContent(data[TLSCrtDataName], data[TLSKeyDataName], certName, false)
	}
	return nil
}

// createCertSecrets creates individual certificate secrets
func (k *BKEKubernetesCertGenerator) createCertSecrets() error {
	for certName, data := range k.certificatesContent {
		if certName == GlobalCASecretName {
			continue
		}
		if err := k.createSingleCertSecret(certName, data); err != nil {
			return err
		}
	}
	return nil
}

// createSingleCertSecret creates a single certificate secret
func (k *BKEKubernetesCertGenerator) createSingleCertSecret(certName string, data map[string][]byte) error {
	secretName := NewCertSecretName(k.certClusterName, certName)
	k.log.Infof("CREATE_SECRETS: creating secret %s/%s for cert %s", k.certNamespace, secretName, certName)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: k.certNamespace,
		},
		Data: data,
		Type: agentutils.BKESecretType,
	}
	controllerRef := metav1.NewControllerRef(k.bkeCluster, k.bkeCluster.GroupVersionKind())
	secret.SetOwnerReferences([]metav1.OwnerReference{*controllerRef})

	return k.createOrUpdateSecret(secret)
}

// createOrUpdateSecret creates or updates a secret
func (k *BKEKubernetesCertGenerator) createOrUpdateSecret(secret *corev1.Secret) error {
	if err := k.client.Create(k.ctx, secret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			if err := k.client.Delete(k.ctx, secret); err != nil {
				return fmt.Errorf("failed to delete secret %q: %w", utils.ClientObjNS(secret), err)
			}
			if err := k.client.Create(k.ctx, secret); err != nil {
				return fmt.Errorf("failed to create secret %q: %w", utils.ClientObjNS(secret), err)
			}
		}
	}
	return nil
}

// maybeCreateKubeConfig creates kubeconfig if needed
func (k *BKEKubernetesCertGenerator) maybeCreateKubeConfig() error {
	if !k.needCreateKubeConfig {
		return nil
	}

	if err := k.GenerateKubeConfig(k.kubeConfigEndpoint); err != nil {
		k.log.Errorf("CREATE_SECRETS: failed to create kubeconfig: %v", err)
		return fmt.Errorf("failed to create kubeconfig secret: %w", err)
	}
	return nil
}

// generateCACertAndKey generates a CA certificate and key.
func (k *BKEKubernetesCertGenerator) generateCACertAndKey(caCert *pkiutil.BKECert) error {
	if caCert.Name == pkiutil.BKEAdminKubeConfig().Name {
		return nil
	}
	crt, key, err := pkiutil.NewCertificateAuthority(caCert)
	if err != nil {
		return fmt.Errorf("failed to generate CA cert and key: %w", err)
	}
	k.fillInCertificateContent(pkiutil.EncodeCertToPEM(crt), pkiutil.EncodeKeyToPEM(key), caCert.Name, true)
	return nil
}

// generateSAKeyAndPublicKey generates a service account key and public key.
func (k *BKEKubernetesCertGenerator) generateSAKeyAndPublicKey(saCert *pkiutil.BKECert) error {
	pub, key, err := pkiutil.NewRSACertAndKey(saCert)
	if err != nil {
		return fmt.Errorf("failed to generate private key: %w", err)
	}
	pubBytes, err := pkiutil.EncodePublicKeyToPEM(pub)
	if err != nil {
		return fmt.Errorf("failed to encode public key: %w", err)
	}
	k.fillInCertificateContent(pubBytes, pkiutil.EncodeKeyToPEM(key), saCert.Name, false)
	return nil
}

// generateCertAndKeyWithCA generates a certificate and key signed by the given CA.
func (k *BKEKubernetesCertGenerator) generateCertAndKeyWithCA(cert *pkiutil.BKECert, isCA bool) error {
	if k.caCertificatesContent == nil || len(k.caCertificatesContent) == 0 {
		if err := k.loadCaCertContent(); err != nil {
			return err
		}
	}
	caCertContent := make(map[string][]byte)
	for caName, data := range k.caCertificatesContent {
		if caName == cert.CAName {
			caCertContent = data
			break
		}

		tCaName := strings.TrimSuffix(cert.CAName, "-ca")
		if strings.Contains(tCaName, caName) {
			caCertContent = data
			break
		}
	}

	if len(caCertContent) == 0 {
		return errors.Errorf("failed to find CA certificate %q", cert.CAName)
	}

	caCrt, err := crypto.ParseCertsPEM(caCertContent[TLSCrtDataName])
	if err != nil {
		return fmt.Errorf("failed to parse CA certificate %q: %w", cert.CAName, err)
	}
	if !caCrt[0].IsCA {
		return errors.Errorf("certificate %q is not a CA", cert.CAName)
	}

	caKey, err := crypto.ParsePrivateKeyPEM(caCertContent[TLSKeyDataName])
	if err != nil {
		return fmt.Errorf("failed to parse CA private key %q: %w", cert.CAName, err)
	}

	rsaKey, ok := caKey.(*rsa.PrivateKey)
	if !ok {
		return errors.Errorf("CA private key %q is not RSA private key", cert.CAName)
	}

	crt, key, err := pkiutil.NewCertAndKey(cert, caCrt[0], rsaKey)
	if err != nil {
		return fmt.Errorf("failed to generate cert and key: %w", err)
	}
	if HasServerAuth(crt) {
		log.Debugf("%q serving cert is signed for DNS names %v and IPs %v",
			cert.BaseName, crt.DNSNames, crt.IPAddresses)

	}

	k.fillInCertificateContent(pkiutil.EncodeCertToPEM(crt), pkiutil.EncodeKeyToPEM(key), cert.Name, isCA)
	return nil
}

// getCertificateFromSecret fetches and parses a certificate from a secret
func (k *BKEKubernetesCertGenerator) getCertificateFromSecret(certName string) (*x509.Certificate, error) {
	secretName := NewCertSecretName(k.certClusterName, certName)
	secret := &corev1.Secret{}

	secretKey := types.NamespacedName{
		Name:      secretName,
		Namespace: k.certNamespace,
	}
	if err := k.client.Get(k.ctx, secretKey, secret); err != nil {
		objNS := utils.ClientObjNS(secret)
		return nil, fmt.Errorf("failed to get secret %q: %w", objNS, err)
	}

	crtBytes, ok := secret.Data[TLSCrtDataName]
	if !ok {
		objNS := utils.ClientObjNS(secret)
		return nil, errors.Errorf(
			"failed to find certificate data %q in secret %q",
			TLSCrtDataName,
			objNS,
		)
	}

	crt, err := crypto.ParseCertsPEM(crtBytes)
	if err != nil {
		objNS := utils.ClientObjNS(secret)
		return nil, errors.Errorf(
			"failed to parse certificate data %q in secret %q: %v",
			TLSCrtDataName,
			objNS,
			err,
		)
	}

	return crt[0], nil
}

// isCertExpiringSoon checks if a certificate will expire within CertExpireAlertDays.
// Returns false for non-cert entries (kubeconfig, SA key pair) or on parse errors
// to avoid blocking the normal flow.
func (k *BKEKubernetesCertGenerator) isCertExpiringSoon(cert *pkiutil.BKECert) bool {
	// kubeconfig is stored as a kubeconfig file in the "value" key, not tls.crt; skip it.
	if cert.Name == KubeConfigCertName {
		return false
	}

	crt, err := k.getCertificateFromSecret(cert.Name)
	if err != nil {
		// SA key pair is not an x509 certificate; getCertificateFromSecret fails on it.
		k.log.Debugf("skip expiration check for %q: %v", cert.Name, err)
		return false
	}

	if crt.NotAfter.Before(time.Now().AddDate(0, 0, constant.CertExpireAlertDays)) {
		return true
	}
	return false
}

// VerifyCertificateSans verifies the SANs of the certificate.
func (k *BKEKubernetesCertGenerator) VerifyCertificateSans() error {
	if err := k.prepareBkeCerts(true); err != nil {
		return err
	}

	for _, cert := range k.bkeCerts {
		if cert.Name == KubeConfigCertName {
			continue
		}
		crt, err := k.getCertificateFromSecret(cert.Name)
		if err != nil {
			return err
		}

		if !reflect.DeepEqual(cert.Config.AltNames.DNSNames, crt.DNSNames) {
			return errors.Errorf("failed to verify certificate %q: SANs are not equal", cert.Name)
		}

		var certAltIps, crtAltIps []string
		for _, ip := range cert.Config.AltNames.IPs {
			certAltIps = append(certAltIps, ip.String())
		}
		for _, ip := range crt.IPAddresses {
			crtAltIps = append(crtAltIps, ip.String())
		}

		if !reflect.DeepEqual(certAltIps, crtAltIps) {
			return errors.Errorf("failed to verify certificate %q: SANs are not equal", cert.Name)
		}
	}
	return nil
}

// GenerateKubeConfig generates the kubeconfig for the cluster, and stores it in the secret.
func (k *BKEKubernetesCertGenerator) GenerateKubeConfig(endpoint string) error {
	k.log.Infof("GENERATE_KUBECONFIG: starting to generate kubeconfig for cluster %s", k.certClusterName)
	k.log.Infof("GENERATE_KUBECONFIG: endpoint = %s", endpoint)
	k.log.Infof("GENERATE_KUBECONFIG: IsHACluster = %v", IsHACluster(k.bkeCluster, k.nodes))

	if err := k.createInitialKubeConfig(endpoint); err != nil {
		return err
	}

	// 是ha集群在设置一个以域名为主机名的kubeconfig，非域名的供程序使用，域名的给节点使用，同时需要环境初始化配置域名解析
	if IsHACluster(k.bkeCluster, k.nodes) {
		return k.handleHAKubeConfig()
	}

	k.log.Infof("GENERATE_KUBECONFIG: not a HA cluster, skipping HA processing")
	k.log.Infof("GENERATE_KUBECONFIG: kubeconfig generation completed successfully")
	return nil
}

// createInitialKubeConfig creates the initial kubeconfig secret
func (k *BKEKubernetesCertGenerator) createInitialKubeConfig(endpoint string) error {
	err := kubeconfig.CreateSecretWithOwner(k.ctx, k.client, util.ObjectKey(k.bkeCluster),
		endpoint, metav1.OwnerReference{
			APIVersion: bkev1beta1.GroupVersion.String(),
			Kind:       "BKECluster",
			Name:       k.bkeCluster.Name,
			UID:        k.bkeCluster.UID,
		})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		k.log.Errorf("GENERATE_KUBECONFIG: failed to create kubeconfig secret: %v", err)
		return fmt.Errorf("failed to create kubeconfig secret: %w", err)
	}
	k.log.Infof("GENERATE_KUBECONFIG: kubeconfig secret created successfully")
	return nil
}

func (k *BKEKubernetesCertGenerator) handleHAKubeConfig() error {
	k.log.Infof("GENERATE_KUBECONFIG: processing HA cluster kubeconfig")

	secretName := NewCertSecretName(k.certClusterName, "kubeconfig")
	secretKey := types.NamespacedName{Name: secretName, Namespace: k.certNamespace}
	k.log.Infof("GENERATE_KUBECONFIG: getting kubeconfig secret %s/%s", k.certNamespace, secretName)
	// Write-through usually makes Create visible immediately; retry briefly in case
	// the Informer seed was skipped (nil cache / informer not ready) and WATCH lags.
	secret := &corev1.Secret{}
	getErr := phaseutil.RetryOnError(
		func() error {
			return k.client.Get(k.ctx, secretKey, secret)
		},
		phaseutil.WithBackoff(phaseutil.ReadBackoff()),
		phaseutil.WithPredicate(phaseutil.IsNotFound),
		phaseutil.WithOnRetry(func(attempt int, err error) {
			k.log.Debugf("GENERATE_KUBECONFIG: kubeconfig secret %s not in cache yet (%d/%d): %v",
				secretKey.String(), attempt, phaseutil.ReadBackoff().Steps, err)
		}),
	)
	if getErr != nil {
		k.log.Errorf("GENERATE_KUBECONFIG: failed to get kubeconfig secret %s/%s: %v",
			k.certNamespace, secretName, getErr)
		return fmt.Errorf("failed to get secret %q: %w", secretKey.String(), getErr)
	}
	k.log.Infof("GENERATE_KUBECONFIG: successfully retrieved kubeconfig secret %s/%s",
		k.certNamespace, secretName)

	if err := k.updateHAKubeConfig(secret, secretName); err != nil {
		return err
	}

	k.log.Infof("GENERATE_KUBECONFIG: kubeconfig generation completed successfully")
	return nil
}

// updateHAKubeConfig updates the HA kubeconfig with domain-based endpoint
func (k *BKEKubernetesCertGenerator) updateHAKubeConfig(secret *corev1.Secret, secretName string) error {
	kubeconfigBytes, ok := secret.Data["value"]
	if !ok {
		k.log.Errorf("GENERATE_KUBECONFIG: kubeconfig secret %s/%s missing 'value' field",
			k.certNamespace, secretName)
		return errors.Errorf("failed to find kubeconfig data %q in secret %q", "value",
			utils.ClientObjNS(secret))
	}
	k.log.Infof("GENERATE_KUBECONFIG: found 'value' field in kubeconfig secret %s/%s",
		k.certNamespace, secretName)

	kubeconfig, err := clientcmd.Load(kubeconfigBytes)
	if err != nil {
		k.log.Errorf("GENERATE_KUBECONFIG: failed to load kubeconfig from secret %s/%s: %v",
			k.certNamespace, secretName, err)
		return fmt.Errorf("failed to load kubeconfig: %w", err)
	}
	k.log.Infof("GENERATE_KUBECONFIG: successfully loaded kubeconfig from secret %s/%s",
		k.certNamespace, secretName)

	newEndpoint := fmt.Sprintf("https://%s:%d", constant.MasterHADomain,
		k.bkeCluster.Spec.ControlPlaneEndpoint.Port)
	k.log.Infof("GENERATE_KUBECONFIG: updating kubeconfig endpoint to %s", newEndpoint)
	kubeconfig.Clusters[k.certClusterName].Server = newEndpoint

	kubeconfigBytes, err = clientcmd.Write(*kubeconfig)
	if err != nil {
		k.log.Errorf("GENERATE_KUBECONFIG: failed to write kubeconfig: %v", err)
		return fmt.Errorf("failed to write kubeconfig: %w", err)
	}
	k.log.Infof("GENERATE_KUBECONFIG: successfully wrote kubeconfig")

	k.log.Infof("GENERATE_KUBECONFIG: adding 'ha' field to kubeconfig secret %s/%s",
		k.certNamespace, secretName)
	err = phaseutil.RetryOnConflict(func() error {
		latestSecret := &corev1.Secret{}
		if err := k.client.Get(k.ctx, client.ObjectKey{Namespace: k.certNamespace, Name: secretName}, latestSecret); err != nil {
			return errors.Wrapf(err, "failed to get secret %q", utils.ClientObjNS(secret))
		}
		latestSecret.Data["ha"] = kubeconfigBytes
		return k.client.Update(k.ctx, latestSecret)
	})
	if err != nil {
		k.log.Errorf("GENERATE_KUBECONFIG: failed to update kubeconfig secret %s/%s: %v",
			k.certNamespace, secretName, err)
		return errors.Wrapf(err, "failed to update secret %q", utils.ClientObjNS(secret))
	}
	k.log.Infof("GENERATE_KUBECONFIG: successfully updated kubeconfig secret %s/%s with 'ha' field",
		k.certNamespace, secretName)
	return nil
}

func NewCertSecretName(clusterName, certName string) string {
	return fmt.Sprintf("%s-%s", clusterName, certName)
}

func HasServerAuth(cert *x509.Certificate) bool {
	for i := range cert.ExtKeyUsage {
		if cert.ExtKeyUsage[i] == x509.ExtKeyUsageServerAuth {
			return true
		}
	}
	return false
}

func IsHACluster(bkeCluster *bkev1beta1.BKECluster, nodes bkenode.Nodes) bool {
	if bkeCluster == nil || len(nodes) == 0 {
		return false
	}
	if bkeCluster.Spec.ControlPlaneEndpoint.IsValid() {
		host := bkeCluster.Spec.ControlPlaneEndpoint.Host
		if nodes.Filter(bkenode.FilterOptions{"IP": host}).Length() == 0 {
			return true
		}
	}
	return false
}

// LoadGlobalCA load global CA certs and key from secret
func (k *BKEKubernetesCertGenerator) LoadGlobalCA() (map[string][]byte, error) {
	if data, found, err := k.tryLoadGlobalCAFromSecret(); err != nil || found {
		return data, err
	}
	localData, err := k.loadLocalGlobalCA()
	if err != nil || localData == nil {
		return nil, err
	}
	if err := k.createGlobalCASecret(localData); err != nil && !apierrors.IsAlreadyExists(err) {
		return nil, err
	}
	k.isUserCustomCA = true
	return localData, nil
}

func (k *BKEKubernetesCertGenerator) tryLoadGlobalCAFromSecret() (map[string][]byte, bool, error) {
	secret := &corev1.Secret{}
	err := k.client.Get(k.ctx, types.NamespacedName{Namespace: GlobalCANamespace, Name: GlobalCASecretName}, secret)
	if err != nil {
		if apierrors.IsNotFound(err) {
			k.log.Infof("global CA secret %s/%s not found", GlobalCANamespace, GlobalCASecretName)
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("failed to get global CA secret %s/%s:%w", GlobalCANamespace, GlobalCASecretName, err)
	}
	if err := k.validateGlobalCASecret(secret); err != nil {
		k.log.Warnf("invalid global CA secret %s/%s: %v", GlobalCANamespace, GlobalCASecretName, err)
		return nil, false, nil
	}
	k.log.Infof("found valid global CA in secret %s/%s", GlobalCANamespace, GlobalCASecretName)
	k.isUserCustomCA = true
	return secret.Data, true, nil
}

func (k *BKEKubernetesCertGenerator) validateGlobalCASecret(secret *corev1.Secret) error {
	if secret.Data == nil {
		return errors.Errorf("secret has no data")
	}
	crtBytes, hasCert := secret.Data[TLSCrtDataName]
	keyBytes, hasKey := secret.Data[TLSKeyDataName]
	if !hasCert || !hasKey || len(crtBytes) == 0 || len(keyBytes) == 0 {
		return errors.Errorf("secret missing certificate or key")
	}
	if _, err := pkiutil.ParseCertsPEM(crtBytes); err != nil {
		return fmt.Errorf("invalid certificate: %w", err)
	}
	if _, err := pkiutil.ParsePrivateKeyPEM(keyBytes); err != nil {
		return fmt.Errorf("invalid private key: %w", err)
	}
	return nil
}

func (k *BKEKubernetesCertGenerator) loadLocalGlobalCA() (map[string][]byte, error) {
	crtBytes, err := os.ReadFile(GlobalCACertPath)
	if err != nil {
		k.log.Warnf("global CA cert file read error: path=%s, error=%v", GlobalCACertPath, err)
		return nil, nil
	}
	if len(crtBytes) == 0 {
		k.log.Warnf("global CA cert file is empty: path=%s", GlobalCACertPath)
		return nil, nil
	}
	keyBytes, err := os.ReadFile(GlobalCAKeyPath)
	if err != nil {
		k.log.Warnf("global CA key file read error: path=%s, error=%v", GlobalCAKeyPath, err)
		return nil, nil
	}
	if len(keyBytes) == 0 {
		k.log.Warnf("global CA key file is empty: path=%s", GlobalCAKeyPath)
		return nil, nil
	}
	if _, err := pkiutil.ParseCertsPEM(crtBytes); err != nil {
		k.log.Errorf("invalid local CA certificate: path=%s, error=%v", GlobalCACertPath, err)
		return nil, fmt.Errorf("invalid local CA certificate: %w", err)
	}
	if _, err := pkiutil.ParsePrivateKeyPEM(keyBytes); err != nil {
		k.log.Errorf("invalid local CA key: path=%s, error=%v", GlobalCAKeyPath, err)
		return nil, fmt.Errorf("invalid local CA key: %w", err)
	}
	chainBytes, err := os.ReadFile(CertChainPath)
	if err != nil {
		k.log.Warnf("global CA cert file read error: path=%s, error=%v", CertChainPath, err)
	}
	if len(chainBytes) == 0 {
		k.log.Warnf("global CA cert file is empty: path=%s", CertChainPath)
	}
	if _, err := pkiutil.ParseCertsPEM(chainBytes); err != nil {
		k.log.Errorf("invalid local chain certification: path=%s, error=%v", CertChainPath, err)
	}
	return map[string][]byte{TLSCrtDataName: crtBytes, TLSKeyDataName: keyBytes, ChainDataName: chainBytes}, nil
}

func (k *BKEKubernetesCertGenerator) createGlobalCASecret(data map[string][]byte) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Namespace: GlobalCANamespace, Name: GlobalCASecretName},
		Data:       data,
		Type:       agentutils.BKESecretType,
	}
	if k.bkeCluster != nil {
		controllerRef := metav1.NewControllerRef(k.bkeCluster, k.bkeCluster.GroupVersionKind())
		secret.SetOwnerReferences([]metav1.OwnerReference{*controllerRef})
	}
	if err := k.client.Create(k.ctx, secret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			k.log.Infof("global CA secret %s/%s already exists", GlobalCANamespace, GlobalCASecretName)
			return err
		}
		return fmt.Errorf("failed to create global CA secret %s/%s: %w", GlobalCANamespace, GlobalCASecretName, err)
	}
	k.log.Infof("created global CA secret %s/%s from local files", GlobalCANamespace, GlobalCASecretName)
	return nil
}

// SetCertsCAName set the CAName for the certs which are not self-signing
func (k *BKEKubernetesCertGenerator) SetCertsCAName() {
	for i, cert := range k.bkeCerts {
		if cert.BaseName == pkiutil.CACertAndKeyBaseName ||
			cert.BaseName == pkiutil.FrontProxyCACertAndKeyBaseName ||
			cert.BaseName == pkiutil.EtcdCACertAndKeyBaseName {
			if k.isUserCustomCA {
				k.bkeCerts[i].CAName = "global-ca"
				k.bkeCerts[i].IsCA = true
			}
		}
	}
}

// LoadConfigForCerts load user custom certification csr json and signing policy json
func (k *BKEKubernetesCertGenerator) LoadConfigForCerts() {
	loader := NewCertConfigLoader(k.ctx, k.client, k.bkeCluster, k.log)

	cfg, err := loader.LoadConfigMapData()
	if err != nil {
		k.log.Warnf("Failed to load certificate configuration from ConfigMap: %v", err)
	}

	if !k.hasAnyConfig(cfg) {
		cfg = k.loadFromLocalAndPersist(loader)
	}
	// Ensure cert definitions are prepared before applying configuration.
	// LoadConfigForCerts may be invoked early (e.g., from setupGlobalCA()),
	// before k.prepareBkeCerts has run. If certs are not prepared yet,
	// ApplyConfigToCerts will be skipped due to canApplyConfig check.
	if k.bkeCerts == nil {
		if err := k.prepareBkeCerts(false); err != nil {
			k.log.Warnf("Failed to prepare BKE certs before applying config: %v", err)
		}
	}

	if !k.canApplyConfig(cfg) {
		return
	}
	k.applyConfig(loader, cfg)
}

// hasAnyConfig check whether to have config files
func (k *BKEKubernetesCertGenerator) hasAnyConfig(cfg *CertConfigData) bool {
	if cfg == nil {
		return false
	}
	for _, ok := range cfg.AvailableKeys {
		if ok {
			return true
		}
	}
	return false
}

// loadFromLocalAndPersist load from local files if ConfigMap not exist
func (k *BKEKubernetesCertGenerator) loadFromLocalAndPersist(loader *CertConfigLoader) *CertConfigData {

	localData, err := loader.LoadLocalConfigData()
	if err != nil {
		k.log.Warnf("Failed to load local certificate configuration: %v", err)
		return nil
	}
	if localData == nil {
		k.log.Infof("Local certificate configuration not found. Using default cert logic.")
		return nil
	}
	return localData
}

// canApplyConfig determine if can get config successfully to apply to bkecert
func (k *BKEKubernetesCertGenerator) canApplyConfig(cfg *CertConfigData) bool {
	if k.bkeCerts == nil {
		return false
	}
	return cfg != nil
}

// applyConfig apply user custom config to bkecerts
func (k *BKEKubernetesCertGenerator) applyConfig(loader *CertConfigLoader, cfg *CertConfigData) {
	if err := loader.ApplyConfigToCerts(k.bkeCerts, cfg, k.certClusterName); err != nil {
		k.log.Warnf("Failed to apply certificate configuration: %v", err)
		return
	}
	k.log.Infof("Successfully loaded and applied certificate configuration")
}
