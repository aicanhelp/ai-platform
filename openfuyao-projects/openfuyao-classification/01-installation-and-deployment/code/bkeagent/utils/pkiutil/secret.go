package pkiutil

import (
	"context"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"github.com/wsva/lib_go/crypto"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

const (
	// CertChainFileName defines certificate chain name
	CertChainFileName = "trust-chain.crt"
	// ChainCrtDataName is the key used to store a certificate chain in the secret's data field.
	ChainCrtDataName = "trust-chain.crt"
)

func StoreClusterAPICert(secret *corev1.Secret, pkiPath string) error {
	bkeCert := convertSecretCertToBKECert(secret)
	if bkeCert == nil {
		return errors.Errorf("can't find bke cert for secret %q", secret.Name)
	}
	bkeCert.PkiPath = pkiPath

	switch {
	case bkeCert.BaseName == ServiceAccountKeyBaseName:
		return storeServiceAccountCert(secret, bkeCert)
	case bkeCert.BaseName == AdminKubeConfigFileName && bkeCert.Name == LocalKubeConfigFileName:
		return storeKubeConfigCert(secret, bkeCert)
	default:
		return storeDefaultCert(secret, bkeCert)
	}
}

// storeServiceAccountCert handles service account certificate storage
// It processes RSA public/private key pairs and writes them to the appropriate location
func storeServiceAccountCert(secret *corev1.Secret, bkeCert *BKECert) error {
	log.Infof("store sa certification for cluster-api secret %q", secret.Name)

	var pubKey *rsa.PublicKey
	var certificateKey []*x509.Certificate
	isRSACertificate := true

	pubKey, err := ParsePublicKeyPEM(secret.Data[TLSCrtDataName])
	if err != nil {
		isRSACertificate = false
		certificateKey, err = crypto.ParseCertsPEM(secret.Data[TLSCrtDataName])
		if err != nil {
			return err
		}
	}

	priKeyInterface, err := crypto.ParsePrivateKeyPEM(secret.Data[TLSKeyDataName])
	if err != nil {
		return err
	}

	// Type assertion to ensure it's an RSA private key
	priKey, ok := priKeyInterface.(*rsa.PrivateKey)
	if !ok {
		return fmt.Errorf("expected RSA private key, got %T", priKeyInterface)
	}

	err = writePubAndKey(bkeCert, certificateKey, pubKey, priKey, isRSACertificate)
	if err != nil {
		return err
	}

	log.Infof("store cluster api cert %q", bkeCert.Name)
	return nil
}

// storeKubeConfigCert handles kubeconfig certificate storage
// It processes kubeconfig data from either 'ha' or 'value' fields in the secret
func storeKubeConfigCert(secret *corev1.Secret, bkeCert *BKECert) error {
	log.Infof("store kubeconfig for cluster-api secret %q", secret.Name)

	var data []byte
	if secret.Data["ha"] != nil {
		log.Infof("store kubeconfig ha data for secret %q", secret.Name)
		data = secret.Data["ha"]
	} else {
		log.Infof("store kubeconfig value data for secret %q", secret.Name)
		data = secret.Data["value"]
	}

	if err := writeKubeConfig(bkeCert, data); err != nil {
		return err
	}

	log.Infof("store cluster api cert %q", bkeCert.Name)
	return nil
}

// storeDefaultCert handles general TLS certificate storage
// It processes standard X.509 certificates and RSA private keys
func storeDefaultCert(secret *corev1.Secret, bkeCert *BKECert) error {
	log.Infof("store certificate for cluster-api secret %q", secret.Name)

	certs, err := crypto.ParseCertsPEM(secret.Data[TLSCrtDataName])
	if err != nil {
		return err
	}

	keyInterface, err := crypto.ParsePrivateKeyPEM(secret.Data[TLSKeyDataName])
	if err != nil {
		return err
	}

	priKey, ok := keyInterface.(*rsa.PrivateKey)
	if !ok {
		return fmt.Errorf("expected RSA private key, got %T", keyInterface)
	}

	if err := WriteCertAndKey(bkeCert, certs[0], priKey); err != nil {
		return err
	}

	log.Infof("store cluster api cert %q", bkeCert.Name)
	return nil
}

// convertSecretCertToBKECert converts a Kubernetes secret to a BKECert object
// It matches the secret name against the list of known BKE certificates and kubeconfigs
// Special handling is applied for kubeconfig certificates which are renamed to "admin"
func convertSecretCertToBKECert(secret *corev1.Secret) *BKECert {
	bkeCertList := GetClusterAPICertList()
	kubeConfigs := GetKubeConfigs()
	for i, kubeConfig := range kubeConfigs {
		if kubeConfig.Name == "kubeconfig" {
			kubeConfigs[i].Name = "admin"
		}
	}
	bkeCertList = append(bkeCertList, kubeConfigs...)
	for _, bkeCert := range bkeCertList {
		if strings.HasSuffix(secret.GetName(), bkeCert.Name) {
			return bkeCert
		}
	}
	return nil
}

// UploadBocloudCertToClusterAPI uploads Bocloud certificate files to the cluster as Kubernetes secrets
// It reads certificate and key files from the filesystem and creates a secret in the specified namespace
// If the secret already exists, it logs the information and continues without error
func UploadBocloudCertToClusterAPI(c client.Client, certSpec *BocloudCert, namespace, clusterName string) error {
	crtFile, err := os.ReadFile(pathForBocloudCert(certSpec))
	if err != nil {
		return errors.Wrapf(err, "failed to read certificate %q cert file %q", certSpec.Name, pathForBocloudCert(certSpec))
	}

	keyFile, err := os.ReadFile(pathForBocloudKey(certSpec))
	if err != nil {
		return errors.Wrapf(err, "failed to read certificate %q key file %q", certSpec.Name, pathForBocloudKey(certSpec))
	}

	secretName := fmt.Sprintf("%s-%s", clusterName, certSpec.Name)
	certSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		Data: map[string][]byte{
			TLSCrtDataName: crtFile,
			TLSKeyDataName: keyFile,
		},
		Type: utils.BKESecretType,
	}

	if err := c.Create(context.Background(), certSecret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			log.Infof("cert secret %q already exists, skip", secretName)
			return nil
		}
		return errors.Wrapf(err, "failed to create cert secret %q", secretName)
	}
	log.Infof("upload cert %q to manager cluster, cert secret %q ", certSpec.Name, secretName)
	return nil
}

// UploadBKECertToClusterAPI uploads BKE certificate files to the cluster as Kubernetes secrets
// It handles different certificate types including service account keys which use public keys instead of certificates
// If the secret already exists, it logs the information and continues without error
func UploadBKECertToClusterAPI(c client.Client, certSpec *BKECert, namespace, clusterName string) error {

	var crtFile []byte
	var err error
	if certSpec.BaseName == ServiceAccountKeyBaseName {
		crtFile, err = os.ReadFile(pathForPublicKey(certSpec))
		if err != nil {
			return errors.Wrapf(err, "failed to read certificate %q cert file %q", certSpec.Name, pathForCert(certSpec))
		}
	} else {
		crtFile, err = os.ReadFile(pathForCert(certSpec))
		if err != nil {
			return errors.Wrapf(err, "failed to read certificate %q cert file %q", certSpec.Name, pathForCert(certSpec))
		}
	}

	keyFile, err := os.ReadFile(pathForKey(certSpec))
	if err != nil {
		return errors.Wrapf(err, "failed to read certificate %q key file %q", certSpec.Name, pathForKey(certSpec))
	}

	secretName := fmt.Sprintf("%s-%s", clusterName, certSpec.Name)
	certSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		Data: map[string][]byte{
			TLSCrtDataName: crtFile,
			TLSKeyDataName: keyFile,
		},
		Type: utils.BKESecretType,
	}

	if err := c.Create(context.Background(), certSecret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			log.Infof("cert secret %q already exists, skip", secretName)
			return nil
		}
		return errors.Wrapf(err, "failed to create cert secret %q", secretName)
	}
	log.Infof("upload cert %q to manager cluster, cert secret %q ", certSpec.Name, secretName)
	return nil
}

// SaveGlobalCAAndCertChainToSecret save local global ca and certificate chain to cluster secret
func SaveGlobalCAAndCertChainToSecret(c client.Client, certSpec *BKECert) error {
	crtFile, err := os.ReadFile(pathForCert(certSpec))
	if err != nil {
		return errors.Wrapf(err, "failed to read certificate %q cert file %q", certSpec.Name, pathForCert(certSpec))
	}
	keyFile, err := os.ReadFile(pathForKey(certSpec))
	if err != nil {
		return errors.Wrapf(err, "failed to read certificate %q key file %q", certSpec.Name, pathForKey(certSpec))
	}

	secretData := map[string][]byte{
		TLSCrtDataName: crtFile,
		TLSKeyDataName: keyFile,
	}

	chainPath := filepath.Join(certSpec.PkiPath, CertChainFileName)
	if utils.Exists(chainPath) {
		chainFile, err := os.ReadFile(chainPath)
		if err != nil {
			log.Warnf("failed to read certificate chain file %q: %v", chainPath, err)
		} else {
			secretData[ChainCrtDataName] = chainFile
			log.Infof("certificate chain file found and will be included in secret")
		}
	}

	certSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		Data: secretData,
		Type: corev1.SecretTypeOpaque,
	}

	if err := c.Create(context.Background(), certSecret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			log.Infof("cert secret %q already exists, skip", utils.GlobalCASecretName)
			return nil
		}
		return errors.Wrapf(err, "failed to create cert secret %q", utils.GlobalCASecretName)
	}
	log.Infof("upload cert %q to manager cluster, cert secret %q ", utils.GlobalCANamespace, utils.GlobalCASecretName)
	return nil
}
