package helm

import (
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"

	helmModel "marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/utils/k8sutil"
)

func TestResetSyncStatusForTest(t *testing.T) {
	repoAsyncTaskMap.Store("repo", "running")
	ResetSyncStatusForTest()
	_, ok := repoAsyncTaskMap.Load("repo")
	assert.False(t, ok)
}

func TestUpdateAndClearSecretData(t *testing.T) {
	c := &helmClient{}
	patch := gomonkey.ApplyFunc(k8sutil.UpdateSecret, func(_ kubernetes.Interface, s *v1.Secret) (*v1.Secret, error) {
		return s, nil
	})
	defer patch.Reset()

	secret := &v1.Secret{
		Data: map[string][]byte{
			"a": []byte("abc"),
			"b": []byte("xyz"),
		},
	}
	updated, err := c.updateAndClearSecretData(secret)
	assert.NoError(t, err)
	assert.NotNil(t, updated)
	assert.Equal(t, []byte{0, 0, 0}, updated.Data["a"])
	assert.Equal(t, []byte{0, 0, 0}, updated.Data["b"])
}

func TestCreateRepoAuthTLSCAEmptyInput(t *testing.T) {
	c := &helmClient{}

	secret, err := c.createRepoCRBasicAuth("repo", nil, nil)
	assert.NoError(t, err)
	assert.Nil(t, secret)

	tlsRef, err := c.createRepoCRTLS("repo", "", "")
	assert.NoError(t, err)
	assert.Nil(t, tlsRef)

	caRef, err := c.createRepoCRCA("repo", "")
	assert.NoError(t, err)
	assert.Nil(t, caRef)
}

func TestUpdateRepoCRBasicAuthBranches(t *testing.T) {
	c := &helmClient{}
	repoCR := &helmModel.HelmChartRepository{
		Spec: helmModel.HelmChartRepositorySpec{},
	}

	ref, err := c.updateRepoCRBasicAuth(repoCR, "repo", nil, nil)
	assert.NoError(t, err)
	assert.Nil(t, ref)

	patchCreate := gomonkey.ApplyFunc(k8sutil.CreateSecret, func(_ kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
		return secret, nil
	})
	ref, err = c.updateRepoCRBasicAuth(repoCR, "repo", []byte("u"), []byte("p"))
	assert.NoError(t, err)
	assert.NotNil(t, ref)
	patchCreate.Reset()

	repoCR.Spec.BasicAuth.Name = "repo-auth"
	patchDelete := gomonkey.ApplyFunc(k8sutil.DeleteSecret, func(_ kubernetes.Interface, _, _ string) error { return nil })
	ref, err = c.updateRepoCRBasicAuth(repoCR, "repo", nil, nil)
	assert.NoError(t, err)
	assert.Nil(t, ref)
	patchDelete.Reset()

	patchUpdate := gomonkey.ApplyFunc(k8sutil.UpdateSecret, func(_ kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
		return secret, nil
	})
	ref, err = c.updateRepoCRBasicAuth(repoCR, "repo", []byte("u"), []byte("p"))
	assert.NoError(t, err)
	assert.NotNil(t, ref)
	patchUpdate.Reset()
}

func TestUpdateRepoCRTLSAndCABranches(t *testing.T) {
	c := &helmClient{}
	repoCR := &helmModel.HelmChartRepository{
		Spec: helmModel.HelmChartRepositorySpec{},
	}

	tlsRef, err := c.updateRepoCRTLS(repoCR, "repo", "", "")
	assert.NoError(t, err)
	assert.Nil(t, tlsRef)

	patchCreateTLS := gomonkey.ApplyFunc(k8sutil.CreateSecret, func(_ kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
		return secret, nil
	})
	tlsRef, err = c.updateRepoCRTLS(repoCR, "repo", "cert", "key")
	assert.NoError(t, err)
	assert.NotNil(t, tlsRef)
	patchCreateTLS.Reset()

	repoCR.Spec.TLS.Name = "repo-tls"
	patchDeleteTLS := gomonkey.ApplyFunc(k8sutil.DeleteSecret, func(_ kubernetes.Interface, _, _ string) error { return nil })
	tlsRef, err = c.updateRepoCRTLS(repoCR, "repo", "", "")
	assert.NoError(t, err)
	assert.Nil(t, tlsRef)
	patchDeleteTLS.Reset()

	patchUpdateTLS := gomonkey.ApplyFunc(k8sutil.UpdateSecret, func(_ kubernetes.Interface, secret *v1.Secret) (*v1.Secret, error) {
		return secret, nil
	})
	tlsRef, err = c.updateRepoCRTLS(repoCR, "repo", "cert", "key")
	assert.NoError(t, err)
	assert.NotNil(t, tlsRef)
	patchUpdateTLS.Reset()

	caRef, err := c.updateRepoCRCA(repoCR, "repo", "")
	assert.NoError(t, err)
	assert.Nil(t, caRef)

	repoCR.Spec.CA.Name = ""
	patchCreateCA := gomonkey.ApplyFunc(k8sutil.CreateConfigMap, func(_ kubernetes.Interface, cm *v1.ConfigMap) (*v1.ConfigMap, error) {
		return cm, nil
	})
	caRef, err = c.updateRepoCRCA(repoCR, "repo", "ca-data")
	assert.NoError(t, err)
	assert.NotNil(t, caRef)
	patchCreateCA.Reset()

	repoCR.Spec.CA.Name = "repo-ca"
	patchDeleteCA := gomonkey.ApplyFunc(k8sutil.DeleteConfigMap, func(_ kubernetes.Interface, _, _ string) error { return nil })
	caRef, err = c.updateRepoCRCA(repoCR, "repo", "")
	assert.NoError(t, err)
	assert.Nil(t, caRef)
	patchDeleteCA.Reset()

	patchUpdateCA := gomonkey.ApplyFunc(k8sutil.UpdateConfigMap, func(_ kubernetes.Interface, cm *v1.ConfigMap) (*v1.ConfigMap, error) {
		return cm, nil
	})
	caRef, err = c.updateRepoCRCA(repoCR, "repo", "ca-data")
	assert.NoError(t, err)
	assert.NotNil(t, caRef)
	patchUpdateCA.Reset()
}

func TestUpdateAndClearSecretDataError(t *testing.T) {
	c := &helmClient{}
	patch := gomonkey.ApplyFunc(k8sutil.UpdateSecret, func(_ kubernetes.Interface, _ *v1.Secret) (*v1.Secret, error) {
		return nil, errors.New("update failed")
	})
	defer patch.Reset()

	updated, err := c.updateAndClearSecretData(&v1.Secret{Data: map[string][]byte{"x": []byte("y")}})
	assert.Error(t, err)
	assert.Nil(t, updated)
}
