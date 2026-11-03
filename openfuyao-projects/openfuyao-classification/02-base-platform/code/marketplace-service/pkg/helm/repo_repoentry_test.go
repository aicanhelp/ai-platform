package helm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	"marketplace-service/pkg/constant"
	models "marketplace-service/pkg/models/helm"
)

func TestRepoCRtoRepoEntry(t *testing.T) {
	clientset := clientsetfake.NewSimpleClientset(
		&v1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "auth", Namespace: "openfuyao-system"},
			Data: map[string][]byte{
				mapKeyUsername: []byte("u"),
				mapKeyPassword: []byte("p"),
			},
		},
		&v1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "tls", Namespace: "openfuyao-system"},
			Data: map[string][]byte{
				mapKeyTLSCrt: []byte("crt"),
				mapKeyTLSKey: []byte("key"),
			},
		},
		&v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "ca", Namespace: "openfuyao-system"},
			Data: map[string]string{
				mapKeyCAKey: "ca-data",
			},
		},
	)
	c := &helmClient{clientset: clientset}

	repoCR := &models.HelmChartRepository{
		Spec: models.HelmChartRepositorySpec{
			DisplayName:           "repo",
			URL:                   "https://repo",
			InsecureSkipTLSVerify: true,
			PassCredentialsAll:    true,
			BasicAuth:             models.SecretReference{Name: "auth"},
			TLS:                   models.SecretReference{Name: "tls"},
			CA:                    models.ConfigMapReference{Name: "ca"},
		},
	}
	entry, err := c.repoCRtoRepoEntry(repoCR)
	assert.NoError(t, err)
	assert.Equal(t, "repo", entry.Name)
	assert.Equal(t, "https://repo", entry.URL)
	assert.Equal(t, "u", entry.Username)
	assert.Equal(t, "p", entry.Password)
	assert.Equal(t, "crt", entry.CertFile)
	assert.Equal(t, "key", entry.KeyFile)
	assert.Equal(t, "ca-data", entry.CAFile)
	assert.True(t, entry.InsecureSkipTLSverify)
	assert.True(t, entry.PassCredentialsAll)
}

func TestRepoCRtoRepoEntryErrors(t *testing.T) {
	clientset := clientsetfake.NewSimpleClientset()
	c := &helmClient{clientset: clientset}

	_, err := c.repoCRtoRepoEntry(&models.HelmChartRepository{
		Spec: models.HelmChartRepositorySpec{
			DisplayName: "repo",
			URL:         "https://repo",
			BasicAuth:   models.SecretReference{Name: "missing-auth"},
		},
	})
	assert.Error(t, err)

	_, err = c.repoCRtoRepoEntry(&models.HelmChartRepository{
		Spec: models.HelmChartRepositorySpec{
			DisplayName: "repo",
			URL:         "https://repo",
			TLS:         models.SecretReference{Name: "missing-tls"},
		},
	})
	assert.Error(t, err)

	_, err = c.repoCRtoRepoEntry(&models.HelmChartRepository{
		Spec: models.HelmChartRepositorySpec{
			DisplayName: "repo",
			URL:         "https://repo",
			CA:          models.ConfigMapReference{Name: "missing-ca"},
		},
	})
	assert.Error(t, err)
}

func TestGetRepoBranches(t *testing.T) {
	c := &helmClient{dynamicClient: mockDynamicClient()}

	resp, status := c.GetRepo("")
	assert.Equal(t, 400, status)
	assert.Equal(t, int32(constant.ClientError), resp.Code)

	resp, status = c.GetRepo("missing")
	assert.Equal(t, 404, status)
	assert.Equal(t, int32(constant.ResourceNotFound), resp.Code)

	c = &helmClient{dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR())}
	resp, status = c.GetRepo(mockRemoteRepoCRName)
	assert.Equal(t, 200, status)
	assert.Equal(t, int32(constant.Success), resp.Code)
}
