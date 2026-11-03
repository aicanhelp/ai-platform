package helm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestHelmChartRepositoryDeepCopy(t *testing.T) {
	origin := &HelmChartRepository{
		TypeMeta: metav1.TypeMeta{
			Kind:       "HelmChartRepository",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "repo-1",
			Namespace: "default",
		},
		Spec: HelmChartRepositorySpec{
			DisplayName: "repo",
			URL:         "https://example.com",
			BasicAuth:   SecretReference{Name: "secret"},
			TLS:         SecretReference{Name: "tls"},
			CA:          ConfigMapReference{Name: "ca"},
		},
	}

	cp := origin.DeepCopy()
	assert.NotNil(t, cp)
	assert.Equal(t, origin.Name, cp.Name)
	assert.Equal(t, origin.Spec.URL, cp.Spec.URL)

	cp.Name = "repo-2"
	assert.NotEqual(t, origin.Name, cp.Name)
}

func TestDeepCopyNilBranches(t *testing.T) {
	var repo *HelmChartRepository
	assert.Nil(t, repo.DeepCopy())
	assert.Nil(t, repo.DeepCopyObject())

	var spec *HelmChartRepositorySpec
	assert.Nil(t, spec.DeepCopy())

	var secret *SecretReference
	assert.Nil(t, secret.DeepCopy())

	var cfg *ConfigMapReference
	assert.Nil(t, cfg.DeepCopy())
}

func TestOtherDeepCopyFunctions(t *testing.T) {
	spec := &HelmChartRepositorySpec{
		DisplayName: "name",
		URL:         "url",
		BasicAuth:   SecretReference{Name: "b"},
		TLS:         SecretReference{Name: "t"},
		CA:          ConfigMapReference{Name: "c"},
	}
	specOut := &HelmChartRepositorySpec{}
	spec.DeepCopyInto(specOut)
	assert.Equal(t, spec.URL, specOut.URL)

	sec := &SecretReference{Name: "sec"}
	secOut := &SecretReference{}
	sec.DeepCopyInto(secOut)
	assert.Equal(t, sec.Name, secOut.Name)

	cm := &ConfigMapReference{Name: "cm"}
	cmOut := &ConfigMapReference{}
	cm.DeepCopyInto(cmOut)
	assert.Equal(t, cm.Name, cmOut.Name)
}

func TestAddHelmChartRepositoryToScheme(t *testing.T) {
	scheme := runtime.NewScheme()
	err := AddHelmChartRepositoryToScheme(scheme)
	assert.NoError(t, err)

	obj := &HelmChartRepository{}
	gvks, _, err := scheme.ObjectKinds(obj)
	assert.NoError(t, err)
	assert.NotEmpty(t, gvks)
}

