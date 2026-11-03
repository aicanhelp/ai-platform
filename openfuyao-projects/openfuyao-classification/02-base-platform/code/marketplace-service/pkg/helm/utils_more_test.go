package helm

import (
	"errors"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/kube"
	"helm.sh/helm/v3/pkg/release"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	models "marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
)

func TestPaging(t *testing.T) {
	items := []*release.Release{
		{Name: "a"},
		{Name: "b"},
		{Name: "c"},
	}

	paged, totalPages, total := Paging(items, nil)
	assert.Len(t, paged, 3)
	assert.Equal(t, 1, totalPages)
	assert.Equal(t, 3, total)

	paged, totalPages, total = Paging(items, &param.Pagination{Limit: 2, Offset: 0, CurrentPage: 1})
	assert.Len(t, paged, 2)
	assert.Equal(t, 2, totalPages)
	assert.Equal(t, 3, total)
}

func TestGetResourceInfo(t *testing.T) {
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "demo",
				"namespace": "default",
				"uid":       "u1",
			},
		},
	}
	obj.SetGroupVersionKind(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"})

	info := getResourceInfo(obj, obj)
	assert.Equal(t, "demo", info.Name)
	assert.Equal(t, "default", info.Namespace)
	assert.Equal(t, "u1", info.Uid)
	assert.Equal(t, "apps/v1", info.APIVersion)
	assert.Equal(t, "Deployment", info.Kind)
}

func TestGetResourcesFromManifestReachableError(t *testing.T) {
	cfg := &action.Configuration{KubeClient: &kube.Client{}}
	patch := gomonkey.ApplyMethod(reflect.TypeOf(&kube.Client{}), "IsReachable", func(_ *kube.Client) error {
		return errors.New("unreachable")
	})
	defer patch.Reset()

	resources, err := GetResourcesFromManifest(cfg, "apiVersion: v1")
	assert.Error(t, err)
	assert.Nil(t, resources)
}

func TestConvertToReleaseWithShowResource(t *testing.T) {
	rel := &release.Release{
		Name:      "r1",
		Namespace: "default",
		Version:   1,
		Info: &release.Info{
			Status:      release.StatusDeployed,
			Notes:       "notes",
			Description: "desc",
		},
	}
	patch := gomonkey.ApplyFunc(GetResourcesFromManifest, func(_ *action.Configuration, _ string) ([]models.ReleaseResource, error) {
		return []models.ReleaseResource{{Name: "r"}}, nil
	})
	defer patch.Reset()
	cfg := &action.Configuration{}
	out := ConvertToRelease(cfg, rel, true)
	assert.NotNil(t, out)
	assert.Equal(t, "r1", out.Name)
	assert.Len(t, out.Resources, 1)

	patchErr := gomonkey.ApplyFunc(GetResourcesFromManifest, func(_ *action.Configuration, _ string) ([]models.ReleaseResource, error) {
		return nil, errors.New("failed")
	})
	defer patchErr.Reset()
	out = ConvertToRelease(cfg, rel, true)
	assert.NotNil(t, out)
}

func TestConvertToReleaseHistoryInfoFields(t *testing.T) {
	rel := &release.Release{
		Name:      "r1",
		Namespace: "ns",
		Version:   2,
		Info: &release.Info{
			Status: release.StatusDeployed,
		},
	}
	h := ConvertToReleaseHistory(rel)
	assert.Equal(t, "deployed", h.Status)
	assert.Equal(t, "r1", h.Name)
}
