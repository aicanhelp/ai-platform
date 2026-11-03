package helm

import (
	"bytes"
	"errors"
	"net/http"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/repo"
	"k8s.io/client-go/kubernetes"

	"marketplace-service/pkg/constant"
	marketplaceErrors "marketplace-service/pkg/errors"
	models "marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/utils/util"
)

func TestGetUploadErrorGeneralResponse(t *testing.T) {
	resp := getUploadErrorGeneralResponse()
	assert.Equal(t, int32(constant.ClientError), resp.Code)
	assert.Contains(t, resp.Msg, ".tgz")
}

func TestGetTokenNotFoundError(t *testing.T) {
	err := getTokenNotFoundError([]byte(`{"x":"y"}`))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "token")
}

func TestNeedShowAndGetChartVersions(t *testing.T) {
	entry := &repo.ChartVersion{Metadata: &chart.Metadata{Version: "1.0.0"}}
	assert.True(t, needShow(entry, ""))
	assert.True(t, needShow(entry, "1.0.0"))
	assert.False(t, needShow(entry, "2.0.0"))

	c := &helmClient{
		dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
		clientset:     mockClientset(),
	}
	cachedData = NewChartCache()
	cachedData.SetChartCache(mockRemoteRepoCRDisplayName, &repo.IndexFile{
		Entries: map[string]repo.ChartVersions{
			"demo": {
				{Metadata: &chart.Metadata{Name: "demo", Version: "1.0.0"}},
				{Metadata: &chart.Metadata{Name: "demo", Version: "1.1.0"}},
			},
		},
	})

	resp, status := c.GetChartVersions(mockRemoteRepoCRName, "demo")
	assert.Equal(t, http.StatusOK, status)
	assert.Equal(t, int32(constant.Success), resp.Code)
	list, ok := resp.Data.([]models.ChartVersionResponse)
	assert.True(t, ok)
	assert.Len(t, list, 2)

	resp, status = c.GetChartVersion(mockRemoteRepoCRName, "demo", "1.1.0")
	assert.Equal(t, http.StatusOK, status)
	list, ok = resp.Data.([]models.ChartVersionResponse)
	assert.True(t, ok)
	assert.Len(t, list, 1)
}

func TestGetChartVersionErrors(t *testing.T) {
	c := &helmClient{dynamicClient: mockDynamicClient()}
	resp, status := c.GetChartVersion("x", "demo", "")
	assert.Equal(t, http.StatusNotFound, status)
	assert.Equal(t, int32(constant.ResourceNotFound), resp.Code)
}

func TestGetChartBytesByVersion(t *testing.T) {
	c := &helmClient{
		dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
		clientset:     mockClientset(),
	}
	cachedData = NewChartCache()
	cachedData.SetChartCache(mockRemoteRepoCRName, &repo.IndexFile{
		Entries: map[string]repo.ChartVersions{
			"demo": {{Metadata: &chart.Metadata{Version: "1.0.0"}, URLs: []string{"http://example/chart.tgz"}}},
		},
	})

	patches := gomonkey.ApplyFunc(LoadChartBytes, func(_ string, _ *repo.Entry) (*bytes.Buffer, error) {
		return bytes.NewBufferString("tarball"), nil
	})
	defer patches.Reset()

	buf, err := c.GetChartBytesByVersion(mockRemoteRepoCRName, "demo", "1.0.0")
	assert.NoError(t, err)
	assert.Equal(t, "tarball", buf.String())

	buf, err = c.GetChartBytesByVersion(mockRemoteRepoCRName, "demo", "2.0.0")
	assert.Error(t, err)
	assert.Nil(t, buf)
}

func TestGetChartDetailAndTemplates(t *testing.T) {
	c := &helmClient{}
	ch := &chart.Chart{
		Raw: []*chart.File{
			{Name: "README.md", Data: []byte("doc")},
			{Name: "values.yaml", Data: []byte("a: b")},
			{Name: "Chart.yaml", Data: []byte("name: demo")},
			{Name: "templates/deploy.yaml", Data: []byte("template")},
		},
	}

	detail := c.getChartDetail(ch)
	assert.Equal(t, "doc", detail.Readme)
	assert.Equal(t, "a: b", detail.Values)
	assert.Equal(t, "name: demo", detail.Chart)

	templates := c.getChartTemplates(ch)
	assert.Len(t, templates, 1)
	assert.Equal(t, "templates/deploy.yaml", templates[0].Name)
}

func TestGetChartFilesBranches(t *testing.T) {
	c := &helmClient{}
	ch := &chart.Chart{
		Raw: []*chart.File{
			{Name: "README.md", Data: []byte("doc")},
			{Name: "templates/deploy.yaml", Data: []byte("template")},
		},
	}

	patch := gomonkey.ApplyFunc(LoadChart, func(_ string, _ *repo.Entry) (*chart.Chart, error) { return ch, nil })
	defer patch.Reset()

	c.dynamicClient = mockDynamicClientWithRepoCR(mockRemoteRepoCR())
	c.clientset = mockClientset()
	cachedData = NewChartCache()
	cachedData.SetChartCache(mockRemoteRepoCRName, &repo.IndexFile{
		Entries: map[string]repo.ChartVersions{
			"demo": {{Metadata: &chart.Metadata{Version: "1.0.0"}, URLs: []string{"http://example/chart.tgz"}}},
		},
	})

	resp, status := c.GetChartFiles(mockRemoteRepoCRName, "demo", "1.0.0", fileTypeDetail)
	assert.Equal(t, http.StatusOK, status)
	assert.IsType(t, &models.ChartDetailResponse{}, resp.Data)

	resp, status = c.GetChartFiles(mockRemoteRepoCRName, "demo", "1.0.0", fileTypeTemplate)
	assert.Equal(t, http.StatusOK, status)
	assert.IsType(t, []models.ChartTemplateResponse{}, resp.Data)

	resp, status = c.GetChartFiles(mockRemoteRepoCRName, "demo", "1.0.0", fileTypeAll)
	assert.Equal(t, http.StatusOK, status)
	assert.IsType(t, map[string]string{}, resp.Data)

	resp, status = c.GetChartFiles(mockRemoteRepoCRName, "demo", "1.0.0", "invalid")
	assert.Equal(t, http.StatusBadRequest, status)
	assert.Equal(t, int32(constant.ClientError), resp.Code)

	patch.Reset()
	patchErr := gomonkey.ApplyFunc(LoadChart, func(_ string, _ *repo.Entry) (*chart.Chart, error) {
		return nil, errors.New("missing")
	})
	defer patchErr.Reset()
	resp, status = c.GetChartFiles(mockRemoteRepoCRName, "demo", "1.0.0", fileTypeAll)
	assert.Equal(t, http.StatusInternalServerError, status)
	assert.Equal(t, int32(constant.ServerError), resp.Code)
}

func TestCountChartsBranches(t *testing.T) {
	c := &helmClient{}

	patchCfgErr := gomonkey.ApplyFunc(util.GetMarketplaceServiceConfig,
		func(_ kubernetes.Interface) (*models.MarketplaceServiceConfig, error) {
			return nil, errors.New("cfg error")
		})
	resp, status := c.CountCharts()
	assert.Equal(t, http.StatusInternalServerError, status)
	assert.Equal(t, int32(http.StatusInternalServerError), resp.Code)
	patchCfgErr.Reset()

	patchLimitErr := gomonkey.ApplyFunc(util.GetMarketplaceServiceConfig,
		func(_ kubernetes.Interface) (*models.MarketplaceServiceConfig, error) {
			return &models.MarketplaceServiceConfig{ChartLimit: "not-number"}, nil
		})
	resp, status = c.CountCharts()
	assert.Equal(t, http.StatusInternalServerError, status)
	assert.Equal(t, int32(http.StatusInternalServerError), resp.Code)
	patchLimitErr.Reset()

	patchCountErr := gomonkey.ApplyFunc(util.GetMarketplaceServiceConfig,
		func(_ kubernetes.Interface) (*models.MarketplaceServiceConfig, error) {
			return &models.MarketplaceServiceConfig{ChartLimit: "100"}, nil
		})
	c.dynamicClient = mockDynamicClientWithRepoCR(mockLocalRepoCR())
	c.clientset = mockClientset()
	resp, status = c.CountCharts()
	assert.Equal(t, http.StatusInternalServerError, status)
	assert.Equal(t, int32(http.StatusInternalServerError), resp.Code)
	patchCountErr.Reset()

	patchSuccess := gomonkey.ApplyFunc(util.GetMarketplaceServiceConfig,
		func(_ kubernetes.Interface) (*models.MarketplaceServiceConfig, error) {
			return &models.MarketplaceServiceConfig{
				ChartLimit:         "100",
				LocalHarborHost:    "http://local-harbor.openfuyao-system.svc.cluster.local",
				LocalHarborProject: "/chartrepo/library",
			}, nil
		}).ApplyFunc(getHttpRequestResponse, func(_ *http.Request, _ *repo.Entry) (*http.Response, []byte, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, []byte(`[{"name":"a","total_versions":8}]`), nil
	})
	cachedData = NewChartCache()
	cachedData.SetChartCache("local", mockRepoIndex())
	c.dynamicClient = mockDynamicClientWithRepoCR(mockLocalRepoCR())
	c.clientset = mockClientset()
	resp, status = c.CountCharts()
	assert.Equal(t, http.StatusOK, status)
	assert.Equal(t, int32(constant.Success), resp.Code)
	data, ok := resp.Data.(TotalResponse)
	assert.True(t, ok)
	assert.Equal(t, int64(8), data.Total)
	assert.Equal(t, int64(100), data.Limit)
	patchSuccess.Reset()
}

func TestUpdateOfficialHarborBearerTokenMissingHeader(t *testing.T) {
	c := &helmClient{}
	err := c.updateOfficialHarborBearerToken(&http.Response{Header: map[string][]string{}})
	assert.Error(t, err)
	var fieldErr *marketplaceErrors.FieldNotFoundError
	assert.ErrorAs(t, err, &fieldErr)
}
