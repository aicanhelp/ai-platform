package v1beta1

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"marketplace-service/pkg/helm"
	helmModel "marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
)

type mockOperation struct {
	listRepo func(query *param.Query, repo string) (*httputil.ResponseJson, int)
	getRepo  func(repo string) (*httputil.ResponseJson, int)

	createRepo func(*helmModel.SafeRepoEntry) (*httputil.ResponseJson, int)
	updateRepo func(*helmModel.SafeRepoEntry) (*httputil.ResponseJson, int)
	deleteRepo func(repo string) (*httputil.ResponseJson, int)

	syncAllRepos    func() (*httputil.ResponseJson, int)
	syncRepo        func(repo string) (*httputil.ResponseJson, int)
	getRepoSyncInfo func(repo string) (*httputil.ResponseJson, int)

	getLatestCharts           func(searchParam *helmModel.ChartSearchParam) (*httputil.ResponseJson, int)
	getChartsWithOfficialTags func(tags []string, query *param.Query) (*httputil.ResponseJson, int)
	uploadChart               func(formFile multipart.File, fileHeader *multipart.FileHeader) (*httputil.ResponseJson, int)
	countCharts               func() (*httputil.ResponseJson, int)
	deleteChart               func(chart string) (*httputil.ResponseJson, int)
	deleteChartVersion        func(chart, version string) (*httputil.ResponseJson, int)
	getChartVersions          func(repoName, chart string) (*httputil.ResponseJson, int)
	getChartVersion           func(repoName, chart, version string) (*httputil.ResponseJson, int)
	getChartFiles             func(repoName, chart, version, fileType string) (*httputil.ResponseJson, int)
}

func (m *mockOperation) CreateRepo(entry *helmModel.SafeRepoEntry) (*httputil.ResponseJson, int) {
	if m.createRepo != nil {
		return m.createRepo(entry)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) UpdateRepo(entry *helmModel.SafeRepoEntry) (*httputil.ResponseJson, int) {
	if m.updateRepo != nil {
		return m.updateRepo(entry)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) DeleteRepo(repoName string) (*httputil.ResponseJson, int) {
	if m.deleteRepo != nil {
		return m.deleteRepo(repoName)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) ListRepo(query *param.Query, repo string) (*httputil.ResponseJson, int) {
	if m.listRepo != nil {
		return m.listRepo(query, repo)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetRepo(repoName string) (*httputil.ResponseJson, int) {
	if m.getRepo != nil {
		return m.getRepo(repoName)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) SyncAllRepos() (*httputil.ResponseJson, int) {
	if m.syncAllRepos != nil {
		return m.syncAllRepos()
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) SyncRepo(repoName string) (*httputil.ResponseJson, int) {
	if m.syncRepo != nil {
		return m.syncRepo(repoName)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetRepoSyncStatus(repoName string) (*httputil.ResponseJson, int) {
	if m.getRepoSyncInfo != nil {
		return m.getRepoSyncInfo(repoName)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetLatestCharts(searchParam *helmModel.ChartSearchParam) (*httputil.ResponseJson, int) {
	if m.getLatestCharts != nil {
		return m.getLatestCharts(searchParam)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetChartsWithOfficialTags(tags []string, query *param.Query) (*httputil.ResponseJson, int) {
	if m.getChartsWithOfficialTags != nil {
		return m.getChartsWithOfficialTags(tags, query)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) CountCharts() (*httputil.ResponseJson, int) {
	if m.countCharts != nil {
		return m.countCharts()
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) UploadChart(formFile multipart.File, fileHeader *multipart.FileHeader) (*httputil.ResponseJson, int) {
	if m.uploadChart != nil {
		return m.uploadChart(formFile, fileHeader)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) DeleteChart(chartName string) (*httputil.ResponseJson, int) {
	if m.deleteChart != nil {
		return m.deleteChart(chartName)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) DeleteChartVersion(chartName, version string) (*httputil.ResponseJson, int) {
	if m.deleteChartVersion != nil {
		return m.deleteChartVersion(chartName, version)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetChartVersions(repoName, chartName string) (*httputil.ResponseJson, int) {
	if m.getChartVersions != nil {
		return m.getChartVersions(repoName, chartName)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetChartVersion(repoName, chartName, version string) (*httputil.ResponseJson, int) {
	if m.getChartVersion != nil {
		return m.getChartVersion(repoName, chartName, version)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetChartFiles(repoName, chartName, version, fileType string) (*httputil.ResponseJson, int) {
	if m.getChartFiles != nil {
		return m.getChartFiles(repoName, chartName, version, fileType)
	}
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}
func (m *mockOperation) GetChartBytesByVersion(_, _, _ string) (*bytes.Buffer, error) {
	return bytes.NewBuffer(nil), nil
}

func newHandlerContainer(h *Handler) *restful.Container {
	ws := restful.WebService{}
	ws.Path("/").Consumes("*/*").Produces(restful.MIME_JSON)
	bindHelmChartsRoute(&ws, h)
	bindHelmRepoGetRoutes(&ws, h)
	bindHelmRepoPostRoutes(&ws, h)

	container := restful.NewContainer()
	container.Add(&ws)
	return container
}

func TestHandlerRoutes(t *testing.T) {
	m := &mockOperation{}
	container := newHandlerContainer(newHandler(m))

	tests := []struct {
		name   string
		method string
		path   string
		body   io.Reader
		want   int
	}{
		{"list repo", http.MethodGet, "/helm-repos", nil, http.StatusOK},
		{"invalid list repo search", http.MethodGet, "/helm-repos?repo=<script>", nil, http.StatusOK},
		{"get repo", http.MethodGet, "/helm-repos/repo1", nil, http.StatusOK},
		{"sync all", http.MethodPost, "/helm-repos/sync", nil, http.StatusOK},
		{"sync one", http.MethodPost, "/helm-repos/repo1/sync", nil, http.StatusOK},
		{"sync status", http.MethodGet, "/helm-repos/repo1/sync", nil, http.StatusOK},
		{"count charts", http.MethodGet, "/helm-charts/count", nil, http.StatusOK},
		{"delete chart", http.MethodDelete, "/helm-charts/chart1", nil, http.StatusOK},
		{"delete chart version", http.MethodDelete, "/helm-charts/chart1/versions/1.0.0", nil, http.StatusOK},
		{"chart versions", http.MethodGet, "/helm-repos/repo1/charts/chart1", nil, http.StatusOK},
		{"chart version", http.MethodGet, "/helm-repos/repo1/charts/chart1/versions/1.0.0", nil, http.StatusOK},
		{"chart files", http.MethodGet, "/helm-repos/repo1/charts/chart1/versions/1.0.0/files?fileType=template", nil, http.StatusOK},
		{"latest charts", http.MethodGet, "/helm-charts?chart=chart1", nil, http.StatusOK},
		{"invalid latest charts search", http.MethodGet, "/helm-charts?chart=<script>", nil, http.StatusOK},
		{"official tags", http.MethodGet, "/helm-charts/official-tags?tag=abc", nil, http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, tt.body)
			req.Header.Set("Accept", restful.MIME_JSON)
			resp := httptest.NewRecorder()
			container.ServeHTTP(resp, req)
			assert.Equal(t, tt.want, resp.Code)
		})
	}
}

func TestCreateAndUpdateRepo(t *testing.T) {
	createCalled := false
	updateCalled := false
	m := &mockOperation{
		createRepo: func(entry *helmModel.SafeRepoEntry) (*httputil.ResponseJson, int) {
			createCalled = true
			assert.Equal(t, "repo1", entry.Name)
			return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
		},
		updateRepo: func(entry *helmModel.SafeRepoEntry) (*httputil.ResponseJson, int) {
			updateCalled = true
			assert.Equal(t, "repo1", entry.Name)
			return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
		},
	}
	container := newHandlerContainer(newHandler(m))

	createBody := bytes.NewBufferString(`{"name":"repo1","url":"https://example.com"}`)
	createReq := httptest.NewRequest(http.MethodPost, "/helm-repos", createBody)
	createReq.Header.Set("Accept", restful.MIME_JSON)
	createReq.Header.Set("Content-Type", "application/json")
	createResp := httptest.NewRecorder()
	container.ServeHTTP(createResp, createReq)
	assert.Equal(t, http.StatusOK, createResp.Code)
	assert.True(t, createCalled)

	updateBody := bytes.NewBufferString(`{"url":"https://example.com"}`)
	updateReq := httptest.NewRequest(http.MethodPut, "/helm-repos/repo1", updateBody)
	updateReq.Header.Set("Accept", restful.MIME_JSON)
	updateReq.Header.Set("Content-Type", "application/json")
	updateResp := httptest.NewRecorder()
	container.ServeHTTP(updateResp, updateReq)
	assert.Equal(t, http.StatusOK, updateResp.Code)
	assert.True(t, updateCalled)
}

func TestCreateAndUpdateRepoBadInput(t *testing.T) {
	m := &mockOperation{}
	container := newHandlerContainer(newHandler(m))

	createReq := httptest.NewRequest(http.MethodPost, "/helm-repos", bytes.NewBufferString("{bad json"))
	createReq.Header.Set("Accept", restful.MIME_JSON)
	createReq.Header.Set("Content-Type", "application/json")
	createResp := httptest.NewRecorder()
	container.ServeHTTP(createResp, createReq)
	assert.Equal(t, http.StatusBadRequest, createResp.Code)

	updateReq := httptest.NewRequest(http.MethodPut, "/helm-repos/repo1", bytes.NewBufferString("{bad json"))
	updateReq.Header.Set("Accept", restful.MIME_JSON)
	updateReq.Header.Set("Content-Type", "application/json")
	updateResp := httptest.NewRecorder()
	container.ServeHTTP(updateResp, updateReq)
	assert.Equal(t, http.StatusBadRequest, updateResp.Code)
}

func TestUploadChart(t *testing.T) {
	uploadCalled := false
	m := &mockOperation{
		uploadChart: func(_ multipart.File, fileHeader *multipart.FileHeader) (*httputil.ResponseJson, int) {
			uploadCalled = true
			assert.Equal(t, "test.tgz", fileHeader.Filename)
			return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
		},
	}
	container := newHandlerContainer(newHandler(m))

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile(param.Chart, "test.tgz")
	assert.NoError(t, err)
	_, err = part.Write([]byte("abc"))
	assert.NoError(t, err)
	assert.NoError(t, writer.Close())

	req := httptest.NewRequest(http.MethodPost, "/helm-charts", body)
	req.Header.Set("Accept", restful.MIME_JSON)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp := httptest.NewRecorder()
	container.ServeHTTP(resp, req)
	assert.Equal(t, http.StatusOK, resp.Code)
	assert.True(t, uploadCalled)

	badReq := httptest.NewRequest(http.MethodPost, "/helm-charts", nil)
	badReq.Header.Set("Accept", restful.MIME_JSON)
	badReq.Header.Set("Content-Type", writer.FormDataContentType())
	badResp := httptest.NewRecorder()
	container.ServeHTTP(badResp, badReq)
	assert.Equal(t, http.StatusBadRequest, badResp.Code)
}

func TestSanitizeRepoEntry(t *testing.T) {
	entry := &helmModel.SafeRepoEntry{
		Name: "<name>",
		URL:  "https://example.com/<x>",
	}
	sanitizeRepoEntry(entry)
	assert.Equal(t, "<name>", entry.Name)
	assert.Equal(t, "https://example.com/<x>", entry.URL)
}

func TestMockOperationImplementsOperation(t *testing.T) {
	var _ helm.Operation = &mockOperation{}
	assert.True(t, true)
}
