package helm

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/repo"

	"marketplace-service/pkg/constant"
	models "marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
)

func makeChart(name string, created time.Time, keywords ...string) *models.ChartVersionResponse {
	return &models.ChartVersionResponse{
		Metadata: &repo.ChartVersion{
			Metadata: &chart.Metadata{
				Name:     name,
				Version:  "1.0.0",
				Keywords: keywords,
			},
			Created: created,
		},
		Repo:    "repo",
		RepoUrl: "url",
	}
}

func TestFilterChartVersionSlice(t *testing.T) {
	now := time.Now()
	charts := []*models.ChartVersionResponse{
		makeChart("alpha", now, "scene-a"),
		makeChart("beta", now.Add(time.Hour), constant.FuyaoExtensionKeyword, "scene-b"),
		makeChart("gamma", now.Add(2*time.Hour), "database"),
	}

	filtered := filterChartVersionSlice(&models.ChartSearchParam{
		Chart: "a",
		Types: []string{"application"},
		Scene: []string{"scene-a"},
	}, charts)
	assert.Len(t, filtered, 1)
	assert.Equal(t, "alpha", filtered[0].Metadata.Name)

	filtered = filterChartVersionSlice(&models.ChartSearchParam{
		Types: []string{"database"},
		Scene: []string{"scene-b", "database"},
	}, charts)
	assert.Len(t, filtered, 1)
	assert.Equal(t, "gamma", filtered[0].Metadata.Name)
}

func TestSortChartVersionSlice(t *testing.T) {
	t0 := time.Now()
	charts := []*models.ChartVersionResponse{
		makeChart("b2", t0.Add(2*time.Hour)),
		makeChart("A1", t0),
		makeChart("a0", t0.Add(time.Hour)),
	}

	sortChartVersionSlice(&param.Query{SortBy: param.Time, Ascending: true}, charts)
	assert.Equal(t, "A1", charts[0].Metadata.Name)

	sortChartVersionSlice(&param.Query{SortBy: param.Time, Ascending: false}, charts)
	assert.Equal(t, "b2", charts[0].Metadata.Name)

	sortChartVersionSlice(&param.Query{SortBy: param.Name, Ascending: true}, charts)
	assert.Equal(t, "A1", charts[0].Metadata.Name)

	sortChartVersionSlice(&param.Query{SortBy: "unknown", Ascending: false}, charts)
	assert.NotNil(t, charts[0])
}

func TestCustomCompareAndPriority(t *testing.T) {
	assert.True(t, customCompare("a", "b", ascendingOrder))
	assert.True(t, customCompare("b", "a", descendingOrder))
	assert.True(t, customCompare("A", "a", ascendingOrder))
	assert.False(t, customCompare("2", "a", ascendingOrder))

	assert.Equal(t, orderZero, getPriority('!'))
	assert.Equal(t, orderOne, getPriority('a'))
	assert.Equal(t, orderTwo, getPriority('2'))
	assert.Equal(t, orderThree, getPriority('\n'))
}

func TestPageResult(t *testing.T) {
	items := []any{"a", "b", "c"}
	resp := pageResult(items, nil)
	assert.Equal(t, 3, resp.TotalItems)
	assert.Equal(t, 3, len(resp.Items))

	resp = pageResult(items, &param.Pagination{CurrentPage: 1, Limit: 2, Offset: 0})
	assert.Equal(t, 2, len(resp.Items))
	assert.Equal(t, 2, resp.TotalPages)
}

func TestFilterSelectedRepositories(t *testing.T) {
	repos := []models.HelmChartRepository{
		{Spec: models.HelmChartRepositorySpec{DisplayName: "repo1"}},
		{Spec: models.HelmChartRepositorySpec{DisplayName: "repo2"}},
	}
	selected := filterSelectedRepositories([]string{"repo2"}, repos)
	assert.Len(t, selected, 1)
	assert.Equal(t, "repo2", selected[0].Spec.DisplayName)
}
