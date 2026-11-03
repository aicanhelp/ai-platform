/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*
Package param
contains constant for application-management-service url parsing
*/
package param

// parameters for URI and documentation
const (
	Repository = "repo"
	Tag        = "tag"
	SortType   = "sortType"
	Chart      = "chart"
	Release    = "release"
	Namespace  = "namespace"
	Version    = "version"
	FileType   = "fileType"
	AppType    = "appType"
	Page       = "page"
	Time       = "time"
	Limit      = "limit"
	Extension  = "extension"
	SortBy     = "sortBy"
	Ascending  = "ascending"
	Scene      = "scene"
	Status     = "status"

	FuyaoPlugin = "fuyaoPlugin"
	FuyaoTurbo  = "fuyaoTurbo"
)

// ReleaseParameter are parameters that the helm needs to list releases
type ReleaseParameter struct {
	Namespace string
	Release   string
	Extension string
	SortBy    string
	Ascending string
	Status    []string
}

const (
	// PluginName is plugin name
	PluginName = "pluginName"
	// PluginEnabled decides whether the plugin will be displayed
	PluginEnabled = "enabled"
)

// Query represents api search terms
type Query struct {
	Pagination *Pagination

	SortBy string

	Ascending bool
}

// Pagination for pagination usage
type Pagination struct {
	// limit per page
	Limit int
	// page offset
	Offset int
	// current page
	CurrentPage int
}

const (
	noPaginationLimit = -1
	noPaginationPage  = 1
)

// GetNoPagination return default pagination for non-pagination needed
func GetNoPagination() *Pagination {
	return newPagination(noPaginationLimit, noPaginationPage)
}

// make sure that pagination is valid
func newPagination(limit int, page int) *Pagination {
	return &Pagination{
		Limit:       limit,
		Offset:      (page - 1) * limit,
		CurrentPage: page,
	}
}

// GetPaginationResult return start & end indexes and total pages for querying
func (p *Pagination) GetPaginationResult(total int) (startIndex, endIndex, totalPages int) {
	// input validation
	if p.Limit <= 0 || p.Offset < 0 {
		return 0, total, 1
	}

	// out of range, restart from page one
	if p.Offset > total {
		p.Offset = 0
	}

	// pagination start
	startIndex = p.Offset
	endIndex = min(startIndex+p.Limit, total)

	// total Pages
	totalPages = (total-1)/p.Limit + 1
	return startIndex, endIndex, totalPages
}
