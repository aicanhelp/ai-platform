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
contains constant for marketplace-service url parsing
*/
package param

import (
	"strconv"

	"github.com/emicklei/go-restful/v3"
)

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
	Name       = "name"
	Limit      = "limit"
	Extension  = "extension"
	SortBy     = "sortBy"
	Order      = "order"
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
	SortBy     string
	Ascending  bool
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
	descending        = "desc"
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

	// no pagination
	if p.Limit == noPaginationLimit {
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

// ParseQueryParameter parse query parameter from request parameter
func ParseQueryParameter(request *restful.Request) *Query {
	query := &Query{}

	limit, err := strconv.Atoi(request.QueryParameter(Limit))
	if err != nil {
		limit = -1
	}
	page, err := strconv.Atoi(request.QueryParameter(Page))
	if err != nil {
		page = 1
	}
	query.Pagination = newPagination(limit, page)

	query.SortBy = request.QueryParameter(SortType)
	order := request.QueryParameter(Order)
	if order == descending {
		query.Ascending = false
	} else {
		query.Ascending = true
	}

	return query
}
