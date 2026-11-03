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

package param

import (
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/emicklei/go-restful/v3"
)

func TestGetNoPagination(t *testing.T) {
	tests := []struct {
		name  string
		limit int
		page  int
		want  *Pagination
	}{
		{
			name: "TestGetNoPagination",
			want: &Pagination{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetNoPagination(); reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPaginationGetValidPagination(t *testing.T) {
	type fields struct {
		Limit       int
		Offset      int
		CurrentPage int
	}
	type args struct {
		total int
	}
	tests := []struct {
		name           string
		fields         fields
		args           args
		wantStartIndex int
		wantEndIndex   int
		wantTotalPages int
	}{
		{
			name: "TestPagination_GetValidPagination",
			fields: fields{
				Limit:       10,
				Offset:      0,
				CurrentPage: 1,
			},
			args: args{
				total: 10,
			},
			wantStartIndex: 0,
			wantEndIndex:   10,
			wantTotalPages: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Pagination{
				Limit:       tt.fields.Limit,
				Offset:      tt.fields.Offset,
				CurrentPage: tt.fields.CurrentPage,
			}
			gotStartIndex, gotEndIndex, gotTotalPages := p.GetPaginationResult(tt.args.total)
			if gotStartIndex != tt.wantStartIndex {
				t.Errorf("GetValidPagination() gotStartIndex = %v, want %v", gotStartIndex, tt.wantStartIndex)
			}
			if gotEndIndex != tt.wantEndIndex {
				t.Errorf("GetValidPagination() gotEndIndex = %v, want %v", gotEndIndex, tt.wantEndIndex)
			}
			if gotTotalPages != tt.wantTotalPages {
				t.Errorf("GetValidPagination() gotTotalPages = %v, want %v", gotTotalPages, tt.wantTotalPages)
			}
		})
	}
}

func TestNewPagination(t *testing.T) {
	type args struct {
		limit int
		page  int
	}
	tests := []struct {
		name string
		args args
		want *Pagination
	}{
		{
			name: "TestNewPagination",
			args: args{
				limit: 10,
				page:  1,
			},
			want: &Pagination{
				Limit:       10,
				Offset:      0,
				CurrentPage: 1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := newPagination(tt.args.limit, tt.args.page); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newPagination() = %v, want %v", got, tt.want)
			}
		})
	}
}

func getTestRequest(method, path string, params map[string]string) *restful.Request {
	query := url.Values{}
	for key, value := range params {
		query.Set(key, value)
	}
	// 将查询参数附加到 URL
	if len(query) > 0 {
		path += "?" + query.Encode()
	}
	// 创建 http.Request
	httpReq := httptest.NewRequest(method, path, nil)
	// 转换为 restful.Request
	restfulReq := restful.NewRequest(httpReq)
	return restfulReq
}

func TestParseQueryParameter(t *testing.T) {
	type args struct {
		request *restful.Request
	}
	tests := []struct {
		name string
		args args
		want *Query
	}{
		{
			name: "TestParseQueryParameter_1",
			args: args{
				request: getTestRequest("GET", "/test", map[string]string{
					"limit":    "10",
					"page":     "2",
					"sortType": "name",
					"order":    "asc",
				}),
			},
			want: &Query{
				Pagination: &Pagination{
					Limit:       10,
					Offset:      10,
					CurrentPage: 2,
				},
				SortBy:    "name",
				Ascending: true,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseQueryParameter(tt.args.request); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseQueryParameter() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetPaginationResult(t *testing.T) {
	tests := []struct {
		name               string
		pagination         Pagination
		total              int
		expectedStartIndex int
		expectedEndIndex   int
		expectedTotalPages int
	}{
		{
			name: "Valid pagination within range",
			pagination: Pagination{
				Limit:  10,
				Offset: 20,
			},
			total:              100,
			expectedStartIndex: 20,
			expectedEndIndex:   30,
			expectedTotalPages: 10,
		},
		{
			name: "Offset exceeds total",
			pagination: Pagination{
				Limit:  10,
				Offset: 150,
			},
			total:              100,
			expectedStartIndex: 0,
			expectedEndIndex:   10,
			expectedTotalPages: 10,
		},
		{
			name: "No pagination limit",
			pagination: Pagination{
				Limit:  noPaginationLimit,
				Offset: 0,
			},
			total:              100,
			expectedStartIndex: 0,
			expectedEndIndex:   100,
			expectedTotalPages: 1,
		},
		{
			name: "Negative limit and offset",
			pagination: Pagination{
				Limit:  -5,
				Offset: -10,
			},
			total:              100,
			expectedStartIndex: 0,
			expectedEndIndex:   100,
			expectedTotalPages: 1,
		},
		{
			name: "Limit is zero",
			pagination: Pagination{
				Limit:  0,
				Offset: 10,
			},
			total:              100,
			expectedStartIndex: 0,
			expectedEndIndex:   100,
			expectedTotalPages: 1,
		},
		{
			name: "Offset is negative",
			pagination: Pagination{
				Limit:  10,
				Offset: -5,
			},
			total:              100,
			expectedStartIndex: 0,
			expectedEndIndex:   100,
			expectedTotalPages: 1,
		},
		{
			name: "Limit greater than total",
			pagination: Pagination{
				Limit:  150,
				Offset: 0,
			},
			total:              100,
			expectedStartIndex: 0,
			expectedEndIndex:   100,
			expectedTotalPages: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startIndex, endIndex, totalPages := tt.pagination.GetPaginationResult(tt.total)
			if startIndex != tt.expectedStartIndex {
				t.Errorf("startIndex = %d; want %d", startIndex, tt.expectedStartIndex)
			}
			if endIndex != tt.expectedEndIndex {
				t.Errorf("endIndex = %d; want %d", endIndex, tt.expectedEndIndex)
			}
			if totalPages != tt.expectedTotalPages {
				t.Errorf("totalPages = %d; want %d", totalPages, tt.expectedTotalPages)
			}
		})
	}
}
