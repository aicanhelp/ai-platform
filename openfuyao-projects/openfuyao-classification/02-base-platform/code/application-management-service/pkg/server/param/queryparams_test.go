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
	"reflect"
	"testing"
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

func TestPaginationCases(t *testing.T) {
	tests := []struct {
		name       string
		pagination *Pagination
		total      int
		wantStart  int
		wantEnd    int
		wantPages  int
	}{
		{
			name: "InvalidInput",
			pagination: &Pagination{
				Limit:       -1,
				Offset:      -1,
				CurrentPage: 1,
			},
			total:     10,
			wantStart: 0,
			wantEnd:   10,
			wantPages: 1,
		},
		{
			name: "OutOfRange",
			pagination: &Pagination{
				Limit:       10,
				Offset:      20,
				CurrentPage: 1,
			},
			total:     10,
			wantStart: 0,
			wantEnd:   10,
			wantPages: 1,
		},
		{
			name: "ValidPagination",
			pagination: &Pagination{
				Limit:       10,
				Offset:      0,
				CurrentPage: 1,
			},
			total:     10,
			wantStart: 0,
			wantEnd:   10,
			wantPages: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			startIndex, endIndex, totalPages := tt.pagination.GetPaginationResult(tt.total)
			if startIndex != tt.wantStart {
				t.Errorf("Got startIndex = %v, want %v", startIndex, tt.wantStart)
			}
			if endIndex != tt.wantEnd {
				t.Errorf("Got endIndex = %v, want %v", endIndex, tt.wantEnd)
			}
			if totalPages != tt.wantPages {
				t.Errorf("Got totalPages = %v, want %v", totalPages, tt.wantPages)
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
