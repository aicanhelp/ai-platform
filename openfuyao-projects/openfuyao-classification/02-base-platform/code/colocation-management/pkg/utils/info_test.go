/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-08-18
 */

package utils

import (
	"fmt"
	"testing"
)

func TestAggregateErrors(t *testing.T) {
	tests := []struct {
		name   string
		errors []error
		want   string
	}{
		{
			name:   "nil errors",
			errors: nil,
			want:   "",
		},
		{
			name:   "empty errors",
			errors: []error{},
			want:   "",
		},
		{
			name:   "errors with nil",
			errors: []error{fmt.Errorf("error1"), nil},
			want:   "[error1]",
		},
		{
			name:   "multiple errors",
			errors: []error{fmt.Errorf("error1"), fmt.Errorf("error2")},
			want:   "[error1 error2]",
		},
		{
			name:   "duplicate errors",
			errors: []error{fmt.Errorf("error1"), fmt.Errorf("error1")},
			want:   "[error1]",
		},
		{
			name:   "only nil entries filtered to empty",
			errors: []error{nil, nil, nil},
			want:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AggregateErrors(tt.errors)

			if tt.want == "" {
				if got != nil {
					t.Errorf("AggregateErrors() = %v, want nil", got)
				}
				return
			}

			if got == nil {
				t.Errorf("AggregateErrors() = nil, want %s", tt.want)
				return
			}

			if got.Error() != tt.want {
				t.Errorf("AggregateErrors() = %s, want %s", got.Error(), tt.want)
			}
		})
	}
}
