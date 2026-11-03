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
package utils

import (
	"encoding/json"
	"reflect"
	"testing"
)

type TestConfig struct {
	Name string `json:"name"`
	Age  string `json:"age"`
}

func TestConfToUnstructure(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    map[string]interface{}
		wantErr bool
	}{
		{
			name: "test",
			input: &TestConfig{
				Name: "me",
				Age:  "five",
			},
			want: map[string]interface{}{
				"name": "me",
				"age":  "five",
			},
			wantErr: false,
		},
		{
			name:    "invalid object conversion",
			input:   "in",
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ConfToUnstructure(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConfToUnstructure() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				var gotMap map[string]interface{}
				if err := json.Unmarshal(got, &gotMap); err != nil {
					t.Fatalf("Failed to unmarshal JSON: %v", err)
				}

				if !reflect.DeepEqual(gotMap, tt.want) {
					t.Errorf("ConfToUnstructure() got = %v, want %v", gotMap, tt.want)
				}
			}
		})
	}
}
