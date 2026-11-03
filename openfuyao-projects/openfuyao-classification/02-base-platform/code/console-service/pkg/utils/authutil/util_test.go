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

package authutil

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestExtractUserFromJWT(t *testing.T) {
	tests := []struct {
		name    string
		token   string
		want    user.Info
		wantErr bool
	}{
		{
			name:    "invalid_token",
			token:   "xxxxxx",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "valid_token_with_subject",
			token:   GenerateToken("testuser"),
			want:    &user.DefaultInfo{Name: "testuser"},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractUserFromJWT(tt.token)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractUserFromJWT() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ExtractUserFromJWT() got = %v, want %v", got, tt.want)
			}
		})
	}
}
