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

package filters

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apiserver/pkg/authentication/user"

	"application-management-service/pkg/constant"
)

func generateToken(sub string) string {
	headerObj := map[string]interface{}{"alg": "none", "typ": "JWT"}
	headerBytes, err := json.Marshal(headerObj)
	if err != nil {
		return ""
	}
	header := base64.RawURLEncoding.EncodeToString(headerBytes)

	payloadObj := map[string]interface{}{"sub": sub}
	payloadBytes, err := json.Marshal(payloadObj)
	if err != nil {
		return ""
	}
	payload := base64.RawURLEncoding.EncodeToString(payloadBytes)

	return header + "." + payload + "."
}

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
			token:   generateToken("testuser"),
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

func TestAuthenticateOpenFuyaoUser(t *testing.T) {
	chain := &restful.FilterChain{
		Target: func(request *restful.Request, response *restful.Response) {
			response.WriteHeader(http.StatusOK)
		},
	}
	tests := []struct {
		name       string
		authHeader string
		want       string
		wantErr    bool
	}{
		{
			name:       "TesFailToFetchAuthHeader",
			authHeader: "bad header",
		},
		{
			name:       "TestAuthHeaderFetched",
			authHeader: "Bearer 12345678",
		},
		{
			name:       "no_token",
			authHeader: "Bearer " + generateToken("testuser"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			httpReq := httptest.NewRequest(http.MethodGet, "/test", nil)
			httpReq.Header.Set(constant.OpenFuyaoAuthHeaderKey, tt.authHeader)
			req := restful.NewRequest(httpReq)
			resp := restful.NewResponse(httptest.NewRecorder())
			AuthenticateOpenFuyaoUser(req, resp, chain)
		})
	}
}
