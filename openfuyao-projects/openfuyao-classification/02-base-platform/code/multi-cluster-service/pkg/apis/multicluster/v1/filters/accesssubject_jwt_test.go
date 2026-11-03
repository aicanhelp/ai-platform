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
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
)

// craftJWT builds a structurally valid, unsigned JWT for ParseUnverified.
func craftJWT(t *testing.T, payload string) string {
	t.Helper()
	enc := base64.RawURLEncoding
	header := enc.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	if !strings.Contains(payload, `"sub"`) {
		t.Fatalf("payload must carry subject: %s", payload)
	}
	return header + "." + enc.EncodeToString([]byte(payload)) + ".c2ln"
}

type chainRecorder struct {
	processed bool
}

func (c *chainRecorder) asChain() *restful.FilterChain {
	rec := c
	return &restful.FilterChain{
		Filters: []restful.FilterFunction{
			func(_ *restful.Request, _ *restful.Response, _ *restful.FilterChain) {
				rec.processed = true
			},
		},
		Target: func(_ *restful.Request, _ *restful.Response) {},
	}
}

func TestExactSubjectAccessJWTExtra(t *testing.T) {
	build := func(authHeader, token string) (*restful.Request, *httptest.ResponseRecorder, *chainRecorder) {
		httpReq := httptest.NewRequest("GET", "/", nil)
		if authHeader != "" {
			httpReq.Header.Set(authHeader, token)
		}
		req := restful.NewRequest(httpReq)
		rec := httptest.NewRecorder()
		return req, rec, &chainRecorder{}
	}

	t.Run("bearer token sets user context", func(t *testing.T) {
		req, rec, chain := build("Authorization", "Bearer "+craftJWT(t, `{"sub":"alice"}`))
		ExactSubjectAccess(req, restful.NewResponse(rec), chain.asChain())
		if !chain.processed {
			t.Error("filter chain not processed for valid token")
		}
		if user, ok := req.Request.Context().Value("user").(string); !ok || user != "alice" {
			t.Errorf("user context = %q (%v), want alice", user, ok)
		}
	})

	t.Run("openfuyao header fallback", func(t *testing.T) {
		req, rec, chain := build("X-OpenFuyao-Authorization", craftJWT(t, `{"sub":"bob"}`))
		ExactSubjectAccess(req, restful.NewResponse(rec), chain.asChain())
		if !chain.processed {
			t.Error("filter chain not processed for openfuyao header token")
		}
		if user, ok := req.Request.Context().Value("user").(string); !ok || user != "bob" {
			t.Errorf("user context = %q (%v), want bob", user, ok)
		}
	})

	t.Run("invalid token stops filter chain", func(t *testing.T) {
		req, rec, chain := build("Authorization", "Bearer not-a-jwt")
		ExactSubjectAccess(req, restful.NewResponse(rec), chain.asChain())
		if chain.processed {
			t.Error("filter chain processed despite invalid token")
		}
	})

	t.Run("no auth header passes through", func(t *testing.T) {
		req, rec, chain := build("", "")
		ExactSubjectAccess(req, restful.NewResponse(rec), chain.asChain())
		if !chain.processed {
			t.Error("filter chain not processed without auth header")
		}
		if _, ok := req.Request.Context().Value("user").(string); ok {
			t.Error("user context unexpectedly set without token")
		}
	})
}
