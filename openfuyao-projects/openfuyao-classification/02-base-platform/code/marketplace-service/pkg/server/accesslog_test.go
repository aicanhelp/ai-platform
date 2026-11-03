/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"marketplace-service/pkg/zlog"
)

// TestRecordAccessLogs_BadRequest
func TestRecordAccessLogs_BadRequest(t *testing.T) {
	req := restful.NewRequest(httptest.NewRequest("POST", "/api/v1/invalid", nil))
	resp := restful.NewResponse(httptest.NewRecorder())
	chain := &restful.FilterChain{}

	patches := gomonkey.ApplyMethod(chain, "ProcessFilter",
		func(_ *restful.FilterChain, r *restful.Request, w *restful.Response) {
			w.WriteHeader(http.StatusBadRequest) // 400（临界值，走Info）
		})
	defer patches.Reset()

	infofCalled := false
	patches.ApplyFunc(zlog.Infof, func(_ string, _ ...interface{}) {
		infofCalled = true
	})

	RecordAccessLogs(req, resp, chain)
	assert.True(t, infofCalled) // 400<=BadRequest，走Info
}
