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
	"net/http"
	"net/http/httptest"
	"testing"

	"console-service/pkg/i18n"
)

func TestWithLocaleFilter(t *testing.T) {
	var capturedLocale string
	handler := WithLocale(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedLocale = i18n.GetLocaleFromContext(r.Context())
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Accept-Language", "en-US")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)

	if capturedLocale != "en-US" {
		t.Errorf("WithLocale() context locale = %q, want en-US", capturedLocale)
	}
	if recorder.Code != http.StatusOK {
		t.Errorf("WithLocale() status = %d, want %d", recorder.Code, http.StatusOK)
	}
}
