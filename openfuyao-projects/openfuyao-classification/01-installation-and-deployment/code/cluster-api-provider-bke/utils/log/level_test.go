/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package log

import (
	"testing"

	olog "gopkg.openfuyao.cn/common-modules/ologger/v26/log"
)

func TestNormalizeAllowedLevel(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		want  string
		wantOK bool
	}{
		{name: "empty", in: "", want: "", wantOK: false},
		{name: "blank", in: "   ", want: "", wantOK: false},
		{name: "invalid", in: "NOT_A_LEVEL", want: "", wantOK: false},
		{name: "warn alias rejected", in: "WARN", want: "", wantOK: false},
		{name: "injection rejected", in: "INFO --evil", want: "", wantOK: false},
		{name: "trace", in: "TRACE", want: olog.TRACE, wantOK: true},
		{name: "debug lower", in: "debug", want: olog.DEBUG, wantOK: true},
		{name: "info padded", in: "  INFO  ", want: olog.INFO, wantOK: true},
		{name: "warning", in: "WARNING", want: olog.WARNING, wantOK: true},
		{name: "error", in: "error", want: olog.ERROR, wantOK: true},
		{name: "critical", in: "CRITICAL", want: olog.CRITICAL, wantOK: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := NormalizeAllowedLevel(tt.in)
			if ok != tt.wantOK || got != tt.want {
				t.Fatalf("NormalizeAllowedLevel(%q)=(%q,%v), want (%q,%v)", tt.in, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}
