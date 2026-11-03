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

import "testing"

func TestScaleToGiB(t *testing.T) {
	tests := getScaleToGiBTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertScaleToGiB(t, tt)
		})
	}
}

func getScaleToGiBTestCases() []struct {
	name string
	args struct {
		str string
	}
	want string
} {
	return []struct {
		name string
		args struct {
			str string
		}
		want string
	}{
		{
			name: "ki",
			args: struct{ str string }{
				str: "1048576Ki",
			},
			want: "1.0Gi",
		},
		{
			name: "Mi",
			args: struct{ str string }{
				str: "1024Mi",
			},
			want: "1.0Gi",
		},
		{
			name: "gi",
			args: struct{ str string }{
				str: "1Gi",
			},
			want: "1Gi",
		},
		{
			name: "none",
			args: struct{ str string }{
				str: "10",
			},
			want: "",
		},
	}
}

func assertScaleToGiB(t *testing.T, tt struct {
	name string
	args struct {
		str string
	}
	want string
}) {
	if got := ScaleToGiB(tt.args.str); got != tt.want {
		t.Errorf("ScaleToGiB() = %v, want %v", got, tt.want)
	}
}
