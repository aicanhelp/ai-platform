/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package remote

import (
	"strings"
	"testing"
)

func TestNewCombineOut(t *testing.T) {
	out := NewCombineOut("192.168.1.1", "ls -la", "output")
	if out.Host != "192.168.1.1" {
		t.Errorf("expected host 192.168.1.1, got %s", out.Host)
	}
	if out.Command != "ls -la" {
		t.Errorf("expected command ls -la, got %s", out.Command)
	}
	if out.Out != "output" {
		t.Errorf("expected out output, got %s", out.Out)
	}
}

func TestCombineOut_String(t *testing.T) {
	tests := []struct {
		name     string
		out      CombineOut
		expected string
	}{
		{
			name:     "with command",
			out:      CombineOut{Host: "192.168.1.1", Command: "ls", Out: "files"},
			expected: `host: "192.168.1.1", cmd: "ls", combined out: "files"`,
		},
		{
			name:     "without command",
			out:      CombineOut{Host: "192.168.1.1", Command: "", Out: "error msg"},
			expected: `host: "192.168.1.1", error: "error msg"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.out.String()
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestCombineOuts_String(t *testing.T) {
	outs := CombineOuts{
		{Host: "192.168.1.1", Command: "ls", Out: "file1"},
		{Host: "192.168.1.2", Command: "pwd", Out: "/home"},
	}
	result := outs.String()
	if !strings.Contains(result, "192.168.1.1") || !strings.Contains(result, "192.168.1.2") {
		t.Error("String() should contain both hosts")
	}
}

func TestNewStdCombine(t *testing.T) {
	sc := NewStdCombine()
	if sc.records == nil {
		t.Error("records should not be nil")
	}
	if sc.mux == nil {
		t.Error("mux should not be nil")
	}
}

func TestStdCombine_Add(t *testing.T) {
	sc := NewStdCombine()
	out := CombineOut{Host: "192.168.1.1", Command: "ls", Out: "files"}
	sc.Add(out)

	result := sc.Get("192.168.1.1")
	if len(result) != 1 {
		t.Errorf("expected 1 record, got %d", len(result))
	}
}

func TestStdCombine_Get(t *testing.T) {
	sc := NewStdCombine()
	out1 := CombineOut{Host: "192.168.1.1", Command: "ls", Out: "files"}
	out2 := CombineOut{Host: "192.168.1.1", Command: "pwd", Out: "/home"}
	sc.Add(out1)
	sc.Add(out2)

	result := sc.Get("192.168.1.1")
	if len(result) != 2 {
		t.Errorf("expected 2 records, got %d", len(result))
	}

	result = sc.Get("192.168.1.2")
	if len(result) != 0 {
		t.Errorf("expected 0 records, got %d", len(result))
	}
}

func TestStdCombine_Len(t *testing.T) {
	sc := NewStdCombine()
	if sc.Len() != 0 {
		t.Errorf("expected 0, got %d", sc.Len())
	}

	sc.Add(CombineOut{Host: "192.168.1.1", Command: "ls", Out: "files"})
	sc.Add(CombineOut{Host: "192.168.1.2", Command: "pwd", Out: "/home"})

	if sc.Len() != 2 {
		t.Errorf("expected 2, got %d", sc.Len())
	}
}

func TestStdCombine_Out(t *testing.T) {
	sc := NewStdCombine()
	out1 := CombineOut{Host: "192.168.1.1", Command: "ls", Out: "files"}
	out2 := CombineOut{Host: "192.168.1.2", Command: "pwd", Out: "/home"}
	sc.Add(out1)
	sc.Add(out2)

	result := sc.Out()
	if len(result) != 2 {
		t.Errorf("expected 2 hosts, got %d", len(result))
	}
	if _, ok := result["192.168.1.1"]; !ok {
		t.Error("expected host 192.168.1.1")
	}
	if _, ok := result["192.168.1.2"]; !ok {
		t.Error("expected host 192.168.1.2")
	}
}
