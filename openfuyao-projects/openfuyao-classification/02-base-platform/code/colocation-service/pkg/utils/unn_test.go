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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestValidate 测试 UNN 的验证功能
func TestValidate(t *testing.T) {
	testCases := []struct {
		desc     string // 测试用例描述
		input    UNN    // 输入值
		expected bool   // 期望结果
	}{
		{
			desc:     "标准格式",
			input:    "uid-123/default/pod-1",
			expected: true,
		},
		{
			desc:     "缺少namespace",
			input:    "uid-123//pod-1",
			expected: false,
		},
		{
			desc:     "缺少name",
			input:    "uid-123/default/",
			expected: false,
		},
		{
			desc:     "缺少uid",
			input:    "/default/pod-1",
			expected: false,
		},
		{
			desc:     "多余部分",
			input:    "uid-123/default/pod-1/extra",
			expected: false,
		},
		{
			desc:     "空字符串",
			input:    "",
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if result := tc.input.Validate(); result != tc.expected {
				t.Errorf("Validate() = %v, want %v", result, tc.expected)
			}
		})
	}
}

// TestComponentMethods 测试各组件获取方法
func TestComponentMethods(t *testing.T) {
	testCases := []struct {
		desc    string
		input   UNN
		expUID  string
		expNS   string
		expName string
		expAll  []string
	}{
		{
			desc:    "完整UNN",
			input:   "uid-123/default/pod-1",
			expUID:  "uid-123",
			expNS:   "default",
			expName: "pod-1",
			expAll:  []string{"uid-123", "default", "pod-1"},
		},
		{
			desc:    "空组件",
			input:   "uid-123//pod-1",
			expUID:  "uid-123",
			expNS:   "",
			expName: "pod-1",
			expAll:  []string{"uid-123", "", "pod-1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if uid := tc.input.Uid(); uid != tc.expUID {
				t.Errorf("Uid() = %v, want %v", uid, tc.expUID)
			}
			if ns := tc.input.Namespace(); ns != tc.expNS {
				t.Errorf("Namespace() = %v, want %v", ns, tc.expNS)
			}
			if name := tc.input.Name(); name != tc.expName {
				t.Errorf("Name() = %v, want %v", name, tc.expName)
			}

			all := tc.input.All()
			if len(all) != len(tc.expAll) {
				t.Fatalf("All() length mismatch: got %d, want %d", len(all), len(tc.expAll))
			}
			for i := range all {
				if all[i] != tc.expAll[i] {
					t.Errorf("All()[%d] = %v, want %v", i, all[i], tc.expAll[i])
				}
			}
		})
	}
}

// TestFormatAndString 测试格式化和字符串方法
func TestFormatAndString(t *testing.T) {
	if got := UNN("").Format(); got != "<uid>/<namespace>/<name>" {
		t.Errorf("Format() = %v, want %v", got, "<uid>/<namespace>/<name>")
	}

	input := "uid-123/default/pod-1"
	if got := UNN(input).String(); got != input {
		t.Errorf("String() = %v, want %v", got, input)
	}
}

// TestConstructors 测试构造函数
func TestConstructors(t *testing.T) {
	// 测试NewUNN
	if got := NewUNN("uid-123", "default", "pod-1"); got != "uid-123/default/pod-1" {
		t.Errorf("NewUNN() = %v, want %v", got, "uid-123/default/pod-1")
	}

	// 测试NewUNNFromObject
	obj := &metav1.ObjectMeta{
		UID:       "uid-123",
		Namespace: "default",
		Name:      "pod-1",
	}
	if got := NewUNNFromObject(obj); got != "uid-123/default/pod-1" {
		t.Errorf("NewUNNFromObject() = %v, want %v", got, "uid-123/default/pod-1")
	}
}

// TestEncodeAndNewUNNFromId 测试 Encode 和 NewUNNFromId
func TestEncodeAndNewUNNFromId(t *testing.T) {
	original := NewUNN("uid-123", "default", "pod-1")
	encoded := original.Encode()
	decoded := NewUNNFromId(encoded)
	if decoded != original {
		t.Errorf("NewUNNFromId(Encode()) = %v, want %v", decoded, original)
	}
}

// TestNewUNNFromId_Invalid 测试无效的base64输入
func TestNewUNNFromId_Invalid(t *testing.T) {
	decoded := NewUNNFromId("!!!invalid base64!!!")
	_ = decoded
}
