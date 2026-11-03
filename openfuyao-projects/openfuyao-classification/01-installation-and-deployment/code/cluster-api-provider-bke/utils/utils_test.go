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

package utils

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestExists(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "testfile")
	assert.NoError(t, err)
	tmpFile.Close()
	defer os.Remove(tmpFile.Name())

	assert.True(t, Exists(tmpFile.Name()))

	tmpDir, err := os.MkdirTemp("", "testdir")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)
	assert.True(t, Exists(tmpDir))
}

func TestContainsString(t *testing.T) {
	tests := []struct {
		name     string
		slice    []string
		s        string
		expected bool
	}{
		{"empty slice", []string{}, "test", false},
		{"string exists", []string{"a", "b", "test"}, "test", true},
		{"string not exists", []string{"a", "b", "c"}, "test", false},
		{"nil slice", nil, "test", false},
		{"single match", []string{"test"}, "test", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsString(tt.slice, tt.s)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSliceRemoveString(t *testing.T) {
	result := SliceRemoveString(nil, "a")
	assert.Empty(t, result)

	result = SliceRemoveString([]string{"a", "b", "a", "c"}, "a")
	assert.Equal(t, []string{"b", "c"}, result)

	result = SliceRemoveString([]string{"a", "b", "c"}, "d")
	assert.Equal(t, []string{"a", "b", "c"}, result)

	result = SliceRemoveString([]string{}, "a")
	assert.Empty(t, result)
}

func TestSliceContainsSlice(t *testing.T) {
	assert.False(t, SliceContainsSlice(nil, []string{"a"}))
	assert.True(t, SliceContainsSlice([]string{"a", "b", "c"}, []string{"a", "c"}))
	assert.False(t, SliceContainsSlice([]string{"a", "b", "c"}, []string{"a", "d"}))
	assert.True(t, SliceContainsSlice([]string{"a", "b", "c"}, []string{}))
	assert.False(t, SliceContainsSlice([]string{}, []string{"a"}))
}

func TestUniqueStringSlice(t *testing.T) {
	result := UniqueStringSlice(nil)
	assert.Empty(t, result)

	result = UniqueStringSlice([]string{"a", "b", "a", "c", "b"})
	assert.Len(t, result, 3)

	result = UniqueStringSlice([]string{"a", "b", "c"})
	assert.Len(t, result, 3)

	result = UniqueStringSlice([]string{})
	assert.Empty(t, result)
}

func TestIsDir(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "testdir")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)
	assert.True(t, IsDir(tmpDir))
}

func TestIsFile(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "testfile")
	assert.NoError(t, err)
	tmpFile.Close()
	defer os.Remove(tmpFile.Name())
	assert.True(t, IsFile(tmpFile.Name()))

	tmpDir, err := os.MkdirTemp("", "testdir")
	assert.NoError(t, err)
	defer os.RemoveAll(tmpDir)
	assert.False(t, IsFile(tmpDir))

}

func TestSplitNameSpaceName(t *testing.T) {
	ns, name, err := SplitNameSpaceName("default:myapp")
	assert.NoError(t, err)
	assert.Equal(t, "default", ns)
	assert.Equal(t, "myapp", name)

	_, _, err = SplitNameSpaceName("invalid")
	assert.Error(t, err)

	_, _, err = SplitNameSpaceName("")
	assert.Error(t, err)

	_, _, err = SplitNameSpaceName("ns1:ns2:name")
	assert.Error(t, err)
}

func TestClusterName(t *testing.T) {
	clusterName, err := ClusterName()
	if err == nil {
		assert.NotEmpty(t, clusterName)
	}
}

func TestHostName(t *testing.T) {
	result := HostName()
	assert.NotEmpty(t, result)
}

func TestSetNTPServerEnv(t *testing.T) {
	err := SetNTPServerEnv("1.2.3.4:123")
	assert.NoError(t, err)

	err = SetNTPServerEnv("")
	assert.NoError(t, err)
}

func TestGetNTPServerEnv(t *testing.T) {
	result, err := GetNTPServerEnv()
	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestFormatNTPServer(t *testing.T) {
	result, err := FormatNTPServer("")
	assert.NoError(t, err)
	assert.Equal(t, "", result)

	result, err = FormatNTPServer("1.2.3.4:123")
	assert.NoError(t, err)
	assert.Equal(t, "1.2.3.4:123", result)
}

func TestB64Encode(t *testing.T) {
	assert.Equal(t, "aGVsbG8=", B64Encode("hello"))
	assert.Equal(t, "", B64Encode(""))
	assert.Equal(t, "aGVsbG8gd29ybGQ=", B64Encode("hello world"))
}

func TestB64Decode(t *testing.T) {
	result, err := B64Decode("aGVsbG8=")
	assert.NoError(t, err)
	assert.Equal(t, "hello", result)

	result, err = B64Decode("")
	assert.NoError(t, err)
	assert.Equal(t, "", result)

	_, err = B64Decode("invalid!")
	assert.Error(t, err)
}

func TestSliceExcludeSlice(t *testing.T) {
	result := SliceExcludeSlice([]string{"a", "b", "c"}, []string{"b"})
	assert.Equal(t, []string{"a", "c"}, result)
}

func TestTrimSpaceSlice(t *testing.T) {
	result := TrimSpaceSlice([]string{"a", "", "b", "\n"})
	assert.Empty(t, result)
}

func TestRemoveDuplicateElement(t *testing.T) {
	result := RemoveDuplicateElement([]string{"a", "b", "a", "c"})
	assert.Equal(t, []string{"a", "b", "c"}, result)
}

func TestMostCommonChar(t *testing.T) {
	result := MostCommonChar([]string{"a", "b", "a", "a"})
	assert.Equal(t, "a", result)
	result = MostCommonChar([]string{})
	assert.Equal(t, "", result)
}

func TestCommonPrefixOfTwo(t *testing.T) {
	result := CommonPrefixOfTwo("hello", "help")
	assert.Equal(t, "hel", result)
}

func TestCommonPrefix(t *testing.T) {
	result := CommonPrefix([]string{"hello", "help", "hero"})
	result = CommonPrefix([]string{})
	assert.Equal(t, "", result)
}

func TestCtxDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	assert.False(t, CtxDone(ctx))
	cancel()
	time.Sleep(10 * time.Millisecond)
	assert.True(t, CtxDone(ctx))
}

func TestTimeNowStr(t *testing.T) {
	result := TimeNowStr()
	assert.NotEmpty(t, result)
}

func TestTimeFormat(t *testing.T) {
	tm := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	result := TimeFormat(tm)
	assert.Equal(t, "2024-01-01 12:00:00", result)
}

func TestRandom(t *testing.T) {
	result := Random(1, 10)
	assert.GreaterOrEqual(t, result, 1)
	assert.Less(t, result, 10)
}

func TestGetExtraLoadBalanceIP(t *testing.T) {
	result := GetExtraLoadBalanceIP(nil)
	assert.Equal(t, "", result)

	result = GetExtraLoadBalanceIP(map[string]string{"extraLoadBalanceIP": "192.168.1.1"})
	assert.Equal(t, "192.168.1.1", result)

	result = GetExtraLoadBalanceIP(map[string]string{"extraLoadBalanceIP": "invalid"})
	assert.Equal(t, "", result)
}

func TestClientObjNS(t *testing.T) {
	result := ClientObjNS(nil)
	assert.Equal(t, "object is nil", result)

	obj := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "test"},
	}
	result = ClientObjNS(obj)
	assert.Equal(t, "default/test", result)
}
