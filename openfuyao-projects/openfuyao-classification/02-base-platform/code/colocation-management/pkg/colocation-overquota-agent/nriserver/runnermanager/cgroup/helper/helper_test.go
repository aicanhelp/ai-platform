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
package helper

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewAndEquals(t *testing.T) {
	assert.True(t, New(1, 2, 3).Equals(New(3, 2, 1)))
	assert.False(t, New(1, 2).Equals(New(1, 2, 3)))
	assert.True(t, New().Equals(New()))
}

func TestParse(t *testing.T) {
	s, err := Parse("0-3")
	assert.NoError(t, err)
	assert.True(t, s.Equals(New(0, 1, 2, 3)))

	s, err = Parse("5")
	assert.NoError(t, err)
	assert.True(t, s.Equals(New(5)))

	s, err = Parse("")
	assert.NoError(t, err)
	assert.True(t, s.Equals(New()))

	s, err = Parse("1-1") // start == end acceptable
	assert.NoError(t, err)
	assert.True(t, s.Equals(New(1)))

	s, err = Parse("0-5,34,46-48")
	assert.NoError(t, err)
	assert.True(t, s.Equals(New(0, 1, 2, 3, 4, 5, 34, 46, 47, 48)))

	_, err = Parse("5-0") // start > end
	assert.Error(t, err)

	_, err = Parse("abc") // invalid int
	assert.Error(t, err)
}

func TestIsCpuSetEqual(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"0-3", "0,1,2,3", true},
		{"0-3", "0-5", false},
		{"", "", true},
		{"0-3", "5-0", false}, // second parse fails
		{"x", "0", false},     // first parse fails
	}
	for _, c := range cases {
		assert.Equal(t, c.want, IsCpuSetEqual(c.a, c.b), "IsCpuSetEqual(%q,%q)", c.a, c.b)
	}
}

func TestCgroupFileRead(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f")
	assert.NoError(t, os.WriteFile(p, []byte("hello"), 0o644))

	b, err := CgroupFileRead(p)
	assert.NoError(t, err)
	assert.Equal(t, "hello", string(b))

	_, err = CgroupFileRead(filepath.Join(dir, "missing"))
	assert.Error(t, err)
}

func TestCgroupFIleWrite(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "w")
	assert.NoError(t, os.WriteFile(p, []byte{}, 0o644)) // pre-create so PathExist is true

	assert.NoError(t, CgroupFIleWrite(p, "data"))
	b, err := os.ReadFile(p)
	assert.NoError(t, err)
	assert.Equal(t, "data", string(b))

	assert.Error(t, CgroupFIleWrite(filepath.Join(dir, "missing"), "x"))
}
