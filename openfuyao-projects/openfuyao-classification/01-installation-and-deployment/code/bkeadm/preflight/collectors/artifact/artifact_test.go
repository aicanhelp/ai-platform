/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package artifact

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	require.NoError(t, os.WriteFile(p, []byte(content), 0o644))
	return p
}

func TestCollectExists(t *testing.T) {
	p := writeTemp(t, "pkg.tar.gz", "data")
	c := New()
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"artifact." + p + ".exists"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectExistsMissing(t *testing.T) {
	c := New()
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"artifact./no/such/file.exists"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, false, facts[0].Value)
}

func TestCollectReadable(t *testing.T) {
	p := writeTemp(t, "pkg.tar.gz", "data")
	c := New()
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"artifact." + p + ".readable"}})
	assert.True(t, facts[0].Available)
	assert.Equal(t, true, facts[0].Value)
}

func TestCollectFormat(t *testing.T) {
	c := New()
	tests := map[string]string{
		"artifact./path/pkg.tar.gz.format":  "tar.gz",
		"artifact./path/pkg.tgz.format":     "tar.gz",
		"artifact./path/pkg.tar.format":     "tar",
		"artifact./path/pkg.rpm.format":     "rpm",
		"artifact./path/pkg.deb.format":     "deb",
		"artifact./path/values.yaml.format": "yaml",
		"artifact./path/values.yml.format":  "yaml",
		"artifact./path/unknown.format":     "unknown",
	}
	for key, want := range tests {
		facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{key}})
		assert.Equal(t, want, facts[0].Value, key)
	}
}

func TestCollectUnknownKind(t *testing.T) {
	c := New()
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"artifact./path.bogus"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrUnsupported, facts[0].Error.Code)
}

func TestCollectBadKey(t *testing.T) {
	c := New()
	facts, _ := c.Collect(context.Background(), api.FactRequest{Keys: []string{"artifact.nodotinpath"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrParseError, facts[0].Error.Code)
}

func TestCollectContextCanceled(t *testing.T) {
	c := New()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	facts, _ := c.Collect(ctx, api.FactRequest{Keys: []string{"artifact./tmp/x.exists"}})
	assert.False(t, facts[0].Available)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)
}
