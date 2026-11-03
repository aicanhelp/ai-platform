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

package generators

import (
	"crypto/rand"
	"errors"
	"regexp"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"gopkg.in/oauth2.v3"
)

// failingReader is a reader that always fails
type failingReader struct{}

func (r *failingReader) Read(p []byte) (n int, err error) {
	return 0, errors.New("read error")
}

func TestNewFuyaoAuthorizeGenerate(t *testing.T) {
	gen := NewFuyaoAuthorizeGenerate()
	assert.NotNil(t, gen)
}

func TestFuyaoAuthorizeGenerate_Token(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() (*oauth2.GenerateBasic, *FuyaoAuthorizeGenerate, *gomonkey.Patches)
		wantErr bool
		validate func(t *testing.T, token string)
	}{
		{
			name: "normal token generation",
			setup: func() (*oauth2.GenerateBasic, *FuyaoAuthorizeGenerate, *gomonkey.Patches) {
				gen := NewFuyaoAuthorizeGenerate()
				basic := &oauth2.GenerateBasic{
					UserID: "test-user",
				}
				patches := gomonkey.NewPatches()
				return basic, gen, patches
			},
			wantErr: false,
			validate: func(t *testing.T, token string) {
				// Verify length: 16 bytes = 32 hex characters (constants.AuthCodeByteLength = 16)
				assert.Equal(t, 32, len(token))
				// Verify hex format: ^[a-f0-9]{32}$
				matched, err := regexp.MatchString(`^[a-f0-9]{32}$`, token)
				assert.NoError(t, err)
				assert.True(t, matched, "token should be lowercase hex string")
			},
		},
		{
			name: "random number read failure",
			setup: func() (*oauth2.GenerateBasic, *FuyaoAuthorizeGenerate, *gomonkey.Patches) {
				gen := NewFuyaoAuthorizeGenerate()
				basic := &oauth2.GenerateBasic{
					UserID: "test-user",
				}
				patches := gomonkey.NewPatches()
				// Patch rand.Reader to return a failing reader
				failingReader := &failingReader{}
				patches.ApplyGlobalVar(&rand.Reader, failingReader)
				return basic, gen, patches
			},
			wantErr: true,
			validate: func(t *testing.T, token string) {
				assert.Empty(t, token)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			basic, gen, patches := tt.setup()
			defer patches.Reset()

			token, err := gen.Token(basic)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.validate != nil {
					tt.validate(t, token)
				}
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, token)
				if tt.validate != nil {
					tt.validate(t, token)
				}
			}
		})
	}
}

func TestFuyaoAuthorizeGenerate_Token_Uniqueness(t *testing.T) {
	gen := NewFuyaoAuthorizeGenerate()
	basic := &oauth2.GenerateBasic{
		UserID: "test-user",
	}

	// Generate 1000 tokens
	tokenCount := 1000
	tokens := make(map[string]bool, tokenCount)

	for i := 0; i < tokenCount; i++ {
		token, err := gen.Token(basic)
		assert.NoError(t, err)
		assert.NotEmpty(t, token)

		// Check for duplicates
		if tokens[token] {
			t.Errorf("Duplicate token found at iteration %d: %s", i, token)
		}
		tokens[token] = true
	}

	// Verify all tokens are unique
	assert.Equal(t, tokenCount, len(tokens), "All tokens should be unique")
}

func TestFuyaoAuthorizeGenerate_Token_Randomness(t *testing.T) {
	gen := NewFuyaoAuthorizeGenerate()
	basic := &oauth2.GenerateBasic{
		UserID: "test-user",
	}

	// Generate multiple tokens and verify they are different
	token1, err := gen.Token(basic)
	assert.NoError(t, err)
	assert.NotEmpty(t, token1)

	token2, err := gen.Token(basic)
	assert.NoError(t, err)
	assert.NotEmpty(t, token2)

	token3, err := gen.Token(basic)
	assert.NoError(t, err)
	assert.NotEmpty(t, token3)

	// Verify all tokens are different
	assert.NotEqual(t, token1, token2, "Token1 and Token2 should be different")
	assert.NotEqual(t, token2, token3, "Token2 and Token3 should be different")
	assert.NotEqual(t, token1, token3, "Token1 and Token3 should be different")

	// Verify all tokens have correct format
	hexPattern := regexp.MustCompile(`^[a-f0-9]{32}$`)
	assert.True(t, hexPattern.MatchString(token1), "Token1 should match hex pattern")
	assert.True(t, hexPattern.MatchString(token2), "Token2 should match hex pattern")
	assert.True(t, hexPattern.MatchString(token3), "Token3 should match hex pattern")
}
