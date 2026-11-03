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

package blockkey

import (
	"reflect"
	"testing"

	"github.com/go-logr/logr/testr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

// Golden vectors are produced by /tmp/golden.py with PYTHONHASHSEED="12345".

// Test constants (grouped by category)
const (
	testSeed = "12345"

	// Block hash test constants
	testBlockSize      = int64(4)
	testSmallBlockSize = int64(2)

	// Token test constants
	testTokenCount      = 16
	testShortTokenCount = 3
)

// [Test Category 1: Constructor Tests]

// TestNew_RequiresSeed verifies New returns error when PYTHONHASHSEED is empty.
func TestNew_RequiresSeed(t *testing.T) {
	if _, err := New(testr.New(t), Config{}); err == nil {
		t.Fatal("expected error for empty PYTHONHASHSEED")
	}
}

// TestNew_WithValidConfig creates builder successfully.
func TestNew_WithValidConfig(t *testing.T) {
	b, err := New(testr.New(t), Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if b == nil {
		t.Fatal("expected non-nil builder")
	}
}

// TestNew_WithXXHash3Algo creates builder with xxhash_cbor algorithm.
func TestNew_WithXXHash3Algo(t *testing.T) {
	b, err := New(testr.New(t), Config{PythonHashSeed: testSeed, Algo: HashXXHash3CBOR, EmitInt64: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	tokens := []int32{10, 20, 30, 40}
	got, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, tokens)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 block, got %d", len(got))
	}
}

// TestNew_DisableInt64Hash emits full hash when EmitInt64=false.
func TestNew_DisableInt64Hash(t *testing.T) {
	b, err := New(testr.New(t), Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: false})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	got, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, []int32{10, 20, 30, 40})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if len(got[0]) != 64 {
		t.Fatalf("expected full hex (64 chars), got %d", len(got[0]))
	}
}

// [Test Category 2: Core Method Tests]

// TestBuildFromRequest_BlockSizeValidation verifies error when block_size <= 0.
func TestBuildFromRequest_BlockSizeValidation(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, EmitInt64: true})
	if _, err := b.BuildFromRequest(apis.ModelContext{BlockSize: 0}, []int32{1, 2}); err == nil {
		t.Fatal("expected error for block_size=0")
	}
}

// TestBuildFromRequest_ShorterThanOneBlock_ReturnsNil returns nil when tokens < block_size.
func TestBuildFromRequest_ShorterThanOneBlock_ReturnsNil(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, EmitInt64: true})
	out, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, []int32{1, 2, testShortTokenCount})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if out != nil {
		t.Fatalf("expected nil, got %v", out)
	}
}

// TestSHA256CBOR_GoldenChain_NoSalt verifies SHA256 chain matches vLLM golden vectors.
func TestSHA256CBOR_GoldenChain_NoSalt(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: true})
	tokens := []int32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}
	got, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, tokens)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []apis.BlockHash{
		"32bafc903fcc3d41",
		"85561c9d5b82a733",
		"ad51afb517a4a0d5",
		"a4bacdb3459e9fcd",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sha256 chain mismatch\n got=%v\nwant=%v", got, want)
	}
}

// TestSHA256CBOR_GoldenChain_WithCacheSalt verifies cache_salt only affects block 0.
func TestSHA256CBOR_GoldenChain_WithCacheSalt(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: true})
	got, err := b.BuildFromRequest(
		apis.ModelContext{BlockSize: testSmallBlockSize, CacheSalt: "abc"},
		[]int32{1, 2, 3, 4, 5, 6},
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	// vLLM applies cache_salt only to block 0 (start_token_idx == 0).
	// Block 1+ have empty extra_keys → CBOR null in the 3-tuple.
	want := []apis.BlockHash{
		"2aa1822e9d32e3f9",
		"bf212bcd6f43c610",
		"f13afb3eddfc87cf",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("salt chain mismatch\n got=%v\nwant=%v", got, want)
	}
}

// TestXXHash3CBOR_GoldenChain verifies XXHash3 chain matches vLLM golden vectors.
func TestXXHash3CBOR_GoldenChain(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashXXHash3CBOR, EmitInt64: true})
	tokens := []int32{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}
	got, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, tokens)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []apis.BlockHash{
		"962478845458fd72",
		"b52a3f6d44df5af0",
		"ad1aaaf27d07b4ce",
		"9b69e0e949610060",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("xxh3 chain mismatch\n got=%v\nwant=%v", got, want)
	}
}

// TestEmitFullHexMode verifies EmitInt64=false outputs full hash hex.
func TestEmitFullHexMode(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: false})
	got, err := b.BuildFromRequest(
		apis.ModelContext{BlockSize: testBlockSize},
		[]int32{10, 20, 30, 40},
	)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1 block, got %d", len(got))
	}
	// First block in the no-salt chain has full sha256 hex (64 chars).
	const wantFull = "9bf368022e49912ba16c481c749a6a55b3117e0ceacb09ae32bafc903fcc3d41"
	if string(got[0]) != wantFull {
		t.Fatalf("full hex mismatch\n got=%s\nwant=%s", got[0], wantFull)
	}
}

// TestEncodeBlockHash_ShortHash verifies handling of hash bytes < 8.
func TestEncodeBlockHash_ShortHash(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: true})
	// Create a scenario where hash would be short (though unlikely in practice)
	// We'll test the encodeBlockHash logic directly by accessing the builder
	builder := b.(*builder)

	// Test with 4-byte hash
	shortHash := []byte{0x01, 0x02, 0x03, 0x04}
	result := builder.encodeBlockHash(shortHash)
	// Should be padded to 8 bytes (leading zeros) and hex-encoded to 16 chars
	if len(result) != 16 {
		t.Errorf("expected 16-char hex, got %d chars: %s", len(result), result)
	}
	// The 4 bytes should appear at the end with leading zeros
	expected := "0000000001020304"
	if string(result) != expected {
		t.Errorf("expected %s, got %s", expected, result)
	}

	// Test with 1-byte hash
	oneByteHash := []byte{0xff}
	result = builder.encodeBlockHash(oneByteHash)
	if len(result) != 16 {
		t.Errorf("expected 16-char hex for 1-byte, got %d", len(result))
	}
	expectedOne := "00000000000000ff"
	if string(result) != expectedOne {
		t.Errorf("expected %s, got %s", expectedOne, result)
	}
}

// TestBuildFromRequest_EmptyTokenList returns nil for empty token list.
func TestBuildFromRequest_EmptyTokenList(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, EmitInt64: true})
	got, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, []int32{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Errorf("expected nil for empty tokens, got %v", got)
	}
}

// TestBuildFromRequest_MultipleBlocks verifies chaining across multiple blocks.
func TestBuildFromRequest_MultipleBlocks(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: true})
	// 32 tokens with block_size=4 → 8 blocks
	tokens := make([]int32, 32)
	for i := 0; i < 32; i++ {
		tokens[i] = int32(i + 1)
	}

	got, err := b.BuildFromRequest(apis.ModelContext{BlockSize: testBlockSize}, tokens)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if len(got) != 8 {
		t.Fatalf("expected 8 blocks, got %d", len(got))
	}

	// Verify each block hash is 16 chars (EmitInt64=true)
	for i, hash := range got {
		if len(hash) != 16 {
			t.Errorf("block %d: expected 16-char hash, got %d", i, len(hash))
		}
	}
}

// TestBuildFromRequest_NegativeBlockSize returns error for negative block_size.
func TestBuildFromRequest_NegativeBlockSize(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, EmitInt64: true})
	_, err := b.BuildFromRequest(apis.ModelContext{BlockSize: -1}, []int32{1, 2, 3, 4})
	if err == nil {
		t.Fatal("expected error for negative block_size")
	}
}

// TestNoneHashBytes_CalledOnce verifies noneHashBytes is computed only once per builder.
func TestNoneHashBytes_CalledOnce(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashSHA256CBOR, EmitInt64: true})
	builder := b.(*builder)

	// Call multiple times - should return same result
	hash1, err1 := builder.noneHashBytes()
	if err1 != nil {
		t.Fatalf("first call failed: %v", err1)
	}

	hash2, err2 := builder.noneHashBytes()
	if err2 != nil {
		t.Fatalf("second call failed: %v", err2)
	}

	if hash1 == nil || hash2 == nil {
		t.Fatal("hash should not be nil")
	}

	// Should be identical (computed once)
	if !reflect.DeepEqual(hash1, hash2) {
		t.Errorf("hash should be identical across calls: %v vs %v", hash1, hash2)
	}

	// Verify hash length (sha256 = 32 bytes)
	if len(hash1) != 32 {
		t.Errorf("sha256 hash should be 32 bytes, got %d", len(hash1))
	}
}

// TestNoneHashBytes_XXHash3 verifies XXHash3 produces 16-byte hash.
func TestNoneHashBytes_XXHash3(t *testing.T) {
	b := mustBuilder(t, Config{PythonHashSeed: testSeed, Algo: HashXXHash3CBOR, EmitInt64: true})
	builder := b.(*builder)

	hash, err := builder.noneHashBytes()
	if err != nil {
		t.Fatalf("noneHashBytes: %v", err)
	}

	// XXHash3 produces 16 bytes
	if len(hash) != 16 {
		t.Errorf("xxh3 hash should be 16 bytes, got %d", len(hash))
	}
}

// mustBuilder builds a test BlockKeyBuilder and fails fast on error.
func mustBuilder(t *testing.T, cfg Config) Builder {
	t.Helper()
	b, err := New(testr.New(t), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return b
}
