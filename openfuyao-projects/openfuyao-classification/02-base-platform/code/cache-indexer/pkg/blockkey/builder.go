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

// Package blockkey reproduces vLLM `hash_block_tokens` so query BlockHash
// values match vLLM ZMQ events and Mooncake keys.
package blockkey

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/fxamacker/cbor/v2"
	"github.com/go-logr/logr"
	"github.com/zeebo/xxh3"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

// Builder is the query-side BlockHash producer.
type Builder interface {
	// BuildFromRequest derives the vLLM-compatible block-hash chain for a request.
	BuildFromRequest(
		modelCtx apis.ModelContext,
		tokenIDs []int32,
	) ([]apis.BlockHash, error)
}

// HashAlgo selects which CBOR-based hash vLLM is using.
type HashAlgo int

// Algorithm selector constants. These are package-level enums describing
// which vLLM-compatible hash family the builder should reproduce.
const (
	// HashSHA256CBOR selects vLLM's sha256_cbor block hash mode.
	HashSHA256CBOR HashAlgo = iota
	// HashXXHash3CBOR selects vLLM's xxhash_cbor block hash mode.
	HashXXHash3CBOR
)

// Protocol constants fixed by CBOR layout and vLLM hash widths.
// They are implementation invariants, not runtime configuration.
const (
	// CBOR definite-length array header for a 3-element tuple.
	cborArrayLenThreeHeader = 0x83
	// Low-64-bit truncation width used by vLLM int block hash mode.
	uint64ByteWidth = 8
	// xxh3-128 always emits 16 raw hash bytes.
	xxh3HashByteWidth = 16
)

// Config controls builder behaviour.
type Config struct {
	// PythonHashSeed mirrors vLLM PYTHONHASHSEED.
	// Empty seeds would diverge from vLLM's stable NONE_HASH.
	PythonHashSeed string
	// Algo selects sha256_cbor (default) or xxhash_cbor.
	Algo HashAlgo
	// EmitInt64 mirrors vLLM int block hash mode.
	// True means low 64 bits, big-endian, 16-char hex.
	EmitInt64 bool
}

type builder struct {
	log     logr.Logger
	cfg     Config
	encMode cbor.EncMode

	noneOnce sync.Once
	noneHash []byte
	noneErr  error
}

// New builds a Builder from an explicit Config.
func New(log logr.Logger, cfg Config) (Builder, error) {
	if cfg.PythonHashSeed == "" {
		return nil, fmt.Errorf(
			"blockkey: PYTHONHASHSEED is empty; required for stable NONE_HASH derivation",
		)
	}
	// CTAP2 canonical encoding mirrors Python `cbor2.dumps(..., canonical=True)`.
	// This keeps cross-language bytes reproducible.
	em, err := cbor.CTAP2EncOptions().EncMode()
	if err != nil {
		return nil, fmt.Errorf("blockkey: cbor encoder: %w", err)
	}
	return &builder{log: log, cfg: cfg, encMode: em}, nil
}

func (b *builder) hash(payload []byte) []byte {
	switch b.cfg.Algo {
	case HashXXHash3CBOR:
		u := xxh3.Hash128(payload)
		out := make([]byte, xxh3HashByteWidth)
		binary.BigEndian.PutUint64(out[:uint64ByteWidth], u.Hi)
		binary.BigEndian.PutUint64(out[uint64ByteWidth:], u.Lo)
		return out
	default:
		s := sha256.Sum256(payload)
		return s[:]
	}
}

// noneHashBytes lazily derives the CBOR-family NONE_HASH from the canonical
// encoding of the seed string, matching vLLM init_none_hash.
func (b *builder) noneHashBytes() ([]byte, error) {
	b.noneOnce.Do(func() {
		// vLLM feeds the raw env-var string into canonical CBOR encoding
		// before hashing it with sha256 or xxh3.
		seedCBOR, err := b.encMode.Marshal(b.cfg.PythonHashSeed)
		if err != nil {
			b.noneErr = fmt.Errorf("encode PYTHONHASHSEED: %w", err)
			return
		}
		b.noneHash = b.hash(seedCBOR)
	})
	return b.noneHash, b.noneErr
}

// BuildFromRequest mirrors vLLM `get_request_block_hasher` over a token
// stream of length floor(len(tokenIDs)/block_size) full blocks.
func (b *builder) BuildFromRequest(
	modelCtx apis.ModelContext,
	tokenIDs []int32,
) ([]apis.BlockHash, error) {
	bs := int(modelCtx.BlockSize)
	if bs <= 0 {
		return nil, fmt.Errorf("blockkey: block_size must be > 0 (got %d)", bs)
	}
	full := len(tokenIDs) / bs
	if full == 0 {
		return nil, nil
	}
	none, err := b.noneHashBytes()
	if err != nil {
		return nil, err
	}
	// Only the first block gets cache_salt extra_keys.
	// Later blocks use CBOR null until lora / multimodal extras are supported.
	extraFirst, err := b.encodeExtraKeys(modelCtx)
	if err != nil {
		return nil, err
	}
	extraRest, err := b.encMode.Marshal(nil)
	if err != nil {
		return nil, err
	}
	out := make([]apis.BlockHash, full)
	parent := none
	for i := 0; i < full; i++ {
		toks := tokenIDs[i*bs : (i+1)*bs]
		extra := extraRest
		if i == 0 {
			extra = extraFirst
		}
		raw, err := b.hashOneBlock(parent, toks, extra)
		if err != nil {
			return nil, err
		}
		out[i] = b.encodeBlockHash(raw)
		parent = raw
	}
	return out, nil
}

func (b *builder) encodeExtraKeys(mc apis.ModelContext) ([]byte, error) {
	if mc.CacheSalt == "" {
		// vLLM returns Python None when extra_keys is empty; CBOR null = 0xf6.
		return b.encMode.Marshal(nil)
	}
	// Python tuple(cache_salt) encodes like a one-item list in canonical CBOR.
	return b.encMode.Marshal([]any{mc.CacheSalt})
}

// hashOneBlock encodes Python's `(parent_hash, tuple(token_ids), extra_keys)`
// as a definite-length 3-element CBOR array and hashes it.
func (b *builder) hashOneBlock(parent []byte, tokens []int32, extraCBOR []byte) ([]byte, error) {
	tokensAny := make([]any, len(tokens))
	for i, t := range tokens {
		tokensAny[i] = int64(t)
	}
	tokensCBOR, err := b.encMode.Marshal(tokensAny)
	if err != nil {
		return nil, fmt.Errorf("encode tokens: %w", err)
	}
	parentCBOR, err := b.encMode.Marshal(parent) // CBOR byte string
	if err != nil {
		return nil, fmt.Errorf("encode parent: %w", err)
	}
	// The outer object is header 0x83 plus three self-delimited CBOR items.
	buf := make([]byte, 0, 1+len(parentCBOR)+len(tokensCBOR)+len(extraCBOR))
	buf = append(buf, cborArrayLenThreeHeader)
	buf = append(buf, parentCBOR...)
	buf = append(buf, tokensCBOR...)
	buf = append(buf, extraCBOR...)
	return b.hash(buf), nil
}

// encodeBlockHash converts the raw hash bytes into the BlockHash string
// format produced by ingest/l1 (so that L1 lookups match).
func (b *builder) encodeBlockHash(raw []byte) apis.BlockHash {
	if !b.cfg.EmitInt64 {
		return apis.BlockHash(hex.EncodeToString(raw))
	}
	// vLLM int mode keeps the low 64 bits of the big-endian hash integer.
	// For an N-byte hash, this is the last 8 bytes.
	if len(raw) < uint64ByteWidth {
		var buf [uint64ByteWidth]byte
		copy(buf[uint64ByteWidth-len(raw):], raw)
		return apis.BlockHash(hex.EncodeToString(buf[:]))
	}
	low := raw[len(raw)-uint64ByteWidth:]
	return apis.BlockHash(hex.EncodeToString(low))
}
