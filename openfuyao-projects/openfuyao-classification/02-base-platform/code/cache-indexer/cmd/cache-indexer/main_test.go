package main

import (
	"context"
	"testing"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/blockkey"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	"gitcode.com/openFuyao/cache-indexer/pkg/discovery"
	ingestl1 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l1"
	ingestl3 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l3"
)

const (
	testHashAlgoSHA = "sha256_cbor"
	testHashAlgoXXH = "xxhash_cbor"
	testHashAlgoBad = "bad-algo"
	testSeed        = "12345"
)

type stopRecorder struct {
	stopped bool
}

func (s *stopRecorder) Stop() { s.stopped = true }

type l1StopRecorder struct {
	stopRecorder
}

func (l *l1StopRecorder) Subscribe(ingestl1.VLLMSubscribeTarget) error { return nil }
func (l *l1StopRecorder) Unsubscribe(string) error                      { return nil }

type l3StopRecorder struct {
	stopRecorder
}

func (l *l3StopRecorder) UpdateTarget(*ingestl3.MooncakePollTarget) error { return nil }

type managerStopRecorder struct {
	stopped bool
}

func (m *managerStopRecorder) Start(_ context.Context) error { return nil }
func (m *managerStopRecorder) Stop() error {
	m.stopped = true
	return nil
}
func (m *managerStopRecorder) Snapshot() *discovery.Snapshot { return nil }

func TestParseHashAlgo(t *testing.T) {
	cases := []struct {
		input string
		want  blockkey.HashAlgo
		err   bool
	}{
		{input: "", want: blockkey.HashSHA256CBOR},
		{input: testHashAlgoSHA, want: blockkey.HashSHA256CBOR},
		{input: testHashAlgoXXH, want: blockkey.HashXXHash3CBOR},
		{input: testHashAlgoBad, want: blockkey.HashSHA256CBOR, err: true},
	}
	for _, tc := range cases {
		got, err := parseHashAlgo(tc.input)
		if tc.err {
			if err == nil {
				t.Fatalf("parseHashAlgo(%q) want error", tc.input)
			}
			continue
		}
		if err != nil {
			t.Fatalf("parseHashAlgo(%q): %v", tc.input, err)
		}
		if got != tc.want {
			t.Fatalf("parseHashAlgo(%q)=%v want=%v", tc.input, got, tc.want)
		}
	}
}

func TestNewBlockKeyConfig(t *testing.T) {
	cfg, err := newBlockKeyConfig(config.BlockKeyConfig{
		PythonHashSeed:        testSeed,
		PrefixCachingHashAlgo: testHashAlgoXXH,
		UseIntBlockHashes:     true,
	})
	if err != nil {
		t.Fatalf("newBlockKeyConfig: %v", err)
	}
	if cfg.PythonHashSeed != testSeed || cfg.Algo != blockkey.HashXXHash3CBOR || !cfg.EmitInt64 {
		t.Fatalf("cfg=%+v", cfg)
	}
}

func TestNewBuilder_InvalidConfigReturnsNil(t *testing.T) {
	builder := newBuilder(logr.Discard(), config.BlockKeyConfig{
		PythonHashSeed:        testSeed,
		PrefixCachingHashAlgo: testHashAlgoBad,
	})
	if builder != nil {
		t.Fatal("builder should be nil for invalid algo")
	}
}

func TestNewBuilder_EmptySeedReturnsNil(t *testing.T) {
	builder := newBuilder(logr.Discard(), config.BlockKeyConfig{})
	if builder != nil {
		t.Fatal("builder should be nil for empty seed")
	}
}

func TestDiscoveryPlaneStop_NilSafe(t *testing.T) {
	var nilPlane *discoveryPlane
	nilPlane.stop()

	l1 := &l1StopRecorder{}
	l3 := &l3StopRecorder{}
	manager := &managerStopRecorder{}
	plane := &discoveryPlane{manager: manager, l1Sub: l1, l3Sub: l3}
	plane.stop()
	if !manager.stopped || !l1.stopped || !l3.stopped {
		t.Fatalf("manager=%v l1=%v l3=%v", manager.stopped, l1.stopped, l3.stopped)
	}
}

func TestStartDiscoveryPlane_RequiresNamespace(t *testing.T) {
	_, err := startDiscoveryPlane(context.Background(), config.DiscoveryConfig{}, config.BlockKeyConfig{}, config.IngestConfig{}, appDeps{})
	if err == nil {
		t.Fatal("expected namespace error")
	}
}

func TestNewAppDeps(t *testing.T) {
	deps := newAppDeps(logr.Discard())
	if deps.l1Indexer == nil || deps.l3Indexer == nil || deps.registry == nil {
		t.Fatalf("deps=%+v", deps)
	}
}
