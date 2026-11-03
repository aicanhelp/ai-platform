package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestLoadReadsConfigFile(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	data := []byte(`
http:
  addr: ":29090"
  shutdownTimeout: 11s
  readHeaderTimeout: 12s
  readTimeout: 13s
  writeTimeout: 14s
  idleTimeout: 15s
  maxHeaderBytes: 2048
  maxHitRateBodyBytes: 4096
log:
  level: error
blockKey:
  pythonHashSeed: "1"
  prefixCachingHashAlgo: xxhash_cbor
  useIntBlockHashes: false
discovery:
  refreshInterval: 21s
  segmentsFetchTimeout: 22s
  labels:
    engineKey: engine.example/key
    engineValue: custom-engine
    pdRoleKey: role.example/key
    pdRoleValue:
      - decode
    kvManagerKey: kv.example/key
    kvManagerValue: custom-kv
  vllm:
    zmqPortName: custom-zmq
  mooncakeMaster:
    httpPortName: custom-http
    rpcPortName: custom-rpc
    httpPort: 9103
    rpcPort: 9104
ingest:
  l1:
    backoffInitial: 31s
    backoffMax: 32s
  l3:
    scheme: https
    pollInterval: 33s
    httpTimeout: 34s
`)
	if err := os.WriteFile(configPath, data, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("POD_NAMESPACE", "cache-indexer")

	cfg := load(configPath)

	if cfg.HTTP.Addr != ":29090" {
		t.Fatalf("HTTP.Addr = %q, want :29090", cfg.HTTP.Addr)
	}
	if cfg.HTTP.ShutdownTimeout != 11*time.Second {
		t.Fatalf("HTTP.ShutdownTimeout = %s, want 11s", cfg.HTTP.ShutdownTimeout)
	}
	if cfg.Log.Level != "error" {
		t.Fatalf("Log.Level = %q, want error", cfg.Log.Level)
	}
	if cfg.BlockKey.PrefixCachingHashAlgo != "xxhash_cbor" {
		t.Fatalf("PrefixCachingHashAlgo = %q, want xxhash_cbor", cfg.BlockKey.PrefixCachingHashAlgo)
	}
	if cfg.BlockKey.UseIntBlockHashes {
		t.Fatal("UseIntBlockHashes = true, want false")
	}
	if cfg.Discovery.Namespace != "cache-indexer" {
		t.Fatalf("Namespace = %q, want downward API namespace", cfg.Discovery.Namespace)
	}
	if !reflect.DeepEqual(cfg.Discovery.Selector.PDRoleValues, []string{"decode"}) {
		t.Fatalf("PDRoleValues = %#v, want config file value", cfg.Discovery.Selector.PDRoleValues)
	}
	if cfg.Discovery.Selector.KVManagerValue != "custom-kv" {
		t.Fatalf("KVManagerValue = %q, want custom-kv", cfg.Discovery.Selector.KVManagerValue)
	}
	if cfg.Discovery.MooncakeMaster.RPCPort != 9104 {
		t.Fatalf("MooncakeMaster.RPCPort = %d, want 9104", cfg.Discovery.MooncakeMaster.RPCPort)
	}
	if cfg.Ingest.L3.Scheme != "https" {
		t.Fatalf("Ingest.L3.Scheme = %q, want https", cfg.Ingest.L3.Scheme)
	}
}

func TestLoadFallsBackToDefaultsWhenConfigFileIsMissing(t *testing.T) {
	cfg := load(filepath.Join(t.TempDir(), "missing.yaml"))

	if cfg.HTTP.Addr != DefaultHTTPAddr {
		t.Fatalf("HTTP.Addr = %q, want %q", cfg.HTTP.Addr, DefaultHTTPAddr)
	}
	if cfg.Log.Level != DefaultLogLevel {
		t.Fatalf("Log.Level = %q, want %q", cfg.Log.Level, DefaultLogLevel)
	}
	if !reflect.DeepEqual(cfg.Discovery.Selector.PDRoleValues, []string{"prefill", "decode", "aggregate"}) {
		t.Fatalf("PDRoleValues = %#v, want defaults", cfg.Discovery.Selector.PDRoleValues)
	}
}
