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

// Package config loads cache-indexer runtime configuration from a config file.
package config

import (
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Runtime config file default values.
const (
	DefaultConfigPath = "/etc/cache-indexer/config.yaml"
)

// HTTP server default values.
const (
	DefaultHTTPAddr                = ":28080"
	DefaultHTTPShutdownTimeout     = 10 * time.Second
	DefaultHTTPReadHeaderTimeout   = 10 * time.Second
	DefaultHTTPReadTimeout         = 30 * time.Second
	DefaultHTTPWriteTimeout        = 30 * time.Second
	DefaultHTTPIdleTimeout         = 120 * time.Second
	DefaultHTTPMaxHeaderBytes      = 1 << 20
	DefaultHTTPMaxHitRateBodyBytes = int64(4 << 20)
)

// Logging default values.
const (
	DefaultLogLevel = "info"
)

// vLLM block-key compatibility default values.
const (
	DefaultPythonHashSeed        = "0"
	DefaultPrefixCachingHashAlgo = "sha256_cbor"
	DefaultUseIntBlockHashes     = true
)

// Kubernetes discovery default values.
const (
	DefaultDiscoveryRefreshInterval      = 10 * time.Second
	DefaultDiscoverySegmentsFetchTimeout = 2 * time.Second
	DefaultDiscoveryEngineLabelKey       = "openfuyao.com/engine"
	DefaultDiscoveryEngineValue          = "vllm"
	DefaultDiscoveryPDRoleLabelKey       = "openfuyao.com/pdRole"
	DefaultDiscoveryPDRoleValues         = "prefill,decode,aggregate"
	DefaultDiscoveryKVManagerLabelKey    = "openfuyao.com/kvmanager"
	DefaultDiscoveryKVManagerValue       = "mooncake"
	DefaultVLLMZMQPortName               = "zmq-pub"
	DefaultMooncakeHTTPPortName          = "http-api"
	DefaultMooncakeRPCPortName           = "rpc"
	DefaultMooncakeMasterHTTPPort        = 9003
)

// Ingest-plane default values.
const (
	DefaultIngestL1BackoffInitial = time.Second
	DefaultIngestL1BackoffMax     = 30 * time.Second
	DefaultIngestL3Scheme         = "http"
	DefaultIngestL3PollInterval   = 10 * time.Second
	DefaultIngestL3HTTPTimeout    = 5 * time.Second
)

// Settings is the root runtime configuration.
type Settings struct {
	HTTP      HTTPConfig      `yaml:"http"`
	Log       LogConfig       `yaml:"log"`
	BlockKey  BlockKeyConfig  `yaml:"blockKey"`
	Discovery DiscoveryConfig `yaml:"discovery"`
	Ingest    IngestConfig    `yaml:"ingest"`
}

// HTTPConfig governs the public HTTP server lifecycle.
type HTTPConfig struct {
	Addr                string        `yaml:"addr"`
	ShutdownTimeout     time.Duration `yaml:"shutdownTimeout"`
	ReadHeaderTimeout   time.Duration `yaml:"readHeaderTimeout"`
	ReadTimeout         time.Duration `yaml:"readTimeout"`
	WriteTimeout        time.Duration `yaml:"writeTimeout"`
	IdleTimeout         time.Duration `yaml:"idleTimeout"`
	MaxHeaderBytes      int           `yaml:"maxHeaderBytes"`
	MaxHitRateBodyBytes int64         `yaml:"maxHitRateBodyBytes"`
}

// LogConfig controls process log level.
type LogConfig struct {
	Level string `yaml:"level"` // debug | info | error
}

// BlockKeyConfig holds settings that must match vLLM prefix-cache hashing.
type BlockKeyConfig struct {
	PythonHashSeed        string `yaml:"pythonHashSeed"`
	PrefixCachingHashAlgo string `yaml:"prefixCachingHashAlgo"`
	UseIntBlockHashes     bool   `yaml:"useIntBlockHashes"`
}

// DiscoveryConfig holds runtime settings for pod and Mooncake
// discovery.
type DiscoveryConfig struct {
	Namespace            string                        `yaml:"-"`
	RefreshInterval      time.Duration                 `yaml:"refreshInterval"`
	SegmentsFetchTimeout time.Duration                 `yaml:"segmentsFetchTimeout"`
	Selector             DiscoverySelectorConfig       `yaml:"labels"`
	VLLM                 VLLMDiscoveryConfig           `yaml:"vllm"`
	MooncakeMaster       MooncakeMasterDiscoveryConfig `yaml:"mooncakeMaster"`
}

// DiscoverySelectorConfig governs how pods are identified during discovery.
type DiscoverySelectorConfig struct {
	EngineLabelKey    string   `yaml:"engineKey"`
	EngineValue       string   `yaml:"engineValue"`
	PDRoleLabelKey    string   `yaml:"pdRoleKey"`
	PDRoleValues      []string `yaml:"pdRoleValue"`
	KVManagerLabelKey string   `yaml:"kvManagerKey"`
	KVManagerValue    string   `yaml:"kvManagerValue"`
}

// VLLMDiscoveryConfig governs vLLM Pod discovery.
type VLLMDiscoveryConfig struct {
	ZMQPortName string `yaml:"zmqPortName"`
}

// MooncakeMasterDiscoveryConfig governs Mooncake Master Pod discovery.
type MooncakeMasterDiscoveryConfig struct {
	HTTPPortName string `yaml:"httpPortName"`
	RPCPortName  string `yaml:"rpcPortName"`
	HTTPPort     int32  `yaml:"httpPort"`
	RPCPort      int32  `yaml:"rpcPort"`
}

// IngestConfig holds L1 and L3 ingest runtime settings.
type IngestConfig struct {
	L1 IngestL1Config `yaml:"l1"`
	L3 IngestL3Config `yaml:"l3"`
}

// IngestL1Config governs vLLM ZMQ subscription reconnect behaviour.
type IngestL1Config struct {
	BackoffInitial time.Duration `yaml:"backoffInitial"`
	BackoffMax     time.Duration `yaml:"backoffMax"`
}

// IngestL3Config governs Mooncake Master HTTP polling behaviour.
type IngestL3Config struct {
	Scheme       string        `yaml:"scheme"`
	PollInterval time.Duration `yaml:"pollInterval"`
	HTTPTimeout  time.Duration `yaml:"httpTimeout"`
}

// Load reads configuration from the default config file; never fails.
func Load() *Settings {
	return load(DefaultConfigPath)
}

func load(path string) *Settings {
	cfg := defaultSettings()
	loadConfigFile(path, cfg)
	applyRuntimeEnv(cfg)
	return cfg
}

func defaultSettings() *Settings {
	return &Settings{
		HTTP:      defaultHTTPConfig(),
		Log:       defaultLogConfig(),
		BlockKey:  defaultBlockKeyConfig(),
		Discovery: defaultDiscoveryConfig(),
		Ingest:    defaultIngestConfig(),
	}
}

func defaultHTTPConfig() HTTPConfig {
	return HTTPConfig{
		Addr:                DefaultHTTPAddr,
		ShutdownTimeout:     DefaultHTTPShutdownTimeout,
		ReadHeaderTimeout:   DefaultHTTPReadHeaderTimeout,
		ReadTimeout:         DefaultHTTPReadTimeout,
		WriteTimeout:        DefaultHTTPWriteTimeout,
		IdleTimeout:         DefaultHTTPIdleTimeout,
		MaxHeaderBytes:      DefaultHTTPMaxHeaderBytes,
		MaxHitRateBodyBytes: DefaultHTTPMaxHitRateBodyBytes,
	}
}

func defaultLogConfig() LogConfig {
	return LogConfig{Level: DefaultLogLevel}
}

func defaultBlockKeyConfig() BlockKeyConfig {
	return BlockKeyConfig{
		PythonHashSeed:        DefaultPythonHashSeed,
		PrefixCachingHashAlgo: DefaultPrefixCachingHashAlgo,
		UseIntBlockHashes:     DefaultUseIntBlockHashes,
	}
}

func defaultDiscoveryConfig() DiscoveryConfig {
	return DiscoveryConfig{
		RefreshInterval:      DefaultDiscoveryRefreshInterval,
		SegmentsFetchTimeout: DefaultDiscoverySegmentsFetchTimeout,
		Selector:             defaultDiscoverySelectorConfig(),
		VLLM:                 defaultVLLMDiscoveryConfig(),
		MooncakeMaster:       defaultMooncakeMasterDiscoveryConfig(),
	}
}

func defaultDiscoverySelectorConfig() DiscoverySelectorConfig {
	return DiscoverySelectorConfig{
		EngineLabelKey:    DefaultDiscoveryEngineLabelKey,
		EngineValue:       DefaultDiscoveryEngineValue,
		PDRoleLabelKey:    DefaultDiscoveryPDRoleLabelKey,
		PDRoleValues:      splitStringList(DefaultDiscoveryPDRoleValues),
		KVManagerLabelKey: DefaultDiscoveryKVManagerLabelKey,
		KVManagerValue:    DefaultDiscoveryKVManagerValue,
	}
}

func defaultVLLMDiscoveryConfig() VLLMDiscoveryConfig {
	return VLLMDiscoveryConfig{ZMQPortName: DefaultVLLMZMQPortName}
}

func defaultMooncakeMasterDiscoveryConfig() MooncakeMasterDiscoveryConfig {
	return MooncakeMasterDiscoveryConfig{
		HTTPPortName: DefaultMooncakeHTTPPortName,
		RPCPortName:  DefaultMooncakeRPCPortName,
		HTTPPort:     DefaultMooncakeMasterHTTPPort,
	}
}

func defaultIngestConfig() IngestConfig {
	return IngestConfig{
		L1: defaultIngestL1Config(),
		L3: defaultIngestL3Config(),
	}
}

func defaultIngestL1Config() IngestL1Config {
	return IngestL1Config{
		BackoffInitial: DefaultIngestL1BackoffInitial,
		BackoffMax:     DefaultIngestL1BackoffMax,
	}
}

func defaultIngestL3Config() IngestL3Config {
	return IngestL3Config{
		Scheme:       DefaultIngestL3Scheme,
		PollInterval: DefaultIngestL3PollInterval,
		HTTPTimeout:  DefaultIngestL3HTTPTimeout,
	}
}

func loadConfigFile(path string, cfg *Settings) {
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil || strings.TrimSpace(string(data)) == "" {
		return
	}
	next := *cfg
	if err := yaml.Unmarshal(data, &next); err == nil {
		*cfg = next
	}
}

func applyRuntimeEnv(cfg *Settings) {
	if v, ok := os.LookupEnv("POD_NAMESPACE"); ok && v != "" {
		cfg.Discovery.Namespace = v
	}
}

func splitStringList(v string) []string {
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}
