/*
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package security

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"sync"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// SecretKeyProviderConfig wires the provider to a Kubernetes API client via
// injected closures, so the security package stays independent of any specific
// client implementation (controller-runtime or client-go).
type SecretKeyProviderConfig struct {
	// Namespace, SecretName and DataKey default to DefaultSecretNamespace,
	// DefaultSecretName and SecretDataKey when empty.
	Namespace  string
	SecretName string
	DataKey    string

	// GetSecret reads the Secret. It MUST return (nil, nil) when the Secret
	// is not found (as opposed to returning an error).
	GetSecret func(ctx context.Context) (*corev1.Secret, error)

	// CreateSecret creates the Secret. When the Secret already exists
	// (concurrent creation by another replica) it MUST return (false, nil)
	// so the provider re-reads the winning Secret.
	CreateSecret func(ctx context.Context, s *corev1.Secret) (bool, error)

	// Guard (optional) is invoked before generating a new key. Returning an
	// error defers this generation attempt (the error is not cached; the next
	// call retries). It is used to detect key-loss scenarios and emit
	// warnings (Events/logs); it does not block generation.
	Guard func(ctx context.Context) error
}

// SecretKeyProvider lazily loads the AES-256 key from a Kubernetes Secret:
//   - when the Secret exists, the key is validated and cached in-process, so
//     subsequent calls (webhook admission hot path) hit memory only;
//   - when the Secret does not exist, a new random 32-byte key is generated
//     (after the Guard check) and the Secret is created;
//   - load failures are not cached: the next call retries.
//
// Key rotation requires a process restart, matching the previous env-based
// behavior.
type SecretKeyProvider struct {
	cfg SecretKeyProviderConfig
	mu  sync.Mutex
	key []byte
}

// NewSecretKeyProvider validates the config and fills defaults.
func NewSecretKeyProvider(cfg SecretKeyProviderConfig) (*SecretKeyProvider, error) {
	if cfg.Namespace == "" {
		cfg.Namespace = DefaultSecretNamespace
	}
	if cfg.SecretName == "" {
		cfg.SecretName = DefaultSecretName
	}
	if cfg.DataKey == "" {
		cfg.DataKey = SecretDataKey
	}
	if cfg.GetSecret == nil {
		return nil, errors.New("SecretKeyProviderConfig.GetSecret must not be nil")
	}
	if cfg.CreateSecret == nil {
		return nil, errors.New("SecretKeyProviderConfig.CreateSecret must not be nil")
	}
	return &SecretKeyProvider{cfg: cfg}, nil
}

// Key returns the cached key or loads it from the Secret.
func (p *SecretKeyProvider) Key(ctx context.Context) ([]byte, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.key != nil {
		return p.key, nil
	}
	return p.loadKeyLocked(ctx)
}

// loadKeyLocked loads the key from the Secret, or generates one when the
// Secret is missing. Callers must hold p.mu.
func (p *SecretKeyProvider) loadKeyLocked(ctx context.Context) ([]byte, error) {
	sec, err := p.cfg.GetSecret(ctx)
	if err != nil {
		return nil, fmt.Errorf("get secret %s/%s: %w", p.cfg.Namespace, p.cfg.SecretName, err)
	}
	if sec != nil {
		return p.useSecretLocked(sec)
	}
	return p.generateKeyLocked(ctx)
}

// useSecretLocked validates and caches the key read from an existing Secret.
// Callers must hold p.mu.
func (p *SecretKeyProvider) useSecretLocked(sec *corev1.Secret) ([]byte, error) {
	k, err := decodeKey(sec, p.cfg.DataKey)
	if err != nil {
		return nil, err
	}
	p.key = k
	log.Infof("AES encryption key loaded from secret %s/%s", p.cfg.Namespace, p.cfg.SecretName)
	return k, nil
}

// generateKeyLocked runs the guard, generates a random 32-byte key, creates
// the Secret and resolves the concurrent-creation race by re-reading the
// winning replica's Secret. Callers must hold p.mu.
func (p *SecretKeyProvider) generateKeyLocked(ctx context.Context) ([]byte, error) {
	// Guard: detect key loss and emit warnings (does not block generation).
	if p.cfg.Guard != nil {
		if err := p.cfg.Guard(ctx); err != nil {
			return nil, err
		}
	}

	key, err := generateRandomKey()
	if err != nil {
		return nil, err
	}
	created, err := p.cfg.CreateSecret(ctx, keySecret(p.cfg, key))
	if err != nil {
		return nil, fmt.Errorf("create secret %s/%s: %w", p.cfg.Namespace, p.cfg.SecretName, err)
	}
	if !created {
		// Concurrent creation by another replica: re-read and use theirs.
		log.Info("encryption key secret was created concurrently by another replica, re-reading")
		sec, err := p.cfg.GetSecret(ctx)
		if err != nil {
			return nil, fmt.Errorf("re-get secret %s/%s: %w", p.cfg.Namespace, p.cfg.SecretName, err)
		}
		if sec == nil {
			return nil, errors.New("encryption key secret disappeared after concurrent create")
		}
		return p.useSecretLocked(sec)
	}
	p.key = key
	log.Infof("AES encryption key generated and stored in secret %s/%s", p.cfg.Namespace, p.cfg.SecretName)
	return key, nil
}

// UpdateFromSecret refreshes the cached key from an externally observed
// Secret (watch Added/Modified event). Invalid Secret data is logged and the
// current cache is kept, so a bad external write cannot break the running
// process. This lets the controller follow an external key rotation without
// a restart.
func (p *SecretKeyProvider) UpdateFromSecret(sec *corev1.Secret) {
	if sec == nil {
		return
	}
	k, err := decodeKey(sec, p.cfg.DataKey)
	if err != nil {
		log.Errorf("ignoring invalid encryption key secret %s/%s observed externally: %v",
			p.cfg.Namespace, p.cfg.SecretName, err)
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.key != nil && bytes.Equal(p.key, k) {
		return
	}
	p.key = k
	log.Infof("AES encryption key refreshed from secret %s/%s (external change)", p.cfg.Namespace, p.cfg.SecretName)
}

// RestoreSecret recreates the Secret from the cached in-memory key after an
// external deletion (watch Deleted event). Because the running process still
// holds the original key, the deletion self-heals with zero key loss: no
// Guard warnings and no BKENode password recovery are needed. When another
// actor wins the race and recreates the Secret first, the watch delivers the
// winning content via UpdateFromSecret.
func (p *SecretKeyProvider) RestoreSecret(ctx context.Context) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.key == nil {
		log.Error("encryption key secret was deleted but no key is cached; " +
			"a new key will be generated on the next startup (key loss will be reported)")
		return
	}
	created, err := p.cfg.CreateSecret(ctx, keySecret(p.cfg, p.key))
	if err != nil {
		log.Errorf("failed to recreate deleted encryption key secret %s/%s: %v",
			p.cfg.Namespace, p.cfg.SecretName, err)
		return
	}
	if !created {
		log.Infof("encryption key secret %s/%s was recreated by another actor; "+
			"the cache will follow its content", p.cfg.Namespace, p.cfg.SecretName)
		return
	}
	log.Infof("encryption key secret %s/%s was deleted externally and has been "+
		"recreated from the in-memory key", p.cfg.Namespace, p.cfg.SecretName)
}

// generateRandomKey generates a random 32-byte AES-256 key.
func generateRandomKey() ([]byte, error) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, fmt.Errorf("generate encryption key: %w", err)
	}
	return key, nil
}

// keySecret builds the Secret object carrying the base64-encoded key.
func keySecret(cfg SecretKeyProviderConfig, key []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cfg.SecretName,
			Namespace: cfg.Namespace,
		},
		Type: corev1.SecretTypeOpaque,
		Data: map[string][]byte{
			cfg.DataKey: []byte(base64.StdEncoding.EncodeToString(key)),
		},
	}
}

// ErrInvalidKeySecret marks permanent failures caused by invalid Secret
// data (missing field, bad base64, wrong length, legacy key value). Unlike
// transient API failures, retrying these is pointless, so callers can fail
// fast on errors.Is(err, ErrInvalidKeySecret).
var ErrInvalidKeySecret = errors.New("invalid encryption key secret")

// decodeKey validates the Secret data: base64 -> 32 bytes -> not the legacy key.
func decodeKey(sec *corev1.Secret, dataKey string) ([]byte, error) {
	raw, ok := sec.Data[dataKey]
	if !ok || len(raw) == 0 {
		return nil, fmt.Errorf("%w: secret %s/%s has no %q data",
			ErrInvalidKeySecret, sec.Namespace, sec.Name, dataKey)
	}
	k, err := base64.StdEncoding.DecodeString(string(raw))
	if err != nil {
		return nil, fmt.Errorf("%w: secret %s/%s data %q: base64 decode failed: %v",
			ErrInvalidKeySecret, sec.Namespace, sec.Name, dataKey, err)
	}
	if len(k) != 32 {
		return nil, fmt.Errorf("%w: secret %s/%s data %q: expected 32 bytes, got %d",
			ErrInvalidKeySecret, sec.Namespace, sec.Name, dataKey, len(k))
	}
	if string(k) == legacySignature {
		return nil, fmt.Errorf("%w: secret %s/%s contains the compromised legacy key; generate a new one",
			ErrInvalidKeySecret, sec.Namespace, sec.Name)
	}
	return k, nil
}
