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
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// secretIO is a closure-based fake for the provider config.
type secretIO struct {
	get    func() (*corev1.Secret, error)
	create func(*corev1.Secret) (bool, error)
	guard  func() error

	getCalls    int
	createCalls int
	guardCalls  int

	created *corev1.Secret
}

func (io *secretIO) config() SecretKeyProviderConfig {
	return SecretKeyProviderConfig{
		GetSecret: func(ctx context.Context) (*corev1.Secret, error) {
			io.getCalls++
			return io.get()
		},
		CreateSecret: func(ctx context.Context, s *corev1.Secret) (bool, error) {
			io.createCalls++
			return io.create(s)
		},
		Guard: func(ctx context.Context) error {
			io.guardCalls++
			return io.guard()
		},
	}
}

func validSecret(key []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: DefaultSecretName, Namespace: DefaultSecretNamespace},
		Data:       map[string][]byte{SecretDataKey: []byte(base64.StdEncoding.EncodeToString(key))},
	}
}

func newTestKey(t *testing.T) []byte {
	t.Helper()
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	return key
}

func TestNewSecretKeyProviderValidation(t *testing.T) {
	_, err := NewSecretKeyProvider(SecretKeyProviderConfig{})
	assert.Error(t, err, "GetSecret/CreateSecret must be required")

	cfg := SecretKeyProviderConfig{
		GetSecret: func(ctx context.Context) (*corev1.Secret, error) { return nil, nil },
	}
	_, err = NewSecretKeyProvider(cfg)
	assert.Error(t, err, "CreateSecret must be required")
}

func TestSecretKeyProvider_ExistingSecret(t *testing.T) {
	key := newTestKey(t)
	io := &secretIO{get: func() (*corev1.Secret, error) { return validSecret(key), nil }}
	p, err := NewSecretKeyProvider(io.config())
	require.NoError(t, err)

	got, err := p.Key(context.Background())
	require.NoError(t, err)
	assert.Equal(t, key, got)

	// Second call is served from cache: no extra API call, no guard call.
	_, err = p.Key(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, io.getCalls)
	assert.Equal(t, 0, io.guardCalls)
	assert.Equal(t, 0, io.createCalls)
}

func TestSecretKeyProvider_InvalidSecretData(t *testing.T) {
	cases := []struct {
		name   string
		secret *corev1.Secret
	}{
		{"missing data key", &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: DefaultSecretName, Namespace: DefaultSecretNamespace},
			Data:       map[string][]byte{},
		}},
		{"invalid base64", &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: DefaultSecretName, Namespace: DefaultSecretNamespace},
			Data:       map[string][]byte{SecretDataKey: []byte("!!!not-base64!!!")},
		}},
		{"wrong length", &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: DefaultSecretName, Namespace: DefaultSecretNamespace},
			Data:       map[string][]byte{SecretDataKey: []byte(base64.StdEncoding.EncodeToString([]byte("too-short")))}},
		},
		{"legacy key value", validSecret([]byte(legacySignature))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			io := &secretIO{get: func() (*corev1.Secret, error) { return tc.secret, nil }}
			p, err := NewSecretKeyProvider(io.config())
			require.NoError(t, err)

			_, err = p.Key(context.Background())
			assert.Error(t, err)
			assert.Contains(t, err.Error(), DefaultSecretName)
			// Permanent failures carry the sentinel so startup retry loops
			// can fail fast instead of retrying pointlessly.
			assert.ErrorIs(t, err, ErrInvalidKeySecret)
		})
	}
}

func TestSecretKeyProvider_GenerateWhenMissing(t *testing.T) {
	var io *secretIO
	io = &secretIO{
		get:    func() (*corev1.Secret, error) { return nil, nil },
		create: func(s *corev1.Secret) (bool, error) { io.created = s; return true, nil },
		guard:  func() error { return nil },
	}
	p, err := NewSecretKeyProvider(io.config())
	require.NoError(t, err)

	key, err := p.Key(context.Background())
	require.NoError(t, err)
	assert.Len(t, key, 32)
	assert.Equal(t, 1, io.guardCalls)
	assert.Equal(t, 1, io.createCalls)

	// The created Secret stores the base64-encoded key.
	require.NotNil(t, io.created)
	decoded, err := base64.StdEncoding.DecodeString(string(io.created.Data[SecretDataKey]))
	require.NoError(t, err)
	assert.Equal(t, key, decoded)
	assert.Equal(t, DefaultSecretName, io.created.Name)
	assert.Equal(t, DefaultSecretNamespace, io.created.Namespace)
}

func TestSecretKeyProvider_GuardErrorDefers(t *testing.T) {
	io := &secretIO{
		get:    func() (*corev1.Secret, error) { return nil, nil },
		create: func(s *corev1.Secret) (bool, error) { return true, nil },
		guard:  func() error { return errors.New("cannot list BKENodes") },
	}
	p, err := NewSecretKeyProvider(io.config())
	require.NoError(t, err)

	_, err = p.Key(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot list BKENodes")
	assert.Equal(t, 0, io.createCalls, "generation must be deferred on guard error")

	// Failure is not cached: the next call retries (guard runs again).
	assert.Equal(t, 1, io.guardCalls)
	_, _ = p.Key(context.Background())
	assert.Equal(t, 2, io.guardCalls)
}

func TestSecretKeyProvider_ConcurrentCreateRace(t *testing.T) {
	winnerKey := newTestKey(t)
	var created *corev1.Secret
	var io *secretIO
	io = &secretIO{
		get: func() (*corev1.Secret, error) {
			// NotFound first, then the winner's Secret after the race.
			if created == nil {
				return nil, nil
			}
			return validSecret(winnerKey), nil
		},
		create: func(s *corev1.Secret) (bool, error) {
			created = s
			return false, nil // AlreadyExists: another replica won.
		},
		guard: func() error { return nil },
	}
	p, err := NewSecretKeyProvider(io.config())
	require.NoError(t, err)

	got, err := p.Key(context.Background())
	require.NoError(t, err)
	assert.Equal(t, winnerKey, got, "must reuse the key created by the winning replica")
}

func TestSecretKeyProvider_GetErrorNotCached(t *testing.T) {
	fail := true
	key := newTestKey(t)
	io := &secretIO{
		get: func() (*corev1.Secret, error) {
			if fail {
				return nil, errors.New("apiserver down")
			}
			return validSecret(key), nil
		},
	}
	p, err := NewSecretKeyProvider(io.config())
	require.NoError(t, err)

	_, err = p.Key(context.Background())
	assert.Error(t, err)

	fail = false
	got, err := p.Key(context.Background())
	require.NoError(t, err)
	assert.Equal(t, key, got)
}

func TestSecretKeyProvider_UpdateFromSecret(t *testing.T) {
	t.Run("valid new key refreshes the cache", func(t *testing.T) {
		oldKey := newTestKey(t)
		io := &secretIO{get: func() (*corev1.Secret, error) { return validSecret(oldKey), nil }}
		p, err := NewSecretKeyProvider(io.config())
		require.NoError(t, err)
		_, err = p.Key(context.Background())
		require.NoError(t, err)

		newKey := newTestKey(t)
		p.UpdateFromSecret(validSecret(newKey))

		got, err := p.Key(context.Background())
		require.NoError(t, err)
		assert.Equal(t, newKey, got, "the cache must follow the external rotation")
	})

	t.Run("invalid secret keeps the current cache", func(t *testing.T) {
		key := newTestKey(t)
		io := &secretIO{get: func() (*corev1.Secret, error) { return validSecret(key), nil }}
		p, err := NewSecretKeyProvider(io.config())
		require.NoError(t, err)
		_, err = p.Key(context.Background())
		require.NoError(t, err)

		p.UpdateFromSecret(&corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: DefaultSecretName, Namespace: DefaultSecretNamespace},
			Data:       map[string][]byte{SecretDataKey: []byte("!!!not-base64!!!")},
		})

		got, err := p.Key(context.Background())
		require.NoError(t, err)
		assert.Equal(t, key, got, "a bad external write must not break the running process")
	})

	t.Run("same key and nil secret are no-ops", func(t *testing.T) {
		key := newTestKey(t)
		io := &secretIO{get: func() (*corev1.Secret, error) { return validSecret(key), nil }}
		p, err := NewSecretKeyProvider(io.config())
		require.NoError(t, err)
		_, err = p.Key(context.Background())
		require.NoError(t, err)

		p.UpdateFromSecret(nil)
		p.UpdateFromSecret(validSecret(key))

		got, err := p.Key(context.Background())
		require.NoError(t, err)
		assert.Equal(t, key, got)
	})
}

func TestSecretKeyProvider_RestoreSecret(t *testing.T) {
	t.Run("recreates the secret with the cached key", func(t *testing.T) {
		key := newTestKey(t)
		var restored *corev1.Secret
		io := &secretIO{
			get:    func() (*corev1.Secret, error) { return validSecret(key), nil },
			create: func(s *corev1.Secret) (bool, error) { restored = s; return true, nil },
		}
		p, err := NewSecretKeyProvider(io.config())
		require.NoError(t, err)
		_, err = p.Key(context.Background())
		require.NoError(t, err)

		p.RestoreSecret(context.Background())

		require.NotNil(t, restored)
		decoded, err := base64.StdEncoding.DecodeString(string(restored.Data[SecretDataKey]))
		require.NoError(t, err)
		assert.Equal(t, key, decoded, "the deleted Secret must be restored with the in-memory key")
		assert.Equal(t, DefaultSecretName, restored.Name)
		assert.Equal(t, DefaultSecretNamespace, restored.Namespace)
	})

	t.Run("concurrent recreation is tolerated", func(t *testing.T) {
		key := newTestKey(t)
		io := &secretIO{
			get:    func() (*corev1.Secret, error) { return validSecret(key), nil },
			create: func(*corev1.Secret) (bool, error) { return false, nil },
		}
		p, err := NewSecretKeyProvider(io.config())
		require.NoError(t, err)
		_, err = p.Key(context.Background())
		require.NoError(t, err)

		p.RestoreSecret(context.Background()) // must not panic or change the cache

		got, err := p.Key(context.Background())
		require.NoError(t, err)
		assert.Equal(t, key, got)
	})

	t.Run("no cached key does not create anything", func(t *testing.T) {
		created := false
		io := &secretIO{
			get:    func() (*corev1.Secret, error) { return nil, nil },
			create: func(*corev1.Secret) (bool, error) { created = true; return true, nil },
		}
		p, err := NewSecretKeyProvider(io.config())
		require.NoError(t, err)

		p.RestoreSecret(context.Background())

		assert.False(t, created)
	})
}

// TestSecretKeyProvider_WithCryptoRoundTrip wires the provider into the
// package-level crypto functions and verifies encrypt/decrypt round trips.
func TestSecretKeyProvider_WithCryptoRoundTrip(t *testing.T) {
	key := newTestKey(t)
	io := &secretIO{get: func() (*corev1.Secret, error) { return validSecret(key), nil }}
	p, err := NewSecretKeyProvider(io.config())
	require.NoError(t, err)

	prev := currentKeyProvider()
	SetKeyProvider(p)
	t.Cleanup(func() { SetKeyProvider(prev) })

	enc, err := AesEncrypt("roundtrip-password")
	require.NoError(t, err)
	dec, err := AesDecrypt(enc)
	require.NoError(t, err)
	assert.Equal(t, "roundtrip-password", dec)
}
