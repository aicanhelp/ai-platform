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

package phaseutil

import (
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// ============================ Predicates ============================

// IsConflict returns true if err is a 409 Conflict.
func IsConflict(err error) bool { return apierrors.IsConflict(err) }

// IsNotFound returns true if err is a 404 NotFound.
func IsNotFound(err error) bool { return apierrors.IsNotFound(err) }

// IsTransient returns true if err is a retryable transient error.
// Covers: 409 Conflict, 429 TooManyRequests, 503 ServiceUnavailable,
// 504 ServerTimeout, timeout, network errors.
func IsTransient(err error) bool {
	if err == nil {
		return false
	}
	if apierrors.IsConflict(err) ||
		apierrors.IsTooManyRequests(err) ||
		apierrors.IsServerTimeout(err) ||
		apierrors.IsServiceUnavailable(err) ||
		apierrors.IsTimeout(err) {
		return true
	}
	return utilnet.IsConnectionRefused(err) ||
		utilnet.IsConnectionReset(err) ||
		utilnet.IsProbableEOF(err)
}

// ============================ Backoff Presets ============================

// DefaultBackoff returns the default backoff: 5 steps, 10ms, factor 1.0.
func DefaultBackoff() wait.Backoff { return retry.DefaultRetry }

// ExponentialBackoff returns an exponential backoff: 8 steps, 50ms, factor 2.0.
func ExponentialBackoff() wait.Backoff {
	return wait.Backoff{
		Steps:    8,
		Duration: 50 * time.Millisecond,
		Factor:   2.0,
		Jitter:   0.1,
	}
}

// ReadBackoff returns a short backoff for read ops: 3 steps, 500ms, factor 2.0.
func ReadBackoff() wait.Backoff {
	return wait.Backoff{
		Steps:    3,
		Duration: 500 * time.Millisecond,
		Factor:   2.0,
		Jitter:   0.1,
	}
}

// ============================ Predefined Retry Functions ============================

// RetryOnConflict retries fn on 409 Conflict only.
// fn should contain a full Get→Mutate→Update cycle.
func RetryOnConflict(fn func() error) error {
	b := DefaultBackoff()
	return RetryOnError(fn,
		WithBackoff(b),
		WithPredicate(IsConflict),
		WithOnRetry(func(attempt int, err error) {
			log.Debugf("Retrying on conflict (%d/%d): %v", attempt, b.Steps, err)
		}),
	)
}

// RetryOnTransient retries fn on any transient error (409/429/503/504/network).
func RetryOnTransient(fn func() error) error {
	b := ExponentialBackoff()
	return RetryOnError(fn,
		WithBackoff(b),
		WithPredicate(IsTransient),
		WithOnRetry(func(attempt int, err error) {
			log.Debugf("Retrying on transient error (%d/%d): %v", attempt, b.Steps, err)
		}),
	)
}

// RetryRead retries fn on transient errors with short backoff (for read ops).
func RetryRead(fn func() error) error {
	b := ReadBackoff()
	return RetryOnError(fn,
		WithBackoff(b),
		WithPredicate(IsTransient),
		WithOnRetry(func(attempt int, err error) {
			log.Debugf("Retrying read (%d/%d): %v", attempt, b.Steps, err)
		}),
	)
}

// ============================ Options-based API ============================

// RetryOption configures RetryOnError behavior.
type RetryOption func(*retryConfig)

type retryConfig struct {
	backoff   wait.Backoff
	predicate func(error) bool
	onRetry   func(attempt int, err error)
}

// WithBackoff sets a custom backoff strategy.
func WithBackoff(b wait.Backoff) RetryOption {
	return func(c *retryConfig) { c.backoff = b }
}

// WithPredicate sets a custom retry predicate.
func WithPredicate(p func(error) bool) RetryOption {
	return func(c *retryConfig) { c.predicate = p }
}

// WithOnRetry sets a callback invoked before each retry attempt.
func WithOnRetry(fn func(attempt int, err error)) RetryOption {
	return func(c *retryConfig) { c.onRetry = fn }
}

// RetryOnError retries fn with customizable options.
// Defaults: DefaultBackoff + IsConflict.
func RetryOnError(fn func() error, opts ...RetryOption) error {
	cfg := &retryConfig{
		backoff:   DefaultBackoff(),
		predicate: IsConflict,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.onRetry == nil {
		return retry.OnError(cfg.backoff, cfg.predicate, fn)
	}

	attempt := 0
	return retry.OnError(cfg.backoff, func(err error) bool {
		attempt++
		if cfg.predicate(err) {
			cfg.onRetry(attempt, err)
			return true
		}
		return false
	}, fn)
}
