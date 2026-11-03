/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package inflight

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"hermes-router/pkg/epp/internal/testutil"
)

func TestLoadCloneReturnsDeepCopy(t *testing.T) {
	var nilLoad *Load
	nilClone, ok := nilLoad.Clone().(*Load)
	if !ok || nilClone != nil {
		t.Fatalf("expected typed-nil clone for nil load, got %+v", nilLoad.Clone())
	}

	load := &Load{Tokens: 12, Requests: 3}
	cloned := load.Clone().(*Load)
	if cloned.Tokens != 12 || cloned.Requests != 3 {
		t.Fatalf("unexpected cloned load: %+v", cloned)
	}
	cloned.Tokens = 99
	if load.Tokens != 12 {
		t.Fatal("expected Clone to return an independent copy")
	}
}

func waitForFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("expected flushed file %s to exist by now", path)
}

func TestStoreForNameWithPersistenceAppliesConfiguration(t *testing.T) {
	outputPath := filepath.Join(t.TempDir(), "persistence.jsonl")
	store := StoreForNameWithPersistence("with-persistence", PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		FlushInterval:  time.Hour,
		OutputPath:     outputPath,
	})
	t.Cleanup(func() { _ = store.Close() })

	store.StartRequest(RequestStart{
		RequestID:        "req-persistence",
		SelectedEndpoint: newTestEndpointMetadata("decode-persistence", "10.0.90.1"),
		StartedAt:        time.Unix(10, 0),
	})
	store.CompleteRequest(RequestCompletion{RequestID: "req-persistence", CompletedAt: time.Unix(11, 0)})

	// The threshold of one completed record must trigger the async flusher.
	waitForFile(t, outputPath)
}

func TestConfigurePersistenceSignalsThresholdFlush(t *testing.T) {
	outputPath := filepath.Join(t.TempDir(), "configure-persistence.jsonl")
	store := newStore()
	t.Cleanup(func() { _ = store.Close() })

	store.StartRequest(RequestStart{
		RequestID:        "req-configure",
		SelectedEndpoint: newTestEndpointMetadata("decode-configure", "10.0.91.1"),
		StartedAt:        time.Unix(10, 0),
	})
	store.CompleteRequest(RequestCompletion{RequestID: "req-configure", CompletedAt: time.Unix(11, 0)})

	// Enabling persistence with a threshold of one already-buffered record
	// must signal the async flusher immediately.
	store.ConfigurePersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		FlushInterval:  time.Hour,
		OutputPath:     outputPath,
	})
	waitForFile(t, outputPath)

	// Disabling persistence clears the last flush error and stays a no-op.
	store.ConfigurePersistence(PersistenceConfig{})
}

func TestConfigurePersistenceNilReceiverIsSafe(t *testing.T) {
	var store *Store
	store.ConfigurePersistence(PersistenceConfig{Enabled: true})
	store.ConfigureLogger(logr.Discard())
	if err := store.Flush(); err != nil {
		t.Fatalf("expected nil-store flush to be a no-op, got %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("expected nil-store close to be a no-op, got %v", err)
	}
}

func TestConfigureLoggerAcceptsLogger(t *testing.T) {
	ctx, _ := testutil.ContextWithRecordingLogger()
	store := newStore()
	t.Cleanup(func() { _ = store.Close() })

	store.ConfigureLogger(log.FromContext(ctx))
	// Exercise the configured logger path through request expiry bookkeeping.
	store.StartRequest(RequestStart{
		RequestID:        "req-logger",
		SelectedEndpoint: newTestEndpointMetadata("decode-logger", "10.0.92.1"),
		StartedAt:        time.Unix(10, 0),
	})
	store.CompleteRequest(RequestCompletion{RequestID: "req-logger", CompletedAt: time.Unix(11, 0)})
}

func TestConfigurePersistenceRunsExpiryLoopUntilClose(t *testing.T) {
	outputPath := filepath.Join(t.TempDir(), "expiry-loop.jsonl")
	store := newStore()
	store.ConfigurePersistence(PersistenceConfig{
		Enabled:        true,
		FlushThreshold: 1,
		FlushInterval:  time.Hour,
		OutputPath:     outputPath,
	})

	// Close must stop the expiry loop goroutine and its ticker.
	if err := store.Close(); err != nil {
		t.Fatalf("expected close to succeed, got %v", err)
	}
	select {
	case <-store.expiryDone:
	default:
		t.Fatal("expected expiry loop to terminate on close")
	}
}
