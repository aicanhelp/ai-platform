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

package npu

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
)

type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func TestTargetCacheReturnsCachedEntryWithinTTL(t *testing.T) {
	clk := &fakeClock{now: time.Unix(100, 0)}
	cache := newTargetCache(time.Second, clk)
	calls := 0
	scrape := func(_ context.Context) (parsedFamilies, error) {
		calls++
		return parsedFamilies{families: map[string]*dto.MetricFamily{}, fetchedAt: clk.Now()}, nil
	}

	if _, err := cache.get(context.Background(), "u", scrape); err != nil {
		t.Fatalf("first get: %v", err)
	}
	if _, err := cache.get(context.Background(), "u", scrape); err != nil {
		t.Fatalf("second get: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 scrape call within TTL, got %d", calls)
	}

	clk.Advance(time.Second)
	if _, err := cache.get(context.Background(), "u", scrape); err != nil {
		t.Fatalf("third get: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected scrape after TTL expiry, got calls=%d", calls)
	}
}

func TestTargetCacheServesRaceWonEntryWhenScrapeFails(t *testing.T) {
	// Exercises the defensive fallback inside the singleflight closure: the
	// outer fresh-check misses, the scrape returns an error, but a fresh
	// entry has appeared in the cache (simulating a concurrent goroutine
	// that populated it). The fallback must serve that entry instead of
	// surfacing the error.
	clk := &fakeClock{now: time.Unix(100, 0)}
	cache := newTargetCache(2*time.Second, clk)

	scrape := func(_ context.Context) (parsedFamilies, error) {
		cache.store("u", parsedFamilies{
			families:  map[string]*dto.MetricFamily{},
			fetchedAt: clk.Now(),
		})
		return parsedFamilies{}, errors.New("transient")
	}
	got, err := cache.get(context.Background(), "u", scrape)
	if err != nil {
		t.Fatalf("expected race-won fresh entry, got err=%v", err)
	}
	if got.families == nil {
		t.Fatalf("expected race-won entry to carry families")
	}
}

func TestTargetCacheReturnsErrorWhenNoCachedEntry(t *testing.T) {
	clk := &fakeClock{now: time.Unix(100, 0)}
	cache := newTargetCache(time.Second, clk)
	_, err := cache.get(context.Background(), "u", func(_ context.Context) (parsedFamilies, error) {
		return parsedFamilies{}, errors.New("boom")
	})
	if err == nil {
		t.Fatalf("expected error when no cached entry exists")
	}
}

func TestTargetCacheSingleflightCollapsesConcurrentMisses(t *testing.T) {
	clk := &fakeClock{now: time.Unix(100, 0)}
	cache := newTargetCache(time.Second, clk)

	var calls atomic.Int64
	release := make(chan struct{})
	scrape := func(_ context.Context) (parsedFamilies, error) {
		calls.Add(1)
		<-release
		return parsedFamilies{families: map[string]*dto.MetricFamily{}, fetchedAt: clk.Now()}, nil
	}

	const n = 32
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if _, err := cache.get(context.Background(), "u", scrape); err != nil {
				t.Errorf("get: %v", err)
			}
		}()
	}
	// Allow all goroutines to enter and queue on the singleflight before
	// releasing the single in-flight scrape.
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := calls.Load(); got != 1 {
		t.Fatalf("expected 1 scrape call, got %d", got)
	}
}

func TestScrapeTargetParsesPrometheusBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "# TYPE container_npu_utilization gauge\ncontainer_npu_utilization{id=\"0\"} 7\n")
	}))
	defer server.Close()

	clk := &fakeClock{now: time.Unix(100, 0)}
	got, err := scrapeTarget(context.Background(), &http.Client{}, clk, server.URL)
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	if got.families[familyAICoreUtilization] == nil {
		t.Fatalf("expected family %q in scrape result", familyAICoreUtilization)
	}
	if !got.fetchedAt.Equal(clk.Now()) {
		t.Fatalf("unexpected fetchedAt: %s", got.fetchedAt)
	}
}

func TestScrapeTargetReturnsErrorForNon2xxStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	_, err := scrapeTarget(context.Background(), &http.Client{}, realClock{}, server.URL)
	if err == nil {
		t.Fatalf("expected non-2xx error")
	}
}

func TestBuildTargetURL(t *testing.T) {
	got := buildTargetURL("host:9100", "metrics")
	if got != "http://host:9100/metrics" {
		t.Fatalf("unexpected URL: %s", got)
	}
}
