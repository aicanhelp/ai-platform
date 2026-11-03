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
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	"golang.org/x/sync/singleflight"
)

// parsedFamilies is the parsed Prometheus output of a single scrape, retained
// in the per-target cache. fetchedAt is used both for TTL accounting and as
// the fallback observation time when no per-sample timestamp exists.
//
// The families map and the *MetricFamily values it points to are treated as
// immutable once stored in the cache: aggregate.applySamples reads them from
// multiple goroutines (one per polling endpoint) without locking.
type parsedFamilies struct {
	families  map[string]*dto.MetricFamily
	fetchedAt time.Time
}

// clock isolates the package from time.Now for deterministic testing.
type clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

// targetCache stores the most recent parsed scrape per target URL with a TTL,
// and uses singleflight so that N concurrent endpoint pollers attached to the
// same exporter target collapse into a single HTTP scrape.
type targetCache struct {
	ttl     time.Duration
	clock   clock
	mu      sync.RWMutex
	entries map[string]parsedFamilies
	group   singleflight.Group
}

type scrapeFunc func(ctx context.Context) (parsedFamilies, error)

func newTargetCache(ttl time.Duration, c clock) *targetCache {
	return &targetCache{ttl: ttl, clock: c, entries: make(map[string]parsedFamilies)}
}

func (c *targetCache) get(ctx context.Context, targetURL string, scrape scrapeFunc) (parsedFamilies, error) {
	if entry, ok := c.fresh(targetURL); ok {
		return entry, nil
	}

	// Detach the in-flight scrape from the first caller's cancellation so a
	// shutdown or per-Poll cancel on one endpoint cannot fail the same scrape
	// for every concurrent endpoint waiting on singleflight. The HTTP client
	// already enforces its own timeout via http.Client.Timeout.
	scrapeCtx := context.WithoutCancel(ctx)
	value, err, _ := c.group.Do(targetURL, func() (any, error) {
		if entry, ok := c.fresh(targetURL); ok {
			return entry, nil
		}
		entry, scrapeErr := scrape(scrapeCtx)
		if scrapeErr != nil {
			// Defensive race fallback: if another goroutine populated the
			// cache between our outer fresh-check miss and this scrape
			// returning, prefer that fresh entry over the transient error.
			// This is not stale-while-erroring across TTL boundaries: an
			// expired entry is treated as a miss by c.fresh and the error
			// surfaces to the caller.
			if cached, ok := c.fresh(targetURL); ok {
				return cached, nil
			}
			return parsedFamilies{}, scrapeErr
		}
		c.store(targetURL, entry)
		return entry, nil
	})
	if err != nil {
		return parsedFamilies{}, err
	}
	return value.(parsedFamilies), nil
}

func (c *targetCache) fresh(targetURL string) (parsedFamilies, bool) {
	c.mu.RLock()
	entry, ok := c.entries[targetURL]
	c.mu.RUnlock()
	if !ok || c.clock.Now().Sub(entry.fetchedAt) >= c.ttl {
		return parsedFamilies{}, false
	}
	return entry, true
}

func (c *targetCache) store(targetURL string, entry parsedFamilies) {
	c.mu.Lock()
	c.entries[targetURL] = entry
	c.mu.Unlock()
}

// scrapeTarget performs an HTTP GET against the exporter target and parses
// the response as Prometheus text exposition.
func scrapeTarget(ctx context.Context, client *http.Client, c clock, targetURL string) (parsedFamilies, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return parsedFamilies{}, fmt.Errorf("create scrape request for %s: %w", targetURL, err)
	}
	response, err := client.Do(request)
	if err != nil {
		return parsedFamilies{}, fmt.Errorf("scrape %s: %w", targetURL, err)
	}
	defer response.Body.Close()

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return parsedFamilies{}, fmt.Errorf("scrape %s: unexpected status %d", targetURL, response.StatusCode)
	}

	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(response.Body)
	if err != nil {
		return parsedFamilies{}, fmt.Errorf("parse prometheus metrics from %s: %w", targetURL, err)
	}
	return parsedFamilies{families: families, fetchedAt: c.Now()}, nil
}

// buildTargetURL composes the plaintext HTTP scrape URL for the given
// host:port and path. HTTPS is intentionally not supported in this iteration.
func buildTargetURL(host, path string) string {
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return (&url.URL{Scheme: "http", Host: host, Path: path}).String()
}
