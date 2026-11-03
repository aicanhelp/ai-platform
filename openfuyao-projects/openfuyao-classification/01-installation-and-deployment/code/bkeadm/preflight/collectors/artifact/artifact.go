/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package artifact collects read-only facts about local installation
// artifacts: existence, readability and archive format. The collector never
// unpacks, installs or deletes artifacts.
package artifact

import (
	"context"
	"fmt"
	"os"
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
)

// Collector is the artifact fact collector.
type Collector struct{}

// New returns an artifact Collector.
func New() *Collector { return &Collector{} }

// ID returns the collector identifier.
func (c *Collector) ID() string { return "file-artifact" }

// Collect returns facts for the requested keys. Keys encode the artifact path
// and the fact kind:
//
//	artifact.<path>.exists
//	artifact.<path>.readable
//	artifact.<path>.format
func (c *Collector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	out := make([]api.Fact, 0, len(req.Keys))
	for index := range req.Keys {
		key := req.Keys[index]
		if err := ctx.Err(); err != nil {
			out = append(out, canceledFact(key, err))
		} else {
			out = append(out, c.collectKey(key))
		}
	}
	return out, nil
}

func canceledFact(key string, err error) api.Fact {
	return api.Fact{Key: key, Available: false,
		Error: &api.FactError{Code: api.FactErrTimeout, Message: err.Error()}}
}

func (c *Collector) collectKey(key string) api.Fact {
	// key: artifact.<path>.<kind>
	rest := strings.TrimPrefix(key, "artifact.")
	// Split into path and kind. The kind is the last segment after the final
	// dot; the path is everything before. Paths may contain dots.
	lastDot := strings.LastIndex(rest, ".")
	if lastDot < 0 {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: "no kind in artifact key"}}
	}
	path := rest[:lastDot]
	kind := rest[lastDot+1:]
	switch kind {
	case "exists":
		return c.existsFact(key, path)
	case "readable":
		return c.readableFact(key, path)
	case "format":
		return c.formatFact(key, path)
	default:
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "unknown artifact kind: " + kind}}
	}
}

func (c *Collector) existsFact(key, path string) api.Fact {
	info, err := os.Stat(path)
	if err != nil {
		return api.Fact{Key: key, Available: true, Value: false}
	}
	return api.Fact{Key: key, Available: true, Value: true,
		Evidence: []api.Evidence{{Kind: "file", Source: "file-artifact", Locator: path,
			Observed: fmt.Sprintf("size=%d", info.Size())}}}
}

func (c *Collector) readableFact(key, path string) api.Fact {
	f, err := os.Open(path)
	if err != nil {
		return api.Fact{Key: key, Available: true, Value: false}
	}
	if err := f.Close(); err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrUnreachable, Message: err.Error()}}
	}
	return api.Fact{Key: key, Available: true, Value: true,
		Evidence: []api.Evidence{{Kind: "file", Source: "file-artifact", Locator: path}}}
}

func (c *Collector) formatFact(key, path string) api.Fact {
	lower := strings.ToLower(path)
	var format string
	switch {
	case strings.HasSuffix(lower, ".tar.gz"), strings.HasSuffix(lower, ".tgz"):
		format = "tar.gz"
	case strings.HasSuffix(lower, ".tar"):
		format = "tar"
	case strings.HasSuffix(lower, ".rpm"):
		format = "rpm"
	case strings.HasSuffix(lower, ".deb"):
		format = "deb"
	case strings.HasSuffix(lower, ".yaml"), strings.HasSuffix(lower, ".yml"):
		format = "yaml"
	default:
		format = "unknown"
	}
	return api.Fact{Key: key, Available: true, Value: format,
		Evidence: []api.Evidence{{Kind: "file", Source: "file-artifact", Locator: path}}}
}
