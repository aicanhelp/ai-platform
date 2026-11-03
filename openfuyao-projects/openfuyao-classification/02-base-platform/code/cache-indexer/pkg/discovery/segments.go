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

package discovery

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"net/http"
	"sort"
	"strings"
	"time"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
)

// segmentsClient reads Mooncake Master /get_all_segments.
// It returns ip -> registered transport endpoint.
type segmentsClient struct {
	httpClient *http.Client
}

func newSegmentsClient(timeout time.Duration) *segmentsClient {
	if timeout <= 0 {
		timeout = config.DefaultDiscoverySegmentsFetchTimeout
	}
	return &segmentsClient{httpClient: &http.Client{Timeout: timeout}}
}

// fetch returns ip -> "host:port" parsed from /get_all_segments.
// The endpoint returns plain text, one "host:port" per line.
func (c *segmentsClient) fetch(ctx context.Context, masterIP string, httpPort int32) (map[string]string, error) {
	url := fmt.Sprintf("http://%s/get_all_segments", net.JoinHostPort(masterIP, fmt.Sprintf("%d", httpPort)))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("master /get_all_segments status=%d", resp.StatusCode)
	}
	out := make(map[string]string)
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		host, _, err := net.SplitHostPort(line)
		if err != nil {
			continue
		}
		// First write wins; if multiple segments share an IP we keep the
		// first one (typically there is exactly one per vLLM pod).
		if _, ok := out[host]; !ok {
			out[host] = line
		}
	}
	return out, sc.Err()
}

// applySegmentsToSnapshot resolves VLLM endpoints from Master segments.
// It returns Pod IPs that have no matching segment.
func applySegmentsToSnapshot(snap *Snapshot, segments map[string]string) []string {
	if snap == nil || len(snap.VLLMPods) == 0 {
		return nil
	}
	missing := make([]string, 0)
	for ip, ep := range snap.VLLMPods {
		seg, ok := segments[ip]
		if !ok {
			missing = append(missing, ip)
			continue
		}
		host, port := splitHostPort(seg)
		ep.MooncakeClient = MooncakeClientEndpoint{
			TransportEndpoint: apis.TransportEndpoint(seg),
			Host:              host,
			TransferPort:      port,
			Resolved:          true,
			Source:            "master-segment",
		}
	}
	sort.Strings(missing)
	return missing
}
