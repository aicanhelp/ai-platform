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
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestSegmentsClient_FetchAndApply(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/get_all_segments" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("10.0.0.1:16377\n10.0.0.1:17000\n10.0.0.3:18888\n\n"))
	}))
	defer srv.Close()

	// Parse host:port from the test server URL (strip "http://").
	addr := strings.TrimPrefix(srv.URL, "http://")
	parts := strings.Split(addr, ":")
	if len(parts) != 2 {
		t.Fatalf("unexpected test server URL: %s", srv.URL)
	}
	port, _ := strconv.Atoi(parts[1])

	c := newSegmentsClient(0)
	got, err := c.fetch(context.Background(), parts[0], int32(port))
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if got["10.0.0.1"] != "10.0.0.1:16377" || got["10.0.0.3"] != "10.0.0.3:18888" {
		t.Fatalf("parsed map = %v", got)
	}

	snap := &Snapshot{VLLMPods: map[string]*VLLMPodEndpoint{
		"10.0.0.1": {PodIP: "10.0.0.1"},
		"10.0.0.2": {PodIP: "10.0.0.2"},
	}}
	missing := applySegmentsToSnapshot(snap, got)
	if len(missing) != 1 || missing[0] != "10.0.0.2" {
		t.Fatalf("missing = %v, want [10.0.0.2]", missing)
	}

	got1 := snap.VLLMPods["10.0.0.1"].MooncakeClient
	if string(got1.TransportEndpoint) != "10.0.0.1:16377" || got1.TransferPort != 16377 || got1.Source != "master-segment" {
		t.Errorf("override mismatch: %+v", got1)
	}
	if snap.VLLMPods["10.0.0.2"].MooncakeClient.Resolved {
		t.Errorf("10.0.0.2 has no segment, must remain unresolved")
	}
}
