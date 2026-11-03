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

package framework

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
)

func newTestServer(t *testing.T, h http.HandlerFunc) *AgentClient {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return NewAgentClient(srv.URL, "")
}

// mustWrite writes s to w, failing the test on error (codecheck: unchecked
// write return values).
func mustWrite(t *testing.T, w http.ResponseWriter, s string) {
	t.Helper()
	if _, err := io.WriteString(w, s); err != nil {
		t.Fatalf("write response body: %v", err)
	}
}

func TestAgentClientHealthz(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		mustWrite(t, w, "ok")
	})
	code, body, err := c.Healthz()
	if err != nil {
		t.Fatalf("Healthz: %v", err)
	}
	if code != 200 || body != "ok" {
		t.Errorf("Healthz = (%d, %q), want (200, ok)", code, body)
	}
}

func TestAgentClientCreateAndDecodeError(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"success":true,"sandboxId":"sb-1","createdAt":1700000000,`+
			`"endpoints":[{"ip":"10.0.0.1","port":44772,"type":"execd"}],"endpointSource":"agent_pod"}`)
	})
	resp, code, err := c.Create(agentapi.SandboxSpec{SandboxID: "sb-1", Image: "busybox", CPU: "100m", Memory: "64Mi"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if code != 200 || !resp.Success || resp.SandboxID != "sb-1" {
		t.Errorf("Create = (%d, %+v)", code, resp)
	}
	if len(resp.Endpoints) != 1 || resp.Endpoints[0].Port != 44772 {
		t.Errorf("Endpoints = %+v", resp.Endpoints)
	}
	if resp.EndpointSource != agentapi.EndpointSourceAgentPod {
		t.Errorf("EndpointSource = %q", resp.EndpointSource)
	}
}

func TestAgentClientCreateRawMalformed(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Fatalf("drain request body: %v", err)
		}
		w.WriteHeader(http.StatusBadRequest)
	})
	code, _, err := c.CreateRaw("{invalid json")
	if err != nil {
		t.Fatalf("CreateRaw: %v", err)
	}
	if code != 400 {
		t.Errorf("CreateRaw code = %d, want 400", code)
	}
}

func TestAgentClientDeleteAndStatus(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/v1/fluxagent/delete":
			if _, err := io.Copy(io.Discard, r.Body); err != nil {
				t.Fatalf("drain request body: %v", err)
			}
			mustWrite(t, w, `{"success":true}`)
		case "/api/v1/fluxagent/status":
			mustWrite(t, w, `{"fluxsandboxagentId":"agent-1","nodeName":"node-1","runtimeType":"containerd",`+
				`"capacity":10,"allocated":1,"images":["docker.io/library/busybox:latest"],`+
				`"sandboxStatuses":[{"sandboxId":"sb-1","phase":"running","createdAt":1}]}`)
		default:
			http.NotFound(w, r)
		}
	})

	del, code, err := c.Delete("sb-1")
	if err != nil || code != 200 || !del.Success {
		t.Fatalf("Delete = (%d, %+v, %v)", code, del, err)
	}

	st, code, err := c.Status()
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if code != 200 || st.FluxSandboxAgentID != "agent-1" || st.Capacity != 10 {
		t.Errorf("Status = (%d, %+v)", code, st)
	}
	if len(st.SandboxStatuses) != 1 || st.SandboxStatuses[0].Phase != "running" {
		t.Errorf("SandboxStatuses = %+v", st.SandboxStatuses)
	}
}

func TestAgentClientDecodeErrorSurfaced(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{not json`)
	})
	if _, _, err := c.Status(); err == nil {
		t.Fatal("expected decode error to surface, got nil")
	}
}

func TestAgentClientMethodNotAllowed(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusMethodNotAllowed)
	})
	resp, err := c.Do(http.MethodPut, "/api/v1/fluxagent/create")
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 405 {
		t.Errorf("Do status = %d, want 405", resp.StatusCode)
	}
}

func TestAgentClientAuthToken(t *testing.T) {
	var gotAuth string
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		mustWrite(t, w, "ok")
	})
	c.token = "s3cr3t"
	if _, _, err := c.Healthz(); err != nil {
		t.Fatalf("Healthz: %v", err)
	}
	if gotAuth != "Bearer s3cr3t" {
		t.Errorf("Authorization = %q, want Bearer s3cr3t", gotAuth)
	}
}

func TestAgentClientLogsFollow(t *testing.T) {
	c := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("sandboxId") != "sb-1" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("follow") != "true" {
			t.Errorf("follow param missing")
		}
		mustWrite(t, w, "line-0\nline-1\n")
	})
	code, body, err := c.Logs("sb-1", true, 0, 500*time.Millisecond)
	if err != nil {
		t.Fatalf("Logs: %v", err)
	}
	if code != 200 || body != "line-0\nline-1\n" {
		t.Errorf("Logs = (%d, %q)", code, body)
	}
}
