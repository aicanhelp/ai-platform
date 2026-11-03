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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
)

// AgentClient is a typed HTTP client for the flux-sandbox-agent control API.
type AgentClient struct {
	// BaseURL points at the agent, e.g. http://127.0.0.1:5758 via port-forward.
	BaseURL string
	// HTTP is the client used for all requests.
	HTTP *http.Client
	// token, when set, is sent as an Authorization Bearer token.
	token string
}

// NewAgentClient builds an agent client for baseURL with an optional auth token.
func NewAgentClient(baseURL, token string) *AgentClient {
	return &AgentClient{
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: 60 * time.Second},
		token:   token,
	}
}

// Healthz calls GET /healthz and returns the status code and body.
func (c *AgentClient) Healthz() (int, string, error) {
	return c.get("/healthz")
}

// Create calls POST /create with the given sandbox spec.
func (c *AgentClient) Create(spec agentapi.SandboxSpec) (agentapi.CreateSandboxResponse, int, error) {
	var out agentapi.CreateSandboxResponse
	code, err := c.postJSON("/api/v1/fluxagent/create", agentapi.CreateSandboxRequest{Sandbox: spec}, &out)
	return out, code, err
}

// CreateRaw posts an arbitrary body to /create and returns code and body,
// for malformed-payload tests.
func (c *AgentClient) CreateRaw(rawBody string) (int, string, error) {
	resp, err := c.do(http.MethodPost, "/api/v1/fluxagent/create", "application/json", rawBody)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), err
}

// Delete calls POST /delete for the given sandbox ID.
func (c *AgentClient) Delete(sandboxID string) (agentapi.DeleteSandboxResponse, int, error) {
	var out agentapi.DeleteSandboxResponse
	code, err := c.postJSON("/api/v1/fluxagent/delete", agentapi.DeleteSandboxRequest{SandboxID: sandboxID}, &out)
	return out, code, err
}

// Status calls GET /status.
func (c *AgentClient) Status() (agentapi.FluxSandboxAgentStatus, int, error) {
	var out agentapi.FluxSandboxAgentStatus
	code, err := c.getJSON("/api/v1/fluxagent/status", &out)
	return out, code, err
}

// Logs calls GET /logs for the sandbox. When follow is true the response is
// read for up to readFor and the accumulated output is returned. tail > 0
// requests only the last tail lines.
func (c *AgentClient) Logs(sandboxID string, follow bool, tail int, readFor time.Duration) (int, string, error) {
	path := "/api/v1/fluxagent/logs?sandboxId=" + sandboxID
	if follow {
		path += "&follow=true"
	}
	if tail > 0 {
		path += "&tail=" + strconv.Itoa(tail)
	}

	ctx := context.Background()
	var cancel context.CancelFunc = func() {}
	if readFor > 0 {
		ctx, cancel = context.WithTimeout(ctx, readFor)
	}
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return 0, "", err
	}
	c.setAuth(req)

	resp, err := c.HTTP.Do(req)
	if err != nil {
		// A follow read that ends via the context deadline is not an error;
		// report whatever we can.
		if ctx.Err() != nil {
			return 0, "", nil
		}
		return 0, "", err
	}
	defer resp.Body.Close()

	b, err := io.ReadAll(resp.Body)
	if err != nil && ctx.Err() != nil {
		// Body read cut short by our own deadline: return what we have.
		return resp.StatusCode, string(b), nil
	}
	return resp.StatusCode, string(b), err
}

// Do performs an arbitrary request against the agent (e.g. for method-not-
// allowed tests) and closes nothing: the caller owns the response body.
func (c *AgentClient) Do(method, path string) (*http.Response, error) {
	req, err := http.NewRequest(method, c.BaseURL+path, nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)
	return c.HTTP.Do(req)
}

func (c *AgentClient) get(path string) (int, string, error) {
	resp, err := c.Do(http.MethodGet, path)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), err
}

func (c *AgentClient) getJSON(path string, out interface{}) (int, error) {
	resp, err := c.Do(http.MethodGet, path)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if err := decodeBody(resp.Body, out); err != nil {
		return resp.StatusCode, fmt.Errorf("decode %s: %w", path, err)
	}
	return resp.StatusCode, nil
}

func (c *AgentClient) postJSON(path string, in, out interface{}) (int, error) {
	body, err := json.Marshal(in)
	if err != nil {
		return 0, err
	}
	resp, err := c.do(http.MethodPost, path, "application/json", string(body))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if err := decodeBody(resp.Body, out); err != nil {
		return resp.StatusCode, fmt.Errorf("decode %s: %w", path, err)
	}
	return resp.StatusCode, nil
}

func (c *AgentClient) do(method, path, contentType, body string) (*http.Response, error) {
	req, err := http.NewRequest(method, c.BaseURL+path, bytes.NewReader([]byte(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	c.setAuth(req)
	return c.HTTP.Do(req)
}

func (c *AgentClient) setAuth(req *http.Request) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

func decodeBody(r io.Reader, out interface{}) error {
	b, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	if len(bytes.TrimSpace(b)) == 0 {
		return nil
	}
	return json.Unmarshal(b, out)
}
