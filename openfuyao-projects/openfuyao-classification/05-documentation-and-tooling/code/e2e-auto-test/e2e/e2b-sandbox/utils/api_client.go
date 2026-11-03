package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// CreateSandboxRequest matches the OpenSandbox CreateSandboxRequest.
type CreateSandboxRequest struct {
	Image          *ImageSpec             `json:"image,omitempty"`
	SnapshotID     string                 `json:"snapshotId,omitempty"`
	Entrypoint     []string               `json:"entrypoint,omitempty"`
	ResourceLimits *ResourceLimits        `json:"resourceLimits,omitempty"`
	Extensions     map[string]string      `json:"extensions,omitempty"`
	Timeout        int                    `json:"timeout,omitempty"`
}

type ImageSpec struct {
	URI string `json:"uri"`
}

type ResourceLimits struct {
	CPU    string `json:"cpu,omitempty"`
	Memory string `json:"memory,omitempty"`
}

// CreateSandboxResponse is the response from POST /v1/sandboxes.
type CreateSandboxResponse struct {
	ID        string        `json:"id"`
	Status    SandboxStatus `json:"status"`
	Entrypoint []string     `json:"entrypoint,omitempty"`
}

// Sandbox is the full sandbox resource from GET /v1/sandboxes/{id}.
type Sandbox struct {
	ID     string        `json:"id"`
	Status SandboxStatus `json:"status"`
}

// SandboxStatus holds the sandbox lifecycle status.
type SandboxStatus struct {
	State string `json:"state"`
}

// CreateSnapshotRequest is the request body for POST /v1/sandboxes/{id}/snapshots.
type CreateSnapshotRequest struct {
	Name string `json:"name,omitempty"`
}

// Snapshot is the response from snapshot operations.
type Snapshot struct {
	ID         string          `json:"id"`
	SandboxID  string          `json:"sandboxId"`
	Name       string          `json:"name,omitempty"`
	Image      string          `json:"image,omitempty"`
	TeamID     string          `json:"teamId,omitempty"`
	Status     SnapshotStatus  `json:"status"`
}

// SnapshotStatus holds the snapshot lifecycle status.
type SnapshotStatus struct {
	State string `json:"state"`
}

// ListSnapshotsResponse is the response from GET /v1/snapshots.
type ListSnapshotsResponse struct {
	Items      []Snapshot      `json:"items"`
	Pagination PaginationInfo  `json:"pagination"`
}

// PaginationInfo holds pagination metadata.
type PaginationInfo struct {
	Page        int  `json:"page"`
	PageSize    int  `json:"pageSize"`
	TotalItems  int  `json:"totalItems"`
	TotalPages  int  `json:"totalPages"`
	HasNextPage bool `json:"hasNextPage"`
}

// ErrorResponse matches the OpenSandbox error response.
type ErrorResponse struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Detail  any    `json:"detail,omitempty"`
}

// APIClient calls the OpenSandbox Server HTTP API.
type APIClient struct {
	baseURL string
	http    *http.Client
}

// NewAPIClient creates a new APIClient with a default 120s timeout.
func NewAPIClient(baseURL string) *APIClient {
	return &APIClient{
		baseURL: baseURL,
		http:    &http.Client{Timeout: 120 * time.Second},
	}
}

// CreateSandbox POST /v1/sandboxes
func (c *APIClient) CreateSandbox(ctx context.Context, req *CreateSandboxRequest) (*CreateSandboxResponse, int, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal create request: %w", err)
	}
	respBody, status, err := c.do(ctx, http.MethodPost, "/v1/sandboxes", body)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var resp CreateSandboxResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, status, fmt.Errorf("unmarshal create response: %w", err)
	}
	return &resp, status, nil
}

// GetSandbox GET /v1/sandboxes/{id}
func (c *APIClient) GetSandbox(ctx context.Context, id string) (*Sandbox, int, error) {
	respBody, status, err := c.do(ctx, http.MethodGet, "/v1/sandboxes/"+id, nil)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var sb Sandbox
	if err := json.Unmarshal(respBody, &sb); err != nil {
		return nil, status, fmt.Errorf("unmarshal sandbox response: %w", err)
	}
	return &sb, status, nil
}

// DeleteSandbox DELETE /v1/sandboxes/{id}
func (c *APIClient) DeleteSandbox(ctx context.Context, id string) (int, error) {
	_, status, err := c.do(ctx, http.MethodDelete, "/v1/sandboxes/"+id, nil)
	return status, err
}

// PauseSandbox POST /v1/sandboxes/{id}/pause (returns 202, async)
func (c *APIClient) PauseSandbox(ctx context.Context, id string) (int, error) {
	_, status, err := c.do(ctx, http.MethodPost, "/v1/sandboxes/"+id+"/pause", nil)
	return status, err
}

// ResumeSandbox POST /v1/sandboxes/{id}/resume (returns 202, async)
func (c *APIClient) ResumeSandbox(ctx context.Context, id string) (int, error) {
	_, status, err := c.do(ctx, http.MethodPost, "/v1/sandboxes/"+id+"/resume", nil)
	return status, err
}

// CreateSnapshot POST /v1/sandboxes/{id}/snapshots (returns 201, sync)
func (c *APIClient) CreateSnapshot(ctx context.Context, sandboxID, name string) (*Snapshot, int, error) {
	body, err := json.Marshal(&CreateSnapshotRequest{Name: name})
	if err != nil {
		return nil, 0, fmt.Errorf("marshal snapshot request: %w", err)
	}
	respBody, status, err := c.do(ctx, http.MethodPost, "/v1/sandboxes/"+sandboxID+"/snapshots", body)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var snap Snapshot
	if err := json.Unmarshal(respBody, &snap); err != nil {
		return nil, status, fmt.Errorf("unmarshal snapshot response: %w", err)
	}
	return &snap, status, nil
}

// GetSnapshot GET /v1/snapshots/{id}
func (c *APIClient) GetSnapshot(ctx context.Context, id string) (*Snapshot, int, error) {
	respBody, status, err := c.do(ctx, http.MethodGet, "/v1/snapshots/"+id, nil)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var snap Snapshot
	if err := json.Unmarshal(respBody, &snap); err != nil {
		return nil, status, fmt.Errorf("unmarshal snapshot response: %w", err)
	}
	return &snap, status, nil
}

// ListSnapshots GET /v1/snapshots?page={page}&pageSize={pageSize}
func (c *APIClient) ListSnapshots(ctx context.Context, page, pageSize int) (*ListSnapshotsResponse, int, error) {
	path := fmt.Sprintf("/v1/snapshots?page=%d&pageSize=%d", page, pageSize)
	respBody, status, err := c.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var resp ListSnapshotsResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, status, fmt.Errorf("unmarshal list snapshots response: %w", err)
	}
	return &resp, status, nil
}

// DeleteSnapshot DELETE /v1/snapshots/{id}
func (c *APIClient) DeleteSnapshot(ctx context.Context, id string) (int, error) {
	_, status, err := c.do(ctx, http.MethodDelete, "/v1/snapshots/"+id, nil)
	return status, err
}

// do is the shared HTTP request helper. Returns (responseBody, statusCode, error).
func (c *APIClient) do(ctx context.Context, method, path string, body []byte) ([]byte, int, error) {
	url := c.baseURL + path
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("http %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response: %w", err)
	}
	return respBody, resp.StatusCode, nil
}
