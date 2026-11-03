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

// CreateInstanceRequest matches the operator's CreateInstanceRequest.
type CreateInstanceRequest struct {
	AppRef    string            `json:"app_ref"`
	PgVersion int               `json:"pg_version"`
	Resources InstanceResources `json:"resources"`
}

// ResizeInstanceRequest matches the operator's ResizeInstanceRequest.
type ResizeInstanceRequest struct {
	Containers []ContainerResource `json:"containers"`
}

type ContainerResource struct {
	Name      string            `json:"name"`
	Resources InstanceResources `json:"resources"`
}

// InstanceResources matches the operator's InstanceResources. All fields
// optional to support partial resize requests.
type InstanceResources struct {
	Cpu         string `json:"cpu,omitempty"`
	Memory      string `json:"memory,omitempty"`
	CpuLimit    string `json:"cpu_limit,omitempty"`
	MemoryLimit string `json:"memory_limit,omitempty"`
}

// InstanceResponse matches the operator's InstanceResponse.
type InstanceResponse struct {
	PodID    string `json:"pod_id"`
	PodName  string `json:"pod_name"`
	Endpoint string `json:"endpoint"`
}

// ErrorResponse matches the operator's ErrorResponse.
type ErrorResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// APIClient calls the operator REST API.
type APIClient struct {
	baseURL string
	token   string
	http    *http.Client
}

func NewAPIClient(baseURL, token string) *APIClient {
	return &APIClient{
		baseURL: baseURL,
		token:   token,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

func NewAPIClientWithTimeout(baseURL, token string, timeout time.Duration) *APIClient {
	return &APIClient{
		baseURL: baseURL,
		token:   token,
		http:    &http.Client{Timeout: timeout},
	}
}

// WithToken returns a new APIClient pointing at the same base URL but using
// the given token (used by auth tests to swap credentials).
func (c *APIClient) WithToken(token string) *APIClient {
	return NewAPIClient(c.baseURL, token)
}

// Allocate POST /api/v1/instances?namespace=<ns>.
func (c *APIClient) Allocate(ctx context.Context, ns string, req *CreateInstanceRequest) (*InstanceResponse, int, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal allocate request: %w", err)
	}
	respBody, status, err := c.do(ctx, http.MethodPost, "/api/v1/instances", ns, body)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var resp InstanceResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, status, fmt.Errorf("unmarshal allocate response: %w", err)
	}
	return &resp, status, nil
}

// Resize PATCH /api/v1/instances/<appRef>?namespace=<ns>.
func (c *APIClient) Resize(ctx context.Context, ns, appRef string, req *ResizeInstanceRequest) (*InstanceResponse, int, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal resize request: %w", err)
	}
	respBody, status, err := c.do(ctx, http.MethodPatch, "/api/v1/instances/"+appRef, ns, body)
	if err != nil {
		return nil, status, err
	}
	if status >= 300 {
		return nil, status, nil
	}
	var resp InstanceResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, status, fmt.Errorf("unmarshal resize response: %w", err)
	}
	return &resp, status, nil
}

// Release DELETE /api/v1/instances/<appRef>?namespace=<ns>.
func (c *APIClient) Release(ctx context.Context, ns, appRef string) (int, error) {
	_, status, err := c.do(ctx, http.MethodDelete, "/api/v1/instances/"+appRef, ns, nil)
	return status, err
}

func (c *APIClient) do(ctx context.Context, method, path, ns string, body []byte) ([]byte, int, error) {
	url := fmt.Sprintf("%s%s?namespace=%s", c.baseURL, path, ns)
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
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
