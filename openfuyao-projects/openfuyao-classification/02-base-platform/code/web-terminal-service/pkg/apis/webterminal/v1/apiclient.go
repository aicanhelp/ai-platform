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

// Package v1 provides the necessary tools and utilities to interact with mcs api service.
// It is responsible for creating API clients, API servers, registering endpoints and handling
// requests and reponses efficiently.
package v1

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	// StatusOKMin defines the minimum allowed value for the HTTP code
	statusOKMin = 200
	// StatusOKMax defines the maximum allowed value for the HTTP code
	statusOKMax = 299
)

// APIClient includes the Restful API endpoint address for the CPE service
// and an HTTP client for making requests to teh service.
type APIClient struct {
	Client *http.Client
}

// NewAPIClient initializes and returns a new instance of APIClient with the specified configuration.
// This function sets up the HTTP client with default settings and prepares it for connection to the UM service.
func NewAPIClient() *APIClient {
	return &APIClient{
		Client: &http.Client{},
	}
}

// Get sends a GET request to the specified service endpoint and returns the response.
// It handles setting up the request, sending it, and processing the response.
func (c *APIClient) Get(path string) ([]byte, error) {
	req, err := http.NewRequest("GET", path, nil)
	if err != nil {
		return nil, err
	}

	return c.do(req)
}

// Post sends a POST request to the specified service endpoint and returns the response.
// It constructs the request with the given data, sends it to the service, and handles the response.
func (c *APIClient) Post(path string, payload interface{}, token string) ([]byte, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", path, bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return c.do(req)
}

func (c *APIClient) do(req *http.Request) ([]byte, error) {
	resp, err := c.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < statusOKMin || resp.StatusCode > statusOKMax {
		return nil, fmt.Errorf("HTTP request failed %s with status code %d", req.RequestURI, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}
