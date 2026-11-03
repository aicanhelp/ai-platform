/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package marketplace

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/repo"

	"gopkg.in/yaml.v3"
)

// 模拟 Harbor API，用于接收 chart 上传并动态生成 index.yaml
type MockHarborServer struct {
	server       *http.Server
	mu           sync.RWMutex
	charts       map[string][]byte 
	responseCode int              
	uploadCount  int              
	deleteCount  int              
	project      string
	requireAuth  bool
	validUsers   map[string]string
}

// 创建一个新的 Mock Harbor 服务器
func NewMockHarborServer(port int, project string) *MockHarborServer {
	m := &MockHarborServer{
		charts:       make(map[string][]byte),
		project:      project,
		responseCode: http.StatusCreated,
		validUsers:   make(map[string]string),
	}

	mux := http.NewServeMux()
	projectPath := project
	if !strings.HasPrefix(project, "/") {
		projectPath = "/chartrepo/" + project
	}
	mux.HandleFunc(fmt.Sprintf("/api%s/charts", projectPath), m.handleCharts)
	mux.HandleFunc(fmt.Sprintf("/api%s/charts/", projectPath), m.handleDeleteChart)
	mux.HandleFunc("/index.yaml", m.handleIndexYAML)

	m.server = &http.Server{
		Addr:    fmt.Sprintf("127.0.0.1:%d", port),
		Handler: mux,
	}
	return m
}

func (m *MockHarborServer) Start() error {
	ln, err := net.Listen("tcp", m.server.Addr)
	if err != nil {
		return err
	}
	m.server.Addr = ln.Addr().String()
	go func() {
		log.Printf("[MockHarbor] 启动 Mock Harbor API 服务器，监听端口: %s", m.server.Addr)
		if err := m.server.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Printf("[MockHarbor] 服务器启动失败: %v", err)
		}
	}()
	time.Sleep(100 * time.Millisecond)
	return nil
}

func (m *MockHarborServer) Stop() error {
	log.Printf("[MockHarbor] 停止 Mock Harbor API 服务器")
	return m.server.Close()
}

func (m *MockHarborServer) GetURL() string {
	_, port, err := net.SplitHostPort(m.server.Addr)
	if err != nil {
		log.Printf("[MockHarbor] GetURL 无法解析地址 '%s': %v，回退到旧模式", m.server.Addr, err)
		return fmt.Sprintf("http://localhost%s", m.server.Addr)
	}
	return fmt.Sprintf("http://127.0.0.1:%s", port)
}

func (m *MockHarborServer) SetResponseCode(code int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responseCode = code
}

func (m *MockHarborServer) handleCharts(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		m.handleListCharts(w, r)
		return
	case http.MethodPost:
		m.handleUploadChart(w, r)
		return
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
}

func (m *MockHarborServer) handleListCharts(w http.ResponseWriter, r *http.Request) {
	log.Printf("[MockHarbor] 收到 charts 列表请求: %s %s", r.Method, r.URL.Path)

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.responseCode != http.StatusCreated && m.responseCode != http.StatusOK {
		w.WriteHeader(m.responseCode)
		return
	}

	// 统计每个 chart 的版本数：通过解析 tgz 获取 chart name/version
	type chartItem struct {
		Name          string `json:"name"`
		TotalVersions int    `json:"total_versions"`
	}

	versionCount := make(map[string]int)
	for _, tgzBytes := range m.charts {
		ch, err := loader.LoadArchive(bytes.NewReader(tgzBytes))
		if err != nil {
			continue
		}
		name := ch.Name()
		if name == "" && ch.Metadata != nil {
			name = ch.Metadata.Name
		}
		if name == "" {
			continue
		}
		versionCount[name]++
	}

	items := make([]chartItem, 0, len(versionCount))
	for name, n := range versionCount {
		items = append(items, chartItem{Name: name, TotalVersions: n})
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(items)
}

func (m *MockHarborServer) UploadCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.uploadCount
}

func (m *MockHarborServer) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.charts = make(map[string][]byte)
	m.uploadCount = 0
	m.deleteCount = 0
	m.responseCode = http.StatusCreated
	m.validUsers = make(map[string]string)
	m.requireAuth = false
}

func (m *MockHarborServer) DeleteCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.deleteCount
}

func (m *MockHarborServer) AddValidUser(username, password string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.validUsers[username] = password
	m.requireAuth = true
}

// 模拟 chart 上传端点
func (m *MockHarborServer) handleUploadChart(w http.ResponseWriter, r *http.Request) {
	log.Printf("[MockHarbor] 收到上传请求: %s %s", r.Method, r.URL.Path)
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.requireAuth {
		user, pass, ok := r.BasicAuth()
		if !ok || m.validUsers[user] != pass {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
	}

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if m.responseCode != http.StatusCreated {
		w.WriteHeader(m.responseCode)
		return
	}

	file, header, err := r.FormFile("chart")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("missing 'chart' form file"))
		return
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	m.charts[header.Filename] = content
	m.uploadCount++
	w.WriteHeader(http.StatusCreated)
}

// 动态生成并提供 index.yaml
func (m *MockHarborServer) handleIndexYAML(w http.ResponseWriter, r *http.Request) {
	log.Printf("[MockHarbor] 收到 index.yaml 请求")
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.charts) == 0 {
		indexYAML := `apiVersion: v1
entries: {}
`
		w.Header().Set("Content-Type", "application/x-yaml")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(indexYAML))
		return
	}

	indexFile := repo.NewIndexFile()
	for _, tgzBytes := range m.charts {
		chart, err := loader.LoadArchive(bytes.NewReader(tgzBytes))
		if err != nil {
			log.Printf("[MockHarbor] 加载 chart archive 失败: %v", err)
			continue
		}
		indexFile.Add(chart.Metadata, chart.Name(), m.GetURL()+"/charts/"+chart.Name()+"-"+chart.Metadata.Version+".tgz", "")
	}

	indexFile.SortEntries()

	yamlBytes, err := yaml.Marshal(indexFile)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/x-yaml")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(yamlBytes)
}

// 模拟 chart 删除端点
func (m *MockHarborServer) handleDeleteChart(w http.ResponseWriter, r *http.Request) {
	log.Printf("[MockHarbor] 收到删除请求: %s %s", r.Method, r.URL.Path)
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.requireAuth {
		user, pass, ok := r.BasicAuth()
		if !ok || m.validUsers[user] != pass {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
	}

	if r.Method != http.MethodDelete {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	// 如果设置了非成功响应码，直接返回
	if m.responseCode != http.StatusCreated && m.responseCode != http.StatusOK && m.responseCode != http.StatusNoContent {
		w.WriteHeader(m.responseCode)
		return
	}

	path := r.URL.Path
	projectPath := m.project
	if !strings.HasPrefix(projectPath, "/") {
		projectPath = "/chartrepo/" + projectPath
	}
	prefix := fmt.Sprintf("/api%s/charts/", projectPath)
	if !strings.HasPrefix(path, prefix) {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	rest := strings.TrimPrefix(path, prefix)
	parts := strings.Split(rest, "/")

	if len(parts) == 0 || parts[0] == "" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	chartName := parts[0]
	version := ""
	if len(parts) > 1 {
		version = parts[1]
	}

	// 删除匹配的 chart
	deleted := false
	for filename := range m.charts {
		if strings.Contains(filename, chartName) {
			if version == "" {
				// 删除所有版本
				delete(m.charts, filename)
				deleted = true
			} else if strings.Contains(filename, version) {
				// 删除指定版本
				delete(m.charts, filename)
				deleted = true
				break
			}
		}
	}

	if !deleted {
		if m.responseCode == http.StatusNotFound {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)
		return
	}

	m.deleteCount++
	w.WriteHeader(http.StatusOK)
}
