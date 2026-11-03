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
	"encoding/base64"
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/onsi/gomega"
)

// 模拟 Helm Repository 的 HTTP 服务器
type MockHelmRepositoryServer struct {
	server            *http.Server
	listener          net.Listener // 用于获取实际端口
	mu                sync.RWMutex
	validUsers        map[string]string // username -> password
	port              int
	return404         bool
	returnInvalidYAML bool
	responseDelay     time.Duration // 增加响应延迟
	indexYAML         string        // 可注入的 index.yaml 内容；为空则使用默认
}

// 创建新的 Mock Helm Repository 服务器
func NewMockHelmRepositoryServer(port int) *MockHelmRepositoryServer {
	m := &MockHelmRepositoryServer{
		validUsers:        make(map[string]string),
		port:              port,
		return404:         false,
		returnInvalidYAML: false,
		responseDelay:     0,
		indexYAML:         "",
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/index.yaml", m.handleIndexYAML)
	mux.HandleFunc("/", m.handleRoot)

	m.server = &http.Server{
		Handler: mux,
	}
	return m
}

// 注入自定义 index.yaml 内容
func (m *MockHelmRepositoryServer) SetIndexYAML(index string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.indexYAML = index
	log.Printf("[MockHelmRepo] 设置自定义 index.yaml，长度=%d", len(index))
}

// 添加有效的用户名和密码
func (m *MockHelmRepositoryServer) AddValidUser(username, password string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.validUsers[username] = password
	log.Printf("[MockHelmRepo] 添加有效用户: %s", username)
}

// 清空所有用户
func (m *MockHelmRepositoryServer) ClearUsers() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.validUsers = make(map[string]string)
	log.Printf("[MockHelmRepo] 已清空所有用户")
}

func (m *MockHelmRepositoryServer) SetReturn404(return404 bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.return404 = return404
	log.Printf("[MockHelmRepo] 设置返回 404: %v", return404)
}

func (m *MockHelmRepositoryServer) SetReturnInvalidYAML(returnInvalidYAML bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.returnInvalidYAML = returnInvalidYAML
	log.Printf("[MockHelmRepo] 设置返回无效 YAML: %v", returnInvalidYAML)
}

// 设置 /index.yaml 响应的人为延迟（用于并发/竞态测试）
func (m *MockHelmRepositoryServer) SetResponseDelay(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responseDelay = d
	log.Printf("[MockHelmRepo] 设置响应延迟: %v", d)
}

func (m *MockHelmRepositoryServer) Start() error {
	if m.listener == nil {
		var listener net.Listener
		var err error
		if m.port == 0 {
			listener, err = net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				return fmt.Errorf("failed to create listener on random port: %v", err)
			}
			m.port = listener.Addr().(*net.TCPAddr).Port
		} else {
			// 尝试监听指定端口，如果失败则使用随机端口
			listener, err = net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", m.port))
			if err != nil {
				log.Printf("[MockHelmRepo] 端口 %d 被占用，使用随机端口", m.port)
				listener, err = net.Listen("tcp", "127.0.0.1:0")
				if err != nil {
					return fmt.Errorf("failed to create listener: %v", err)
				}
				m.port = listener.Addr().(*net.TCPAddr).Port
			}
		}
		m.listener = listener
		m.server.Addr = fmt.Sprintf("127.0.0.1:%d", m.port)
	}

	startErr := make(chan error, 1)
	go func() {
		log.Printf("[MockHelmRepo] 启动 Mock Helm Repository 服务器，监听端口: %d", m.port)
		if err := m.server.Serve(m.listener); err != nil && err != http.ErrServerClosed {
			log.Printf("[MockHelmRepo] 服务器启动失败: %v", err)
			startErr <- err
		}
	}()
	time.Sleep(200 * time.Millisecond)
	select {
	case err := <-startErr:
		return err
	default:
		return nil
	}
}

func (m *MockHelmRepositoryServer) Stop() error {
	log.Printf("[MockHelmRepo] 停止 Mock Helm Repository 服务器")
	if m.listener != nil {
		return m.listener.Close()
	}
	return m.server.Close()
}

func (m *MockHelmRepositoryServer) GetURL() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return fmt.Sprintf("http://127.0.0.1:%d", m.port)
}

func (m *MockHelmRepositoryServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" {
		http.Redirect(w, r, "/index.yaml", http.StatusMovedPermanently)
		return
	}
	http.NotFound(w, r)
}

// 处理 /index.yaml 请求
func (m *MockHelmRepositoryServer) handleIndexYAML(w http.ResponseWriter, r *http.Request) {
	log.Printf("[MockHelmRepo] 收到请求: %s %s", r.Method, r.URL.Path)

	m.mu.RLock()
	return404 := m.return404
	delay := m.responseDelay
	m.mu.RUnlock()

	if delay > 0 {
		time.Sleep(delay)
	}

	if return404 {
		log.Printf("[MockHelmRepo] 返回 404 Not Found")
		http.Error(w, "Not Found", http.StatusNotFound)
		return
	}

	authHeader := r.Header.Get("Authorization")
	if authHeader != "" {
		if !strings.HasPrefix(authHeader, "Basic ") {
			log.Printf("[MockHelmRepo] 无效的 Authorization 头格式")
			http.Error(w, "Invalid Authorization header", http.StatusUnauthorized)
			return
		}

		encoded := strings.TrimPrefix(authHeader, "Basic ")
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			log.Printf("[MockHelmRepo] 解码 Basic Auth 失败: %v", err)
			http.Error(w, "Invalid Authorization header", http.StatusUnauthorized)
			return
		}

		parts := strings.SplitN(string(decoded), ":", 2)
		if len(parts) != 2 {
			log.Printf("[MockHelmRepo] 无效的 Basic Auth 格式")
			http.Error(w, "Invalid Authorization header", http.StatusUnauthorized)
			return
		}

		username := parts[0]
		password := parts[1]

		m.mu.RLock()
		validPassword, exists := m.validUsers[username]
		m.mu.RUnlock()

		if !exists || validPassword != password {
			log.Printf("[MockHelmRepo] 认证失败: 用户名=%s (存在=%v)", username, exists)
			w.Header().Set("WWW-Authenticate", `Basic realm="Helm Repository"`)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		log.Printf("[MockHelmRepo] 认证成功: 用户名=%s", username)
	} else {
		m.mu.RLock()
		requiresAuth := len(m.validUsers) > 0
		m.mu.RUnlock()

		if requiresAuth {
			log.Printf("[MockHelmRepo] 需要认证但未提供认证信息")
			w.Header().Set("WWW-Authenticate", `Basic realm="Helm Repository"`)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
	}

	m.mu.RLock()
	returnInvalidYAML := m.returnInvalidYAML
	customIndex := m.indexYAML
	m.mu.RUnlock()

	var indexYAML string
	if returnInvalidYAML {
		indexYAML = `entries:
  test-chart:
  - version: 1.0.0
    name: test-chart
    description: A test chart for E2E testing
    urls:
    - http://localhost/test-chart-1.0.0.tgz
    created: "2024-01-01T00:00:00Z"
generated: "2024-01-01T00:00:00Z"
`
		log.Printf("[MockHelmRepo] 返回无效的 index.yaml（缺少 apiVersion）")
	} else if strings.TrimSpace(customIndex) != "" {
		indexYAML = customIndex
	} else {
		indexYAML = `apiVersion: v1
entries:
  test-chart:
  - version: 1.0.0
    name: test-chart
    description: A test chart for E2E testing
    urls:
    - http://localhost/test-chart-1.0.0.tgz
    created: "2024-01-01T00:00:00Z"
  - version: 1.1.0
    name: test-chart
    description: A test chart for E2E testing (updated)
    urls:
    - http://localhost/test-chart-1.1.0.tgz
    created: "2024-01-02T00:00:00Z"
generated: "2024-01-01T00:00:00Z"
`
	}

	w.Header().Set("Content-Type", "application/x-yaml")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte(indexYAML)); err != nil {
		log.Printf("[MockHelmRepo] 写入响应失败: %v", err)
	}
	log.Printf("[MockHelmRepo] 返回 index.yaml")
}

var (
	mockHelmPortMu   sync.Mutex
	nextMockHelmPort = 9050
	mockHelmServers  = map[int]*MockHelmRepositoryServer{}
)

// 启动一个新的 MockHelmRepositoryServer（动态端口），返回其 URL 和 stop 函数
func startNewMockHelmRepo() (string, func()) {
	mockHelmPortMu.Lock()
	port := nextMockHelmPort
	nextMockHelmPort++
	mockHelmPortMu.Unlock()

	s := NewMockHelmRepositoryServer(port)
	err := s.Start()
	gomega.Expect(err).To(gomega.Succeed(), fmt.Sprintf("failed to start mock helm repo on port %d", port))

	mockHelmPortMu.Lock()
	mockHelmServers[port] = s
	mockHelmPortMu.Unlock()

	stop := func() {
		mockHelmPortMu.Lock()
		defer mockHelmPortMu.Unlock()
		if ss, ok := mockHelmServers[port]; ok {
			_ = ss.Stop()
			delete(mockHelmServers, port)
		}
	}

	return s.GetURL(), stop
}
