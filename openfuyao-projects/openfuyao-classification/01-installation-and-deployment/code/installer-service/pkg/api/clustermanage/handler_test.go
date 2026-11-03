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

package clustermanage

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"io"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	"installer-service/pkg/installer"
	"installer-service/pkg/utils/httputil"
)

type handlerMock struct {
	// CreateCluster
	createClusterErr error
	// GetDefaultConfig
	defaultConfigResp installer.DefaultResp
	defaultConfigErr  error
	// JudgeClusterNode
	judgeNodeFunc func(*installer.ClusterNodeInfo) error
	// GetAllClusters
	allClustersResp installer.ClusterResponse
	allClustersErr  error
	// GetNodesByQuery
	nodesByQueryResp installer.NodeResponse
	nodesByQueryErr  error
	// GetClusterFull
	clusterFullResp installer.ClusterFullResponse
	clusterFullErr  error
	// DeleteCluster
	deleteClusterResp   *httputil.ResponseJson
	deleteClusterStatus int
	// CreateBKENodes
	createNodesErr error
	// DeleteBKENodes
	deleteNodesErr error
	// UpgradeOpenFuyao
	upgradeErr error
	// UpgradeClusterVersion
	upgradeClusterVersionErr error
	upgradeClusterName       string
	upgradeVersion           string
	upgradeClusterCalled     bool
	// UploadPatchFile
	uploadPatchErr error
	// GetOpenFuyaoVersions
	versionsResp []string
	versionsErr  error
	// GetInstallableVersions
	installableVersionsResp []string
	installableVersionsErr  error
	// GetOpenFuyaoUpgradeVersions
	upgradeVersionsResp []string
	upgradeVersionsErr  error
	// GetClustersByName
	clusterByNameResp *v1beta1.BKECluster
	clusterByNameErr  error
	// AutoUpgradePatchPrepare
	autoUpgradeErr error
	// GetClusterLog
	clusterLogResp   *httputil.ResponseJson
	clusterLogStatus int
}

func (m *handlerMock) CreateCluster(object string) error {
	return m.createClusterErr
}
func (m *handlerMock) GetDefaultConfig() (installer.DefaultResp, error) {
	return m.defaultConfigResp, m.defaultConfigErr
}
func (m *handlerMock) JudgeClusterNode(info *installer.ClusterNodeInfo) error {
	if m.judgeNodeFunc != nil {
		return m.judgeNodeFunc(info)
	}
	return nil
}
func (m *handlerMock) GetClusterLog(_ string, _ *websocket.Conn) (*httputil.ResponseJson, int) {
	return m.clusterLogResp, m.clusterLogStatus
}
func (m *handlerMock) GetClusters() ([]v1beta1.BKECluster, error) { return nil, nil }
func (m *handlerMock) GetClustersByName(_ string) (*v1beta1.BKECluster, error) {
	return m.clusterByNameResp, m.clusterByNameErr
}
func (m *handlerMock) GetClustersByQuery(_ installer.ClusterRequest) (installer.ClusterResponse, error) {
	return installer.ClusterResponse{}, nil
}
func (m *handlerMock) GetClusterFull(_ string) (installer.ClusterFullResponse, error) {
	return m.clusterFullResp, m.clusterFullErr
}
func (m *handlerMock) GetAllClusters() (installer.ClusterResponse, error) {
	return m.allClustersResp, m.allClustersErr
}
func (m *handlerMock) GetNodesByQuery(_ installer.NodeRequest) (installer.NodeResponse, error) {
	return m.nodesByQueryResp, m.nodesByQueryErr
}
func (m *handlerMock) DeleteCluster(_ string) (*httputil.ResponseJson, int) {
	return m.deleteClusterResp, m.deleteClusterStatus
}
func (m *handlerMock) GetClusterConfig(_ string) (installer.ClusterConfig, error) {
	return installer.ClusterConfig{}, nil
}
func (m *handlerMock) PatchYaml(_ string, _ bool) error          { return nil }
func (m *handlerMock) ScaleDownCluster(_ string, _ string) error { return nil }
func (m *handlerMock) CreateBKENodes(_ string, _ []installer.ClusterNode) error {
	return m.createNodesErr
}
func (m *handlerMock) DeleteBKENodes(_ string, _ []string) error {
	return m.deleteNodesErr
}
func (m *handlerMock) UploadPatchFile(_ string, _ string) error {
	return m.uploadPatchErr
}
func (m *handlerMock) GetOpenFuyaoVersions() ([]string, error) {
	return m.versionsResp, m.versionsErr
}
func (m *handlerMock) GetInstallableVersions() ([]string, error) {
	return m.installableVersionsResp, m.installableVersionsErr
}
func (m *handlerMock) GetOpenFuyaoUpgradeVersions(_ string) ([]string, error) {
	return m.upgradeVersionsResp, m.upgradeVersionsErr
}
func (m *handlerMock) GetUpgradeableVersions(_ string) ([]string, error) {
	return m.upgradeVersionsResp, m.upgradeVersionsErr
}
func (m *handlerMock) AutoUpgradePatchPrepare(_ string, _ installer.AutoUpgradeRequest) error {
	return m.autoUpgradeErr
}
func (m *handlerMock) UpgradeOpenFuyao(_ string, _ string) error {
	return m.upgradeErr
}
func (m *handlerMock) UpgradeClusterVersion(clusterName string, version string) error {
	m.upgradeClusterName = clusterName
	m.upgradeVersion = version
	m.upgradeClusterCalled = true
	if m.upgradeClusterVersionErr != nil {
		return m.upgradeClusterVersionErr
	}
	return m.upgradeErr
}
func (m *handlerMock) GetClientSet() kubernetes.Interface  { return nil }
func (m *handlerMock) GetDynamicClient() dynamic.Interface { return nil }

func newRestfulPair(method, url, body string) (*restful.Request, *restful.Response, *httptest.ResponseRecorder) {
	var httpReq *http.Request
	if body != "" {
		httpReq = httptest.NewRequest(method, url, bytes.NewBufferString(body))
	} else {
		httpReq = httptest.NewRequest(method, url, nil)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	restReq := restful.NewRequest(httpReq)
	recorder := httptest.NewRecorder()
	restResp := restful.NewResponse(recorder)
	restResp.SetRequestAccepts("application/json")
	return restReq, restResp, recorder
}

func TestHandlerCreateCluster(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "无效JSON请求",
			body:         "invalid",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "空请求体",
			body:         "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "GetDefaultConfig失败返回500",
			body:         `{"cluster":{"name":"test-cluster","openFuyaoVersion":"v1.0.0"},"nodes":[]}`,
			mock:         &handlerMock{defaultConfigErr: errors.New("config error")},
			expectStatus: http.StatusInternalServerError,
		},
		{
			name:         "BuildCreateClusterYaml失败_集群名为空返回400",
			body:         `{"cluster":{"name":"","openFuyaoVersion":"v1.0.0"},"nodes":[]}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/clusters", tt.body)
			h.createCluster(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerJudgeClusterNode(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "成功判断节点",
			body:         `{"nodeName":"node1","clusterName":"cluster1"}`,
			mock:         &handlerMock{judgeNodeFunc: func(_ *installer.ClusterNodeInfo) error { return nil }},
			expectStatus: http.StatusOK,
		},
		{
			name: "节点判断失败返回400",
			body: `{"nodeName":"node1","clusterName":"cluster1"}`,
			mock: &handlerMock{judgeNodeFunc: func(_ *installer.ClusterNodeInfo) error {
				return errors.New("node validation failed")
			}},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "空请求体返回500",
			body:         "",
			mock:         &handlerMock{},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/judge-node", tt.body)
			h.judgeClusterNode(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerListNode(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "集群名为空返回400",
			clusterName:  "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "GetNodesByQuery成功返回200",
			clusterName:  "test-cluster",
			mock:         &handlerMock{nodesByQueryResp: installer.NodeResponse{}},
			expectStatus: http.StatusOK,
		},
		{
			name:         "GetNodesByQuery失败返回500",
			clusterName:  "test-cluster",
			mock:         &handlerMock{nodesByQueryErr: errors.New("query error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("GET", "/nodes", "")
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.listNode(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerDeleteCluster(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:        "成功删除集群",
			clusterName: "del-cluster",
			mock: &handlerMock{
				deleteClusterResp:   httputil.GetDefaultSuccessResponseJson(),
				deleteClusterStatus: http.StatusOK,
			},
			expectStatus: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("DELETE", "/clusters/"+tt.clusterName, "")
			req.PathParameters()["cluster-name"] = tt.clusterName
			h.deleteCluster(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerGetDefaultConfig(t *testing.T) {
	tests := []struct {
		name         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "成功获取默认配置",
			mock:         &handlerMock{},
			expectStatus: http.StatusOK,
		},
		{
			name:         "获取默认配置失败返回500",
			mock:         &handlerMock{defaultConfigErr: errors.New("config error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("GET", "/configs", "")
			h.getDefaultConfig(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerScaleDownCluster(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		body         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "空请求体返回400",
			body:         "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "节点列表为空返回400",
			body:         `{"nodes":[]}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "DeleteBKENodes成功",
			clusterName:  "down-cluster",
			body:         `{"nodes":["node1"]}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusOK,
		},
		{
			name:         "DeleteBKENodes失败返回500",
			clusterName:  "down-cluster",
			body:         `{"nodes":["node1"]}`,
			mock:         &handlerMock{deleteNodesErr: errors.New("delete nodes failed")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/scale-down-cluster", tt.body)
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.scaleDownCluster(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerScaleUpCluster(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		body         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "空请求体返回400",
			body:         "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "节点列表为空返回400",
			body:         `{"nodes":[]}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "CreateBKENodes成功",
			clusterName:  "up-cluster",
			body:         `{"nodes":[{"name":"node1","ip":"1.2.3.4"}]}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusOK,
		},
		{
			name:         "CreateBKENodes失败返回500",
			clusterName:  "up-cluster",
			body:         `{"nodes":[{"name":"node1","ip":"1.2.3.4"}]}`,
			mock:         &handlerMock{createNodesErr: errors.New("create nodes failed")},
			expectStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/scale-up-cluster", tt.body)
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.scaleUpCluster(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerUpgradeCluster(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		body         string
		mock         *handlerMock
		expectStatus int
		expectCall   bool
	}{
		{
			name:         "无效请求体返回400",
			body:         "invalid-json",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "版本为空返回400",
			body:         `{"version":""}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "UpgradeOpenFuyao成功",
			clusterName:  "upgrade-cluster",
			body:         `{"version":"v25.09.1"}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusOK,
			expectCall:   true,
		},
		{
			name:         "UpgradeOpenFuyao失败返回500",
			clusterName:  "upgrade-cluster",
			body:         `{"version":"v25.09.1"}`,
			mock:         &handlerMock{upgradeClusterVersionErr: errors.New("upgrade failed")},
			expectStatus: http.StatusInternalServerError,
			expectCall:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/upgrade-cluster", tt.body)
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.upgradeCluster(req, resp)
			assert.Equal(t, tt.expectCall, tt.mock.upgradeClusterCalled)
			if tt.expectCall {
				assert.Equal(t, tt.clusterName, tt.mock.upgradeClusterName)
				assert.Equal(t, "v25.09.1", tt.mock.upgradeVersion)
			}
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerGetOpenFuyaoVersions(t *testing.T) {
	tests := []struct {
		name         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "成功获取版本列表",
			mock:         &handlerMock{versionsResp: []string{"v1.0", "v2.0"}},
			expectStatus: http.StatusOK,
		},
		{
			name:         "获取版本失败返回500",
			mock:         &handlerMock{versionsErr: errors.New("versions error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("GET", "/versions", "")
			h.getOpenFuyaoVersions(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
			if tt.expectStatus == http.StatusOK {
				var body httputil.ResponseJson
				err := json.Unmarshal(rec.Body.Bytes(), &body)
				assert.NoError(t, err)
				assert.NotNil(t, body.Data)
			}
		})
	}
}

func TestHandlerGetInstallableVersions(t *testing.T) {
	tests := []struct {
		name         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "成功获取可安装版本列表",
			mock:         &handlerMock{installableVersionsResp: []string{"v25.09", "v25.12"}},
			expectStatus: http.StatusOK,
		},
		{
			name:         "获取可安装版本失败返回500",
			mock:         &handlerMock{installableVersionsErr: errors.New("versions error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("GET", "/versions", "")
			h.getInstallableVersions(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
			if tt.expectStatus == http.StatusOK {
				var body httputil.ResponseJson
				err := json.Unmarshal(rec.Body.Bytes(), &body)
				assert.NoError(t, err)
				assert.NotNil(t, body.Data)
			}
		})
	}
}

func TestHandlerGetUpgradeOpenFuyaoVersions(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		queryVersion string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:        "通过集群资源获取当前版本_成功",
			clusterName: "uv-cluster",
			mock: &handlerMock{
				clusterByNameResp: &v1beta1.BKECluster{
					Spec: v1beta1.BKEClusterSpec{
						ClusterConfig: &v1beta1.BKEConfig{
							Cluster: v1beta1.Cluster{OpenFuyaoVersion: "v1.0"},
						},
					},
				},
				upgradeVersionsResp: []string{"v2.0"},
			},
			expectStatus: http.StatusOK,
		},
		{
			name:         "集群不存在且无currentVersion返回500",
			clusterName:  "missing-cluster",
			mock:         &handlerMock{clusterByNameErr: errors.New("not found")},
			expectStatus: http.StatusInternalServerError,
		},
		{
			name:         "集群不存在但有currentVersion参数_成功",
			clusterName:  "missing-cluster",
			queryVersion: "v1.0",
			mock: &handlerMock{
				clusterByNameErr:    errors.New("not found"),
				upgradeVersionsResp: []string{"v2.0"},
			},
			expectStatus: http.StatusOK,
		},
		{
			name: "无集群名_从查询参数获取currentVersion_成功",
			mock: &handlerMock{
				upgradeVersionsResp: []string{"v2.0"},
			},
			queryVersion: "v1.0",
			expectStatus: http.StatusOK,
		},
		{
			name:        "GetOpenFuyaoUpgradeVersions失败返回500",
			clusterName: "uv-cluster",
			mock: &handlerMock{
				clusterByNameResp: &v1beta1.BKECluster{
					Spec: v1beta1.BKEClusterSpec{
						ClusterConfig: &v1beta1.BKEConfig{
							Cluster: v1beta1.Cluster{OpenFuyaoVersion: "v1.0"},
						},
					},
				},
				upgradeVersionsErr: errors.New("upgrade versions error"),
			},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			url := "/clusters/upgrade-versions"
			if tt.queryVersion != "" {
				url += "?currentVersion=" + tt.queryVersion
			}
			req, resp, rec := newRestfulPair("GET", url, "")
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.getUpgradeOpenFuyaoVersions(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

func TestHandlerAutoUpgrade(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		body         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "空请求体返回400",
			body:         "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "AutoUpgradePatchPrepare成功",
			clusterName:  "auto-cluster",
			body:         `{"patchDir":"/tmp/patch","patchName":"v1"}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusOK,
		},
		{
			name:         "AutoUpgradePatchPrepare失败返回500",
			clusterName:  "auto-cluster",
			body:         `{"patchDir":"/tmp/patch","patchName":"v1"}`,
			mock:         &handlerMock{autoUpgradeErr: errors.New("auto upgrade failed")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/auto-upgrade", tt.body)
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.autoUpgrade(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code, "HTTP状态码不匹配")
		})
	}
}

// wsWriteMock embeds handlerMock and overrides GetClusterLog to write to the websocket
type wsWriteMock struct{ handlerMock }

func (m *wsWriteMock) GetClusterLog(_ string, conn *websocket.Conn) (*httputil.ResponseJson, int) {
	_ = conn.WriteMessage(websocket.TextMessage, []byte("hello"))
	return httputil.GetDefaultSuccessResponseJson(), http.StatusOK
}

func TestHandlerGetClusterLog(t *testing.T) {
	tests := []struct {
		name  string
		useWs bool
	}{
		{name: "upgrade success", useWs: true},
		{name: "upgrade fail", useWs: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var mock installer.Operation
			if tt.useWs {
				mock = &wsWriteMock{}
			} else {
				mock = &handlerMock{}
			}

			h := &Handler{installerHandler: mock}

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				restReq := restful.NewRequest(r)
				restResp := restful.NewResponse(w)
				restResp.SetRequestAccepts("application/json")
				parts := strings.Split(r.URL.Path, "/")
				if len(parts) >= 3 {
					restReq.PathParameters()["cluster-name"] = parts[2]
				}
				h.getClusterLog(restReq, restResp)
			}))
			defer srv.Close()

			if tt.useWs {
				wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/clusters/test/logs"
				header := http.Header{}
				header.Set("Origin", srv.URL) // 同 Host，模拟 Console 经 Ingress 同域访问
				conn, resp, err := websocket.DefaultDialer.Dial(wsURL, header)
				require.NoError(t, err)
				require.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode)
				_, msg, err := conn.ReadMessage()
				require.NoError(t, err)
				assert.Equal(t, "hello", string(msg))
				_ = conn.Close()
			} else {
				resp, err := http.Get(srv.URL + "/clusters/test/logs")
				require.NoError(t, err)
				defer resp.Body.Close()
				body, _ := io.ReadAll(resp.Body)
				// depending on environment/restful internals the handler may produce 400 or 500
				assert.True(t, resp.StatusCode == http.StatusInternalServerError || resp.StatusCode == http.StatusBadRequest)
				assert.True(t, strings.Contains(string(body), "Internal Server Error") || strings.Contains(string(body), "Failed to upgrade"))
			}
		})
	}
}

func TestHandlerListClusterGet(t *testing.T) {
	tests := []struct {
		name         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "成功获取集群列表",
			mock:         &handlerMock{allClustersResp: installer.ClusterResponse{Items: []*installer.ClusterData{{Name: "c1"}}}},
			expectStatus: http.StatusOK,
		},
		{
			name:         "GetAllClusters失败返回500",
			mock:         &handlerMock{allClustersErr: errors.New("list error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("GET", "/clusters", "")
			h.listClusterGet(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code)
		})
	}
}

func TestHandlerGetClusterFull(t *testing.T) {
	tests := []struct {
		name         string
		clusterName  string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "集群名为空返回400",
			clusterName:  "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "成功获取集群详情",
			clusterName:  "detail-cluster",
			mock:         &handlerMock{clusterFullResp: installer.ClusterFullResponse{ClusterConfig: installer.ClusterConfig{ClusterName: "detail-cluster"}}},
			expectStatus: http.StatusOK,
		},
		{
			name:         "GetClusterFull失败返回500",
			clusterName:  "detail-cluster",
			mock:         &handlerMock{clusterFullErr: errors.New("detail error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("GET", "/clusters/"+tt.clusterName, "")
			if tt.clusterName != "" {
				req.PathParameters()["cluster-name"] = tt.clusterName
			}
			h.getClusterFull(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code)
		})
	}
}

func TestHandlerUploadPatchFile(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		mock         *handlerMock
		expectStatus int
	}{
		{
			name:         "无效JSON返回400",
			body:         "invalid",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "空请求体返回400",
			body:         "",
			mock:         &handlerMock{},
			expectStatus: http.StatusBadRequest,
		},
		{
			name:         "上传成功返回200",
			body:         `{"PatchFileName":"v1.yaml","PatchFileContent":"content"}`,
			mock:         &handlerMock{},
			expectStatus: http.StatusOK,
		},
		{
			name:         "上传失败返回500",
			body:         `{"PatchFileName":"v1.yaml","PatchFileContent":"content"}`,
			mock:         &handlerMock{uploadPatchErr: errors.New("upload error")},
			expectStatus: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &Handler{installerHandler: tt.mock}
			req, resp, rec := newRestfulPair("POST", "/patches", tt.body)
			h.uploadPatchFile(req, resp)
			assert.Equal(t, tt.expectStatus, rec.Code)
		})
	}
}
