/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package helm

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"reflect"
	"testing"
	"time"

	"github.com/jarcoal/httpmock"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/repo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/zlog"
)

const (
	mockChartNginxName      = "nginx-all-type"
	mockChartNginxVersion   = "0.1.0"
	mockChartDemoName       = "DemoApp"
	mockChartDemoAPPVersion = "1.0.0"
)

var (
	harborURL  = "/api/chartrepo/library/charts"
	nginxURL   = harborURL + "/" + mockChartNginxName
	demoAPPURL = harborURL + "/" + mockChartDemoName
)

// mockTarGzInMemory creates a .tar.gz file in memory from the specified files
func mockTarGzInMemory(files map[string][]byte) (*bytes.Buffer, error) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	defer gw.Close()

	tw := tar.NewWriter(gw)
	defer tw.Close()

	for filename, content := range files {
		header := &tar.Header{
			Name: filename,
			Size: int64(len(content)),
			Mode: 0600,
		}

		err := tw.WriteHeader(header)
		if err != nil {
			return nil, err
		}

		_, err = tw.Write(content)
		if err != nil {
			return nil, err
		}
	}
	err := tw.Close() // 关闭 tar writer
	if err != nil {
		return nil, err
	}

	err = gw.Close() // 关闭 gzip writer
	if err != nil {
		return nil, err
	}

	return &buf, nil
}

// inMemoryFile implements multipart.File interface
type inMemoryFile struct {
	*bytes.Reader
}

func (f *inMemoryFile) Close() error {
	return nil
}

func mockHarborSecret() *v1.Secret {

	return &v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "v1",
			APIVersion: "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      localHarborSecret,
			Namespace: constant.MarketplaceServiceDefaultNamespace,
		},
		StringData: map[string]string{
			localHarborSecretMapKey: "some-value",
		},
		Type: "Opaque",
	}
}

func mockRepoIndex() *repo.IndexFile {
	return &repo.IndexFile{
		ServerInfo:  nil,
		APIVersion:  "v1",
		Generated:   time.Now(),
		Entries:     mockRepoIndexEntires(),
		PublicKeys:  nil,
		Annotations: nil,
	}
}

func mockRepoIndexEntires() map[string]repo.ChartVersions {
	// 创建一个 map[string]repo.ChartVersions
	chartVersions := make(map[string]repo.ChartVersions)

	// 添加一些示例数据
	chartVersions[mockChartNginxName] = repo.ChartVersions{

		{
			Metadata: &chart.Metadata{
				Name:         mockChartNginxName,
				Home:         "",
				Sources:      nil,
				Version:      mockChartNginxVersion,
				Description:  "",
				Keywords:     nil,
				Maintainers:  nil,
				Icon:         "",
				APIVersion:   "",
				Condition:    "",
				Tags:         "",
				AppVersion:   "",
				Deprecated:   false,
				Annotations:  nil,
				KubeVersion:  "",
				Dependencies: nil,
				Type:         "",
			},
			URLs:                    []string{"charts/demo-app-1.0.0.tgz"},
			Created:                 time.Time{},
			Removed:                 false,
			Digest:                  "",
			ChecksumDeprecated:      "",
			EngineDeprecated:        "",
			TillerVersionDeprecated: "",
			URLDeprecated:           ""},
	}
	chartVersions[mockChartDemoName] = repo.ChartVersions{
		{
			Metadata: &chart.Metadata{
				Name:         mockChartDemoName,
				Home:         "",
				Sources:      nil,
				Version:      mockChartDemoAPPVersion,
				Description:  "",
				Keywords:     nil,
				Maintainers:  nil,
				Icon:         "",
				APIVersion:   "",
				Condition:    "",
				Tags:         "",
				AppVersion:   "",
				Deprecated:   false,
				Annotations:  nil,
				KubeVersion:  "",
				Dependencies: nil,
				Type:         "",
			},
			URLs:                    []string{"charts/nginx-all-type-0.1.0.tgz"},
			Created:                 time.Time{},
			Removed:                 false,
			Digest:                  "",
			ChecksumDeprecated:      "",
			EngineDeprecated:        "",
			TillerVersionDeprecated: "",
			URLDeprecated:           "",
		},
	}
	return chartVersions
}

func AddSecrectInMockedClientset(clientset kubernetes.Interface, secret *v1.Secret) {
	if secret.StringData != nil {
		if secret.Data == nil {
			secret.Data = make(map[string][]byte)
		}
		for key, value := range secret.StringData {
			secret.Data[key] = []byte(value)
		}
		// 清空 StringData 以避免重复转换
		secret.StringData = nil
	}
	_, err := clientset.CoreV1().Secrets(constant.MarketplaceServiceDefaultNamespace).Create(context.TODO(), secret, metav1.CreateOptions{})
	if err != nil {
		zlog.Errorln(err)
	}
}

// mockMultipartFileInMemory creates a multipart.File from an in-memory .tar.gz buffer
func mockMultipartFileInMemory(buffer *bytes.Buffer, fileName string) (multipart.File, *multipart.FileHeader, error) {
	var multipartBuffer bytes.Buffer
	fieldName := "chart"
	writer := multipart.NewWriter(&multipartBuffer)

	// 手动创建一个 Part，并设置头信息
	partHeaders := textproto.MIMEHeader{}
	partHeaders.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, fieldName, fileName))
	partHeaders.Set("Content-Type", "application/octet-stream")

	part, err := writer.CreatePart(partHeaders)
	if err != nil {
		return nil, nil, err
	}

	// 将缓冲区中的数据复制到 Part
	_, err = io.Copy(part, buffer)
	if err != nil {
		return nil, nil, err
	}

	err = writer.Close()
	if err != nil {
		return nil, nil, err
	}

	reader := multipart.NewReader(&multipartBuffer, writer.Boundary())
	form, err := reader.ReadForm(int64(multipartBuffer.Len()))
	if err != nil {
		return nil, nil, err
	}

	fileHeaders := form.File[fieldName]
	if len(fileHeaders) == 0 {
		return nil, nil, fmt.Errorf("未找到文件字段：%s", fieldName)
	}
	fileHeader := fileHeaders[0]

	file, err := fileHeader.Open()
	if err != nil {
		return nil, nil, err
	}

	return file, fileHeader, nil
}

func getNginxMultipartFile() (multipart.File, *multipart.FileHeader) {
	files := map[string][]byte{
		"Chart.yaml": []byte(mockNginxChartYaml()),
	}
	// Create a .tar.gz in memory
	tarGzBuffer, err := mockTarGzInMemory(files)
	if err != nil {
		zlog.Errorln("Error creating .tar.gz in memory:", err)
		return nil, nil
	}

	// Create a multipart.File from the in-memory .tar.gz buffer
	multipartFile, fileHeader, err := mockMultipartFileInMemory(tarGzBuffer, "output.tar.gz")
	if err != nil {
		fmt.Println("Error creating multipart.File in memory:", err)
		return nil, nil
	}
	return multipartFile, fileHeader
}

func patchMockedDynamicClientRepoCRULR(dynamicClient dynamic.Interface, name, URL string) {
	_, err := dynamicClient.Resource(gvr).Namespace("").Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
		return
	}
	patch := []map[string]interface{}{
		{
			"op":    "replace",
			"path":  "/spec/url",
			"value": URL,
		},
	}

	patchBytes, err := json.Marshal(patch)
	if err != nil {
		zlog.Fatalf("Error marshalling patch: %s", err.Error())
	}

	_, err = dynamicClient.Resource(gvr).Namespace("").Patch(context.TODO(), name, types.JSONPatchType, patchBytes, metav1.PatchOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
	}
	cr, err := dynamicClient.Resource(gvr).Namespace("").Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
	}
	zlog.Infof("%v", cr)
}

func patchMockedClientsetConfigMapWithURL(clientset kubernetes.Interface, name, URL string) {
	configMap, err := clientset.CoreV1().ConfigMaps(constant.MarketplaceServiceDefaultNamespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
	}
	// 更新 ConfigMap 的数据字段
	configMap.Data["local-harbor-host"] = URL

	// 应用更新
	_, err = clientset.CoreV1().ConfigMaps(constant.MarketplaceServiceDefaultNamespace).Update(context.TODO(), configMap, metav1.UpdateOptions{})
	if err != nil {
		zlog.Errorln(err.Error())
	}
	if err != nil {
		zlog.Fatalf("failed to update resource: %v", err)
	}
}

func mockLocalHarbor() *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			handleGet(w, r)
		case http.MethodPost:
			handlePost(w, r)
		case http.MethodDelete:
			handleDelete(w, r)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	return server
}

func SetupMockHTTP(method string, url string, responseBody string, statusCode int) {
	httpmock.Activate()

	httpmock.RegisterResponder(method, url,
		httpmock.NewStringResponder(statusCode, responseBody))
}

func TearDownMockHTTP() {
	httpmock.DeactivateAndReset()
}

func SetupOfficialHarborPageMocking() {
	SetupMockHTTP("GET", "https://helm.openfuyao.cn/_core/index.yaml",
		mockIndexResponseFromOfficialHarbor(), 200)

}

func getTestCountChartRestfulResponse() string {
	resp := "[{\"name\":\"Boostkit-hbase-demo\",\"total_versions\":1,\"latest_version\":\"0.1.0\",\"created\":\"2024-09-30T08:13:42.575853615Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"Boostkit-mysql-demo\",\"total_versions\":1,\"latest_version\":\"0.1.0\",\"created\":\"2024-10-08T03:14:36.567671086Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"Boostkit-omni-spark-demo\",\"total_versions\":1,\"latest_version\":\"0.1.0\",\"created\":\"2024-09-30T03:36:33.623317125Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"Boostkit-spark-demo\",\"total_versions\":1,\"latest_version\":\"0.1.0\",\"created\":\"2024-09-30T03:36:39.602061262Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"colocation-agent\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-09-25T10:24:01.408554556Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"colocation-helm\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-09-29T09:54:14.702506372Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"colocation-operator\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-09-30T02:48:18.428435146Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"colocation-package\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-17T08:16:33.735940567Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"colocation-service\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-16T11:47:26.922227865Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"computing-power-engine\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-16T03:16:41.065288662Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"kae-device-plugin\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-11T09:09:50.511189119Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"kae-operator\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-09-29T02:42:55.414063722Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"kae-optimization\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-10T07:22:00.277176797Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"logging-package\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-16T20:53:01.522916435Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"monitoring-dashboard\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-15T08:40:02.360388762Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"multi-cluster-service\",\"total_versions\":2,\"latest_version\":\"0.0.0-openFuyao-v24.09-poc\",\"created\":\"2024-10-15T03:16:29.159946679Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"nginx\",\"total_versions\":1,\"latest_version\":\"0.2.0\",\"created\":\"2024-09-03T14:39:56.378583271Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"npu-operator\",\"total_versions\":1,\"latest_version\":\"0.1.0\",\"created\":\"2024-10-17T02:58:42.018309093Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"volcano-config-package\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-09-30T02:18:45.010733509Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"volcano-config-service\",\"total_versions\":2,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-12T06:03:46.303076696Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false},{\"name\":\"volcano-config-service-helm\",\"total_versions\":1,\"latest_version\":\"0.0.0-latest\",\"created\":\"2024-10-10T02:01:14.803465276Z\",\"updated\":\"0001-01-01T00:00:00Z\",\"icon\":\"\",\"home\":\"\",\"deprecated\":false}]"
	return resp
}

func handleGet(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == harborURL {
		w.Header().Set("Content-Type", "application/x-yaml")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(getTestCountChartRestfulResponse()))
	} else if r.URL.Path == nginxURL || r.URL.Path == demoAPPURL {
		w.Header().Set("Content-Type", "application/x-yaml")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("123"))
	} else {
		w.WriteHeader(http.StatusNotFound)
	}
}

func handlePost(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == harborURL {
		w.Header().Set("Content-Type", "application/x-yaml")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte("POST response"))
	} else {
		w.WriteHeader(http.StatusNotFound)
	}
}

func handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == nginxURL || r.URL.Path == demoAPPURL {
		w.Header().Set("Content-Type", "application/x-yaml")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("DELETE response"))
	} else {
		w.WriteHeader(http.StatusNotFound)
	}
}

func mockNginxChartYaml() string {
	return `
apiVersion: v2
appVersion: 1.16.0
description: A Helm chart for Kubernetes
name: nginx-all-type
type: application
version: 0.1.0
keywords: [openfuyao-extension,fuyao-select,compute-power-engine-plugin,artificial-intelligence,computing,database,developer-tool,CI/CD,monitor,log,network,observability,security,storage]
`
}

func mockDemoAppChartYaml() string {
	return `
apiVersion: v2
appVersion: 1.16.0
description: A Helm chart for Kubernetes
name: DemoApp
type: application
version: 0.1.0
keywords: [openfuyao-extension]
`
}

func Test_customCompare(t *testing.T) {
	type args struct {
		a     string
		b     string
		order int
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "Test_customCompare_success",
			args: args{
				a:     "abc",
				b:     "Abc",
				order: 1,
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := customCompare(tt.args.a, tt.args.b, tt.args.order); got != tt.want {
				t.Errorf("customCompare() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_helmClient_DeleteChart(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		chartName string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_DeleteChart_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockLocalRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				chartName: mockChartNginxName,
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "delete success",
				Data: nil,
			},
			want1: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				clientset:     tt.fields.clientset,
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
			}
			server := mockLocalHarbor()
			patchMockedClientsetConfigMapWithURL(tt.fields.clientset, constant.MarketplaceServiceConfigmap, server.URL)
			cachedData.SetChartCache("local", mockRepoIndex())
			got, got1 := c.DeleteChart(tt.args.chartName)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("DeleteChart() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("DeleteChart() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_GetChartFiles(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		repoName  string
		chartName string
		version   string
		fileType  string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_GetChartFiles_failed",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				repoName:  mockRemoteRepoCRName,
				chartName: mockChartNginxName,
				version:   mockChartNginxVersion,
				fileType:  fileTypeAll,
			},
			want: &httputil.ResponseJson{
				Code: constant.ServerError,
				Msg:  "not found chart nginx-all-type with version 0.1.0 in repo some_random_name, please check input or sync repository",
				Data: nil,
			},
			want1: http.StatusInternalServerError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				dynamicClient: tt.fields.dynamicClient,
				kubeConfig:    tt.fields.kubeConfig,
				clientset:     tt.fields.clientset,
			}
			server := mockLocalHarbor()
			cachedData.SetChartCache(mockRemoteRepoCRName, mockRepoIndex())
			patchMockedDynamicClientRepoCRULR(tt.fields.dynamicClient, mockRemoteRepoCRName, server.URL)
			got, got1 := c.GetChartFiles(tt.args.repoName, tt.args.chartName, tt.args.version, tt.args.fileType)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetChartFiles() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("GetChartFiles() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func Test_helmClient_GetLatestCharts(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		searchParam *helm.ChartSearchParam
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_GetLatestCharts_normal",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				searchParam: &helm.ChartSearchParam{
					Repositories: nil,
					Chart:        "",
					Types:        []string{"application"},
					Query: &param.Query{
						Pagination: param.GetNoPagination(),
						SortBy:     param.Name,
						Ascending:  false,
					},
					Scene: []string{"application"},
				},
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: &helm.ListResponse{
					TotalItems:  0,
					Items:       []interface{}{},
					CurrentPage: 1,
					TotalPages:  1,
				},
			},
			want1: http.StatusOK,
		},
		{
			name: "Test_helmClient_GetLatestCharts_sort_time",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				searchParam: &helm.ChartSearchParam{
					Repositories: nil,
					Chart:        "",
					Types:        []string{"application", "openfuyao-extension", "test"},
					Query: &param.Query{
						Pagination: param.GetNoPagination(),
						SortBy:     param.Time,
						Ascending:  false,
					},
					Scene: []string{"monitor"},
				},
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: &helm.ListResponse{
					TotalItems:  0,
					Items:       []interface{}{},
					CurrentPage: 1,
					TotalPages:  1,
				},
			},
			want1: http.StatusOK,
		},
		{
			name: "Test_helmClient_GetLatestCharts_sort_name",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockRemoteRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				searchParam: &helm.ChartSearchParam{
					Repositories: nil,
					Chart:        "",
					Types:        []string{"application"},
					Query: &param.Query{
						Pagination: param.GetNoPagination(),
						SortBy:     param.Name,
						Ascending:  false,
					},
					Scene: []string{"monitor"},
				},
			},
			want: &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  "success",
				Data: &helm.ListResponse{
					TotalItems:  0,
					Items:       []interface{}{},
					CurrentPage: 1,
					TotalPages:  1,
				},
			},
			want1: http.StatusOK,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &helmClient{
				kubeConfig:    tt.fields.kubeConfig,
				dynamicClient: tt.fields.dynamicClient,
				clientset:     tt.fields.clientset,
			}
			server := mockLocalHarbor()
			patchMockedClientsetConfigMapWithURL(tt.fields.clientset, constant.MarketplaceServiceConfigmap, server.URL)
			got, got1 := c.GetLatestCharts(tt.args.searchParam)
			if !reflect.DeepEqual(got, tt.want) {
			}
			if got1 != tt.want1 {
			}
		})
	}
}

func Test_helmClient_UploadChart(t *testing.T) {
	type fields struct {
		kubeConfig    *rest.Config
		dynamicClient dynamic.Interface
		clientset     kubernetes.Interface
	}
	type args struct {
		formFile   multipart.File
		fileHeader *multipart.FileHeader
	}
	nginxMultipartFile, header := getNginxMultipartFile()
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *httputil.ResponseJson
		want1  int
	}{
		{
			name: "Test_helmClient_UploadChart_success",
			fields: fields{
				kubeConfig:    nil,
				dynamicClient: mockDynamicClientWithRepoCR(mockLocalRepoCR()),
				clientset:     mockClientset(),
			},
			args: args{
				formFile:   nginxMultipartFile,
				fileHeader: header,
			},
			want: &httputil.ResponseJson{
				Code: constant.FileCreated,
				Msg:  "upload success",
				Data: nil,
			},
			want1: http.StatusCreated,
		},
	}
	for _, ttt := range tests {
		t.Run(ttt.name, func(t *testing.T) {
			c := &helmClient{
				dynamicClient: ttt.fields.dynamicClient,
				kubeConfig:    ttt.fields.kubeConfig,
				clientset:     ttt.fields.clientset,
			}
			server := mockLocalHarbor()
			patchMockedClientsetConfigMapWithURL(ttt.fields.clientset, constant.MarketplaceServiceConfigmap, server.URL)
			cachedData.SetChartCache("local", mockRepoIndex())
			got, got1 := c.UploadChart(ttt.args.formFile, ttt.args.fileHeader)
			if !reflect.DeepEqual(got, ttt.want) {
				t.Errorf("UploadChart() got = %v, want %v", got, ttt.want)
			}
			if got1 != ttt.want1 {
				t.Errorf("UploadChart() got1 = %v, want %v", got1, ttt.want1)
			}
		})
	}
}

func Test_helmClient_GetChartsWithOfficialTags(t *testing.T) {
	setupClient := func() *helmClient {
		return &helmClient{
			dynamicClient: mockDynamicClientWithRepoCR(mockOfficialRepoCR()),
			clientset:     mockClientset(),
		}
	}

	tests := []struct {
		name  string
		tags  []string
		query *param.Query
		want  *httputil.ResponseJson
		want1 int
	}{
		{
			name:  "Test_helmClient_GetChartsWithOfficialTags_success",
			tags:  []string{"openfuyao-premium"},
			query: &param.Query{Pagination: &param.Pagination{}},
			want: &httputil.ResponseJson{
				Code: 200,
				Msg:  "success",
				Data: []*helm.ChartVersionResponseWithTag{{Tag: "openfuyao-premium"}},
			},
			want1: 200,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := setupClient()
			SetupOfficialHarborPageMocking()
			defer TearDownMockHTTP()
			cachedData.SetChartCache("openfuyao", mockRepoIndex())

			got, got1 := c.GetChartsWithOfficialTags(tt.tags, tt.query)

			if got.Code != tt.want.Code || got.Msg != tt.want.Msg {
				t.Fatalf("GetChartsWithOfficialTags() got = %+v, want %+v", got, tt.want)
			}

			data, ok := got.Data.([]*helm.ChartVersionResponseWithTag)
			if !ok || len(data) == 0 {
				t.Fatal("Failed to convert got.Data to []*helm.ChartVersionResponseWithTag or empty data")
			}

			wantData, ok := tt.want.Data.([]*helm.ChartVersionResponseWithTag)
			if !ok || len(wantData) == 0 {
				t.Fatal("Failed to convert tt.want.Data to []*helm.ChartVersionResponseWithTag or empty wantData")
			}

			if data[0].Tag != wantData[0].Tag ||
				data[0].ChartVersionResponses[0].Metadata.Name != "logging-package" {
				t.Errorf("GetChartsWithOfficialTags() data mismatch")
			}

			if got1 != tt.want1 {
				t.Errorf("GetChartsWithOfficialTags() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
