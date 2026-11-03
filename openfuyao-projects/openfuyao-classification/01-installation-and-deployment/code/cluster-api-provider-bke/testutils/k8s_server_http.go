/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package testutils

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func getDefaultMap() map[string]interface{} {
	rootReturn := &metav1.RootPaths{Paths: []string{"/api", "/apis"}}
	versionReturn := &metav1.APIVersions{Versions: []string{"v1.29.9"}}
	apiReturn := &metav1.APIVersions{Versions: []string{"v1"}}
	apisReturn := &metav1.APIGroupList{
		Groups: []metav1.APIGroup{{Name: "apps", Versions: []metav1.GroupVersionForDiscovery{
			{GroupVersion: "apps/v1", Version: "v1"}}},
		},
	}
	v1Return := &metav1.APIResourceList{
		GroupVersion: "v1", APIResources: []metav1.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
			{Name: "nodes", Namespaced: false, Kind: "Node"},
		},
	}
	appsV1Return := &metav1.APIResourceList{
		GroupVersion: "apps/v1",
		APIResources: []metav1.APIResource{{Name: "deployments", Namespaced: true, Kind: "Deployment"}},
	}
	handlerMap := map[string]interface{}{
		"/":             rootReturn,
		"/version":      versionReturn,
		"/api":          apiReturn,
		"/apis":         apisReturn,
		"/api/v1":       v1Return,
		"/apis/apps/v1": appsV1Return,
	}

	return handlerMap
}

// TestGetK8sServerHttp for kubernetes.Clientset test
func TestGetK8sServerHttp(mapObjs map[string]interface{}) (*rest.Config, *httptest.Server) {
	handlerMap := getDefaultMap()
	if mapObjs != nil && len(mapObjs) > 0 {
		for k, v := range mapObjs {
			handlerMap[k] = v
		}
	}

	httpHandlerFunc := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 根据请求路径返回不同的响应
		key := r.URL.Path
		var obj interface{}
		var ok = false
		if obj, ok = handlerMap[key]; !ok {
			fmt.Printf("[ info_k8s_http_server_request: %s not found ]\n", key)
			w.WriteHeader(http.StatusNotFound)
			if err := json.Unmarshal(
				[]byte(`{"kind":"Status","apiVersion":"v1","status":"Failure","message":"not found"}`), &obj); err != nil {
				fmt.Println(err)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		objBytes, err := json.Marshal(obj)
		if err != nil {
			fmt.Println(err)
		}

		if _, err := w.Write(objBytes); err != nil {
			fmt.Println(err)
		}
	})
	ts := httptest.NewServer(httpHandlerFunc)
	config := &rest.Config{
		Host:            ts.URL,
		TLSClientConfig: rest.TLSClientConfig{Insecure: true},
		Timeout:         1 * time.Minute, // 1分钟超时
	}
	return config, ts
}

// RestConfigToKubeConfig for rest.Config
func RestConfigToKubeConfig(config *rest.Config, contextName string) ([]byte, error) {
	// 1. 构建集群配置
	cluster := &clientcmdapi.Cluster{
		Server:                   config.Host,
		CertificateAuthorityData: config.CAData,
		InsecureSkipTLSVerify:    config.Insecure,
	}

	// 2. 构建用户认证信息
	authInfo := &clientcmdapi.AuthInfo{
		ClientCertificateData: config.CertData,
		ClientKeyData:         config.KeyData,
		Token:                 config.BearerToken,
		TokenFile:             config.BearerTokenFile,
		Username:              config.Username,
		Password:              config.Password,
		AuthProvider:          config.AuthProvider,
		Exec:                  config.ExecProvider,
	}

	// 3. 构建上下文
	context := &clientcmdapi.Context{
		Cluster:  "cluster",
		AuthInfo: "user",
	}

	// 4. 构建完整的 kubeconfig
	kubeConfig := &clientcmdapi.Config{
		APIVersion:  "v1",
		Kind:        "Config",
		Preferences: clientcmdapi.Preferences{},
		Clusters: map[string]*clientcmdapi.Cluster{
			"cluster": cluster,
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"user": authInfo,
		},
		Contexts: map[string]*clientcmdapi.Context{
			contextName: context,
		},
		CurrentContext: contextName,
	}

	// 5. 转换为 YAML 字节
	return clientcmd.Write(*kubeConfig)
}
