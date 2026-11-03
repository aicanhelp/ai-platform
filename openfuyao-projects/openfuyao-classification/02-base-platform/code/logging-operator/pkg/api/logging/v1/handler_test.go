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

package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

const ResponseStatusErrCode = 400

func TestInitializeRequests(t *testing.T) {
	h := &Handler{
		RequestsMap: make(map[string]*HTTPRequest),
	}

	initializeRequests(h)

	// 定义期望的请求细节，例如对特定 URL 的检查
	expectedURLs := map[string]string{
		"getFileNameList":  "/loki/api/v1/label/filename/values",
		"filterRequest":    "/loki/api/v1/query_range",
		"getResourceList":  "/loki/api/v1/series",
		"getConfigmapList": "/api/kubernetes/api/v1/namespaces/loki/configmaps/loki-promtail",
		"getPodList":       "/api/kubernetes/api/v1/namespaces/loki/pods",
		"getPod":           "/api/kubernetes/api/v1/namespaces/loki/pods/",
		"patchConfigMap":   "/api/kubernetes/api/v1/namespaces/loki/configmaps/loki-promtail",
		"getAlertingRules": "/loki/api/v1/rules",
	}

	for key, expectedURL := range expectedURLs {
		req, exists := h.RequestsMap[key]
		if !exists {
			t.Errorf("Request key %s not initialized", key)
		} else {
			if req.URL != expectedURL {
				t.Errorf("Expected URL %s for key %s, got %s", expectedURL, key, req.URL)
			}
			if key == "patchConfigMap" && req.Method != "PATCH" {
				t.Errorf("Expected PATCH method for 'patchConfigMap', got %s", req.Method)
			}
		}
	}

	// 检查 PATCH 请求的特殊内容类型
	patchRequest, exists := h.RequestsMap["patchConfigMap"]
	if exists {
		expectedContentType := "application/merge-patch+json"
		if patchRequest.Headers[ContentType] != expectedContentType {
			t.Errorf("Expected content type %s for 'patchConfigMap', "+
				"got %s", expectedContentType, patchRequest.Headers[ContentType])
		}
	}
}

func TestSendHTTPRequestResourceOnly(t *testing.T) {
	// Mock server to simulate HTTP response
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		response := map[string]interface{}{
			"status": "success",
			"data":   "mock data",
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "failed to encode response", http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	// Create HTTPRequest object
	req := &HTTPRequest{
		Method: "GET",
		Body:   nil,
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
	}

	// Call the function to be tested
	result, err := sendHTTPRequestResourceOnly(req, server.URL)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Verify the response
	expected := map[string]interface{}{
		"status": "success",
		"data":   "mock data",
	}
	if !equal(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}
func equal(a, b map[string]interface{}) bool {
	for key, value := range a {
		if b[key] != value {
			return false
		}
	}
	for key, value := range b {
		if a[key] != value {
			return false
		}
	}
	return true
}

func TestInitializeTimeUnits(t *testing.T) {
	expected := map[string]string{
		"5min":   "5m",
		"15min":  "15m",
		"30min":  "30m",
		"1hour":  "1h",
		"2hour":  "2h",
		"6hour":  "6h",
		"12hour": "12h",
		"1day":   "24h",
		"2day":   "48h",
		"3day":   "72h",
		"7day":   "168h",
	}

	result := initializeTimeUnits()

	if !reflect.DeepEqual(result, expected) {
		t.Errorf("initializeTimeUnits() = %v; want %v", result, expected)
	}
}

// Mock implementation of the LogError function to capture log messages during testing
var logMessages []string

func LogError(message string) {
	logMessages = append(logMessages, message)
}

// Test framework and unit test cases
func TestCalculateQueryTimesUnix(t *testing.T) {
	testCases := []struct {
		duration    string
		expectStart int64
		expectEnd   int64
		expectLog   string
	}{
		{"1h", time.Now().Add(-1 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"2h30m", time.Now().Add(-2*time.Hour - 30*time.Minute).Unix(), time.Now().Unix(), ""},
		{"45m", time.Now().Add(-45 * time.Minute).Unix(), time.Now().Unix(), ""},
		{"300ms", time.Now().Add(-300 * time.Millisecond).Unix(), time.Now().Unix(), ""},
		{"5m", time.Now().Add(-5 * time.Minute).Unix(), time.Now().Unix(), ""},
		{"15m", time.Now().Add(-15 * time.Minute).Unix(), time.Now().Unix(), ""},
		{"30m", time.Now().Add(-30 * time.Minute).Unix(), time.Now().Unix(), ""},
		{"1h", time.Now().Add(-1 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"2h", time.Now().Add(-2 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"6h", time.Now().Add(-6 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"12h", time.Now().Add(-12 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"24h", time.Now().Add(-24 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"48h", time.Now().Add(-48 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"72h", time.Now().Add(-72 * time.Hour).Unix(), time.Now().Unix(), ""},
		{"500h", time.Now().Add(-500 * time.Hour).Unix(), time.Now().Unix(), ""},
	}

	for _, tc := range testCases {
		logMessages = []string{} // Reset log messages before each test
		start, end := calculateQueryTimesUnix(tc.duration)

		if tc.expectLog != "" {
			if len(logMessages) == 0 || logMessages[0] != tc.expectLog {
				t.Errorf("calculateQueryTimesUnix(%s) expected log message '%s', got '%s'", tc.duration, tc.expectLog, logMessages)
			}
		} else {
			if start != fmt.Sprintf("%d", tc.expectStart) {
				t.Errorf("calculateQueryTimesUnix(%s) expected start time '%d', got '%s'", tc.duration, tc.expectStart, start)
			}
			if end != fmt.Sprintf("%d", tc.expectEnd) {
				t.Errorf("calculateQueryTimesUnix(%s) expected end time '%d', got '%s'", tc.duration, tc.expectEnd, end)
			}
		}
	}
}

func TestQueryParamBuilder(t *testing.T) {
	testCases := []struct {
		startTime    string
		endTime      string
		sizes        string
		query        string
		expectParams url.Values
	}{
		{
			"1622505600", "1622592000", "10", "select * from table",
			url.Values{"start": {"1622505600"}, "end": {"1622592000"}, "limit": {"10"}, "query": {"select * from table"}},
		},
		{
			"1622505600", "1622592000", "20", "select name from table where age > 30",
			url.Values{"start": {"1622505600"},
				"end":   {"1622592000"},
				"limit": {"20"}, "query": {"select name from table where age > 30"}},
		},
		{
			"", "", "", "",
			url.Values{"start": {""}, "end": {""}, "limit": {""}, "query": {""}},
		},
		{
			"1622505600", "", "50", "select * from table where status = 'active'",
			url.Values{"start": {"1622505600"},
				"end": {""}, "limit": {"50"}, "query": {"select * from table where status = 'active'"}},
		},
		{
			"1622505600", "1622592000", "", "",
			url.Values{"start": {"1622505600"}, "end": {"1622592000"}, "limit": {""}, "query": {""}},
		},
	}

	for _, tc := range testCases {
		params := queryParamBuilder(tc.startTime, tc.endTime, tc.sizes, tc.query)

		if !reflect.DeepEqual(params, tc.expectParams) {
			t.Errorf("queryParamBuilder(%s, %s, %s, %s) = %v; want %v",
				tc.startTime, tc.endTime, tc.sizes, tc.query, params, tc.expectParams)
		}
	}
}

func TestLokiQueryBuilder(t *testing.T) {
	testCases := []struct {
		params        LokiQueryParams
		expectedQuery string
	}{
		{
			LokiQueryParams{},
			"{}",
		},
		{
			LokiQueryParams{Namespace: "default"},
			`{namespace="default"}`,
		},
		{
			LokiQueryParams{Pod: "mypod"},
			`{pod="mypod"}`,
		},
		{
			LokiQueryParams{Container: "mycontainer"},
			`{container="mycontainer"}`,
		},
		{
			LokiQueryParams{Filename: "/var/log/myfile.log"},
			`{filename="/var/log/myfile.log"}`,
		},
		{
			LokiQueryParams{Keyword: "error"},
			`{} |= "error"`,
		},
		{
			LokiQueryParams{LogLevel: "error"},
			fmt.Sprintf(`{} |~ "%s"`, logLevelRegex["error"]),
		},
		{
			LokiQueryParams{
				Namespace: "default",
				Pod:       "mypod",
				Container: "mycontainer",
				Filename:  "/var/log/myfile.log",
				Keyword:   "error",
				LogLevel:  "error",
			},
			fmt.Sprintf(
				`{namespace="default", pod="mypod", container="mycontainer", filename="/var/log/myfile.log"} |= "error" |~ "%s"`,
				logLevelRegex["error"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "default",
				Pod:       "mypod",
				Container: "mycontainer",
				Filename:  "/var/log/myfile.log",
				Keyword:   "error",
				LogLevel:  "error",
			},
			fmt.Sprintf(
				`{namespace="default", pod="mypod", container="mycontainer", filename="/var/log/myfile.log"} |= "error" |~ "%s"`,
				logLevelRegex["error"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "kube-system",
				Pod:       "api-server",
				Container: "apiserver",
				Filename:  "/var/log/apiserver.log",
				Keyword:   "critical",
				LogLevel:  "critical",
			},
			fmt.Sprintf(`{namespace="kube-system", pod="api-server", container="apiserver", `+
				`filename="/var/log/apiserver.log"} |= "critical" |~ "%s"`,
				logLevelRegex["critical"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "monitoring",
				Pod:       "prometheus",
				Container: "prometheus",
				Filename:  "/var/log/prometheus.log",
				Keyword:   "info",
				LogLevel:  "info",
			},
			fmt.Sprintf(`{namespace="monitoring", pod="prometheus", `+
				`container="prometheus", filename="/var/log/prometheus.log"} |= "info" |~ "%s"`,
				logLevelRegex["info"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "default",
				Pod:       "mypod",
				Container: "mycontainer",
				Filename:  "/var/log/myfile.log",
				Keyword:   "debug",
				LogLevel:  "debug",
			},
			fmt.Sprintf(
				`{namespace="default", pod="mypod", container="mycontainer", filename="/var/log/myfile.log"} |= "debug" |~ "%s"`,
				logLevelRegex["debug"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "kube-system",
				Pod:       "kube-dns",
				Container: "kubedns",
				Filename:  "/var/log/kubedns.log",
				Keyword:   "warning",
				LogLevel:  "warning",
			},
			fmt.Sprintf(
				`{namespace="kube-system", pod="kube-dns", container="kubedns", `+
					`filename="/var/log/kubedns.log"} |= "warning" |~ "%s"`,
				logLevelRegex["warning"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "default",
				Pod:       "mypod",
				Container: "mycontainer",
				Filename:  "/var/log/myfile.log",
				LogLevel:  "error",
			},
			fmt.Sprintf(
				`{namespace="default", pod="mypod", container="mycontainer", filename="/var/log/myfile.log"} |~ "%s"`,
				logLevelRegex["error"],
			),
		},
		{
			LokiQueryParams{
				Namespace: "kube-system",
				Pod:       "kube-dns",
				Container: "kubedns",
				Keyword:   "timeout",
			},
			`{namespace="kube-system", pod="kube-dns", container="kubedns"} |= "timeout"`,
		},
		{
			LokiQueryParams{Namespace: "default", LogLevel: "info"},
			fmt.Sprintf(`{namespace="default"} |~ "%s"`, logLevelRegex["info"]),
		},
		{
			LokiQueryParams{Pod: "mypod", Keyword: "debug"},
			`{pod="mypod"} |= "debug"`,
		},
		{
			LokiQueryParams{Container: "mycontainer", LogLevel: "debug"},
			fmt.Sprintf(`{container="mycontainer"} |~ "%s"`, logLevelRegex["debug"]),
		},
		{
			LokiQueryParams{Filename: "/var/log/myfile.log", Keyword: "critical"},
			`{filename="/var/log/myfile.log"} |= "critical"`,
		},
	}

	for _, tc := range testCases {
		query := lokiQueryBuilder(tc.params)
		if query != tc.expectedQuery {
			t.Errorf("lokiQueryBuilder(%v) = %v; want %v", tc.params, query, tc.expectedQuery)
		}
	}
}

func TestGetList(t *testing.T) {
	testCases := []struct {
		name         string
		result       map[string]interface{}
		params       QueryResourcesParams
		expectedNS   []string
		expectedPods []string
		expectedCont []string
	}{
		{
			name: "Filter by namespace and pod",
			result: map[string]interface{}{
				"data": []interface{}{
					map[string]interface{}{"namespace": "ns1", "pod": "pod1", "container": "cont1"},
					map[string]interface{}{"namespace": "ns1", "pod": "pod2", "container": "cont2"},
					map[string]interface{}{"namespace": "ns2", "pod": "pod1", "container": "cont1"},
				},
			},
			params:       QueryResourcesParams{Namespace: "ns1", Pod: "pod1", Container: ""},
			expectedNS:   []string{"ns1"},
			expectedPods: []string{"pod1"},
			expectedCont: []string{"cont1"},
		},
		{
			name: "Filter by namespace, pod, and container",
			result: map[string]interface{}{
				"data": []interface{}{
					map[string]interface{}{"namespace": "ns1", "pod": "pod1", "container": "cont1"},
					map[string]interface{}{"namespace": "ns1", "pod": "pod1", "container": "cont2"},
					map[string]interface{}{"namespace": "ns2", "pod": "pod1", "container": "cont1"},
				},
			},
			params:       QueryResourcesParams{Namespace: "ns1", Pod: "pod1", Container: "cont1"},
			expectedNS:   []string{"ns1"},
			expectedPods: []string{"pod1"},
			expectedCont: []string{"cont1"},
		},
		{
			name:         "Empty data",
			result:       map[string]interface{}{"data": []interface{}{}},
			params:       QueryResourcesParams{Namespace: "", Pod: "", Container: ""},
			expectedNS:   []string{},
			expectedPods: []string{},
			expectedCont: []string{},
		},
		{
			name:         "Invalid data structure",
			result:       map[string]interface{}{"data": "invalid"},
			params:       QueryResourcesParams{Namespace: "", Pod: "", Container: ""},
			expectedNS:   []string{},
			expectedPods: []string{},
			expectedCont: []string{},
		},
		{
			name: "Non-matching filters",
			result: map[string]interface{}{
				"data": []interface{}{
					map[string]interface{}{"namespace": "ns1", "pod": "pod1", "container": "cont1"},
					map[string]interface{}{"namespace": "ns2", "pod": "pod2", "container": "cont2"},
				},
			},
			params:       QueryResourcesParams{Namespace: "ns3", Pod: "", Container: ""},
			expectedNS:   []string{},
			expectedPods: []string{},
			expectedCont: []string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			namespaces, pods, containers := getList(tc.result, tc.params)
			if !equalSlices(namespaces, tc.expectedNS) {
				t.Errorf("expected namespaces %v, but got %v", tc.expectedNS, namespaces)
			}
			if !equalSlices(pods, tc.expectedPods) {
				t.Errorf("expected pods %v, but got %v", tc.expectedPods, pods)
			}
			if !equalSlices(containers, tc.expectedCont) {
				t.Errorf("expected containers %v, but got %v", tc.expectedCont, containers)
			}
		})
	}
}
func equalSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	set := make(map[string]struct{}, len(a))
	for _, v := range a {
		set[v] = struct{}{}
	}
	for _, v := range b {
		if _, ok := set[v]; !ok {
			return false
		}
	}
	return true
}

func TestFindContainerByName(t *testing.T) {
	containers := []Container{
		{Name: "container1"},
		{Name: "container2"},
		{Name: "container3"},
	}

	testCases := []struct {
		name          string
		containerName string
		expectedName  string
		expectError   bool
	}{
		{
			name:          "Container Found",
			containerName: "container1",
			expectedName:  "container1",
			expectError:   false,
		},
		{
			name:          "Container Not Found",
			containerName: "container4",
			expectedName:  "",
			expectError:   true,
		},
		{
			name:          "Empty Container List",
			containerName: "container1",
			expectedName:  "",
			expectError:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var result *Container
			var err error
			if tc.name == "Empty Container List" {
				result, err = findContainerByName([]Container{}, tc.containerName)
			} else {
				result, err = findContainerByName(containers, tc.containerName)
			}

			if tc.expectError {
				if err == nil {
					t.Errorf("expected error, but got none")
				} else if err.Error() != fmt.Sprintf("container %s not found", tc.containerName) {
					t.Errorf("expected error message 'container %s not found', but got '%s'", tc.containerName, err.Error())
				} else {
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if result == nil || result.Name != tc.expectedName {
					t.Errorf("expected container name %s, but got %v", tc.expectedName, result)
				}
			}
		})
	}
}

func TestExtractReadOnlyMounts(t *testing.T) {
	testCases := []struct {
		name           string
		container      Container
		expectedMounts []VolumeMount
	}{
		{
			name: "Container with ReadOnly Mounts",
			container: Container{
				Name: "container1",
				VolumeMounts: []VolumeMount{
					{Name: "mount1", ReadOnly: true},
					{Name: "mount2", ReadOnly: false},
					{Name: "mount3", ReadOnly: true},
				},
			},
			expectedMounts: []VolumeMount{
				{Name: "mount1", ReadOnly: true},
				{Name: "mount3", ReadOnly: true},
			},
		},
		{
			name: "Container without ReadOnly Mounts",
			container: Container{
				Name: "container2",
				VolumeMounts: []VolumeMount{
					{Name: "mount1", ReadOnly: false},
					{Name: "mount2", ReadOnly: false},
				},
			},
			expectedMounts: []VolumeMount{},
		},
		{
			name: "Container with No Mounts",
			container: Container{
				Name:         "container3",
				VolumeMounts: []VolumeMount{},
			},
			expectedMounts: []VolumeMount{},
		},
		{
			name: "Container with All ReadOnly Mounts",
			container: Container{
				Name: "container4",
				VolumeMounts: []VolumeMount{
					{Name: "mount1", ReadOnly: true},
					{Name: "mount2", ReadOnly: true},
				},
			},
			expectedMounts: []VolumeMount{
				{Name: "mount1", ReadOnly: true},
				{Name: "mount2", ReadOnly: true},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mounts := extractReadOnlyMounts(&tc.container)
			if len(mounts) != len(tc.expectedMounts) {
				t.Errorf("expected %d read-only mounts, got %d", len(tc.expectedMounts), len(mounts))
			}
			for i, mount := range mounts {
				if mount != tc.expectedMounts[i] {
					t.Errorf("expected mount %v, got %v", tc.expectedMounts[i], mount)
				}
			}
		})
	}
}

func TestParsePodDetail(t *testing.T) {
	testCases := []struct {
		name           string
		respBody       []byte
		containerName  string
		expectedMounts []VolumeMount
		expectError    bool
		expectedErrMsg string
	}{
		{
			name: "Successful JSON Parsing and Container Found",
			respBody: []byte(`{
				"Spec": {
					"Containers": [
						{
							"Name": "container1",
							"VolumeMounts": [
								{"Name": "mount1", "ReadOnly": true},
								{"Name": "mount2", "ReadOnly": false}
							]
						},
						{
							"Name": "container2",
							"VolumeMounts": [
								{"Name": "mount3", "ReadOnly": true}
							]
						}
					]
				}
			}`),
			containerName: "container1",
			expectedMounts: []VolumeMount{
				{Name: "mount1", ReadOnly: true},
			},
			expectError: false,
		},
		{
			name:           "JSON Parsing Error",
			respBody:       []byte(`{invalid json}`),
			containerName:  "container1",
			expectedMounts: []VolumeMount{},
			expectError:    true,
			expectedErrMsg: "error parsing JSON response:",
		},
		{
			name: "Container Not Found",
			respBody: []byte(`{
				"Spec": {
					"Containers": [
						{
							"Name": "container1",
							"VolumeMounts": [
								{"Name": "mount1", "ReadOnly": true},
								{"Name": "mount2", "ReadOnly": false}
							]
						}
					]
				}
			}`),
			containerName:  "container3",
			expectedMounts: []VolumeMount{},
			expectError:    true,
			expectedErrMsg: "container container3 not found",
		},
		{
			name: "Container Found with No ReadOnly Mounts",
			respBody: []byte(`{
				"Spec": {
					"Containers": [
						{
							"Name": "container1",
							"VolumeMounts": [
								{"Name": "mount1", "ReadOnly": false},
								{"Name": "mount2", "ReadOnly": false}
							]
						}
					]
				}
			}`),
			containerName:  "container1",
			expectedMounts: []VolumeMount{},
			expectError:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mounts, err := parsePodDetail(tc.respBody, tc.containerName)

			if tc.expectError {
				if err == nil {
					t.Errorf("expected error, but got none")
				} else if !contains(err.Error(), tc.expectedErrMsg) {
					t.Errorf("expected error message to contain '%s', but got '%s'", tc.expectedErrMsg, err.Error())
				} else {
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if len(mounts) != len(tc.expectedMounts) {
					t.Errorf("expected %d read-only mounts, got %d", len(tc.expectedMounts), len(mounts))
				}
				for i, mount := range mounts {
					if mount != tc.expectedMounts[i] {
						t.Errorf("expected mount %v, got %v", tc.expectedMounts[i], mount)
					}
				}
			}
		})
	}
}

func contains(str, substr string) bool {
	return str != "" && substr != "" && strings.Contains(str, substr)
}

func TestFindPodByNamePrefix(t *testing.T) {
	testCases := []struct {
		name           string
		pods           []v1.Pod
		prefix         string
		expectedPod    string
		expectError    bool
		expectedErrMsg string
	}{
		{
			name: "Pod Found with Prefix",
			pods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "otherpod-1"}},
			},
			prefix:      "mypod",
			expectedPod: "mypod-1",
			expectError: false,
		},
		{
			name: "No Pod Found with Prefix",
			pods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "otherpod-1"}},
			},
			prefix:         "nopod",
			expectedPod:    "",
			expectError:    true,
			expectedErrMsg: "no pod found with prefix nopod",
		},
		{
			name: "Multiple Pods with Same Prefix",
			pods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-3"}},
			},
			prefix:      "mypod",
			expectedPod: "mypod-1",
			expectError: false,
		},
		{
			name: "Prefix Matches Partially",
			pods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-abc"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-def"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "anotherpod-1"}},
			},
			prefix:      "mypod-a",
			expectedPod: "mypod-abc",
			expectError: false,
		},
		{
			name:           "No Pods Available",
			pods:           []v1.Pod{},
			prefix:         "mypod",
			expectedPod:    "",
			expectError:    true,
			expectedErrMsg: "no pod found with prefix mypod",
		},
		{
			name: "Exact Match with Prefix",
			pods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-2"}},
			},
			prefix:      "mypod",
			expectedPod: "mypod",
			expectError: false,
		},
		{
			name: "Special Characters in Prefix",
			pods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-@123"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "mypod-!abc"}},
			},
			prefix:      "mypod-@",
			expectedPod: "mypod-@123",
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a fake client
			client := fake.NewSimpleClientset()

			// Add the provided pods to the fake client
			for _, pod := range tc.pods {
				_, err := client.CoreV1().Pods("").Create(context.TODO(), &pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create pod: %v", err)
				}
			}

			handler := &Handler{PodClient: client.CoreV1().Pods("")}

			// Call the function
			podName, err := handler.findPodByNamePrefix(tc.prefix)

			// Check the results
			if tc.expectError {
				if err == nil {
					t.Errorf("expected error, but got none")
				} else if !strings.Contains(err.Error(), tc.expectedErrMsg) {
					t.Errorf("expected error message to contain '%s', but got '%s'", tc.expectedErrMsg, err.Error())
				} else {
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if podName != tc.expectedPod {
					t.Errorf("expected pod name '%s', but got '%s'", tc.expectedPod, podName)
				}
			}
		})
	}
}

func TestGetPodNode(t *testing.T) {
	testCases := []struct {
		name           string
		podName        string
		pod            *v1.Pod
		expectedNode   string
		expectError    bool
		expectedErrMsg string
	}{
		{
			name:    "Pod Found",
			podName: "mypod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod"},
				Spec:       v1.PodSpec{NodeName: "node1"},
			},
			expectedNode: "node1",
			expectError:  false,
		},
		{
			name:           "Pod Not Found",
			podName:        "nonexistent",
			pod:            nil,
			expectedNode:   "",
			expectError:    true,
			expectedErrMsg: `failed to get pod: pods "nonexistent" not found`,
		},
		{
			name:    "Pod Found with No Node Assigned",
			podName: "mypod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod"},
				Spec:       v1.PodSpec{},
			},
			expectedNode: "",
			expectError:  false,
		},
		{
			name:    "Another Pod Found",
			podName: "anotherpod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "anotherpod"},
				Spec:       v1.PodSpec{NodeName: "node2"},
			},
			expectedNode: "node2",
			expectError:  false,
		},
		{
			name:    "Pod with Special Characters in Name",
			podName: "pod-123-!@#",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-123-!@#"},
				Spec:       v1.PodSpec{NodeName: "node-special"},
			},
			expectedNode: "node-special",
			expectError:  false,
		},
		{
			name:    "Pod with Long Name",
			podName: "this-is-a-very-long-pod-name-to-test",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "this-is-a-very-long-pod-name-to-test"},
				Spec:       v1.PodSpec{NodeName: "node-long"},
			},
			expectedNode: "node-long",
			expectError:  false,
		},
		{
			name:    "Pod Found on Node with Special Characters",
			podName: "mypod-special-node",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod-special-node"},
				Spec:       v1.PodSpec{NodeName: "node-!@#"},
			},
			expectedNode: "node-!@#",
			expectError:  false,
		},
		{
			name:    "Pod Found on Empty Node",
			podName: "mypod-empty-node",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod-empty-node"},
				Spec:       v1.PodSpec{NodeName: ""},
			},
			expectedNode: "",
			expectError:  false,
		},
		{
			name:    "Pod Found on Node with Numbers",
			podName: "mypod-numbers",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod-numbers"},
				Spec:       v1.PodSpec{NodeName: "node123"},
			},
			expectedNode: "node123",
			expectError:  false,
		},
		{
			name:    "Pod Found on Node with Hyphens",
			podName: "mypod-hyphens",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod-hyphens"},
				Spec:       v1.PodSpec{NodeName: "node-123-abc"},
			},
			expectedNode: "node-123-abc",
			expectError:  false,
		},
		{
			name:    "Pod Found with No Node Assigned",
			podName: "mypod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod"},
				Spec:       v1.PodSpec{},
			},
			expectedNode: "",
			expectError:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()

			// Add the provided pod to the fake client
			if tc.pod != nil {
				_, err := client.CoreV1().Pods("").Create(context.TODO(), tc.pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create pod: %v", err)
				}
			}

			handler := &Handler{PodClient: client.CoreV1().Pods("")}

			// Call the function
			nodeName, err := handler.getPodNode(tc.podName)

			// Check the results
			if tc.expectError {
				if err == nil {
					t.Errorf("expected error, but got none")
				} else if err.Error() != tc.expectedErrMsg {
					t.Errorf("expected error message '%s', but got '%s'", tc.expectedErrMsg, err.Error())
				} else {
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if nodeName != tc.expectedNode {
					t.Errorf("expected node name '%s', but got '%s'", tc.expectedNode, nodeName)
				}
			}
		})
	}
}
func TestGetReadOnlyVolumesByNode(t *testing.T) {
	testCases := []struct {
		name           string
		pods           []v1.Pod
		expectedResult map[string][]VolumeMount
		expectedErrMsg string
		expectError    bool
	}{
		{
			name: "Pod Found with ReadOnly Volumes",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: true},
									{Name: "vol2", MountPath: "/path2", ReadOnly: false},
								},
							},
							{
								Name: "other",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol3", MountPath: "/path3", ReadOnly: true},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{
				"loki-promtail-1/promtail": {
					{MountPath: "/path1", Name: "vol1", ReadOnly: true},
				},
			},
			expectError: false,
		},
		{
			name: "No ReadOnly Volumes",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: false},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{},
			expectError:    false,
		},
		{
			name:           "Pod Not Found",
			pods:           []v1.Pod{},
			expectedResult: nil,
			expectedErrMsg: "no pod found with prefix loki-promtail",
			expectError:    true,
		},
		{
			name: "Pod Found with Multiple ReadOnly Volumes",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: true},
									{Name: "vol2", MountPath: "/path2", ReadOnly: true},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{
				"loki-promtail-1/promtail": {
					{MountPath: "/path1", Name: "vol1", ReadOnly: true},
					{MountPath: "/path2", Name: "vol2", ReadOnly: true},
				},
			},
			expectError: false,
		},
		{
			name: "Pod Found with No Containers",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec:       v1.PodSpec{},
				},
			},
			expectedResult: map[string][]VolumeMount{},
			expectError:    false,
		},
		{
			name: "Pod Found with No VolumeMounts",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:         "promtail",
								VolumeMounts: []v1.VolumeMount{},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{},
			expectError:    false,
		},
		{
			name: "Pod Found with Mixed VolumeMounts",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: true},
									{Name: "vol2", MountPath: "/path2", ReadOnly: false},
									{Name: "vol3", MountPath: "/path3", ReadOnly: true},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{
				"loki-promtail-1/promtail": {
					{MountPath: "/path1", Name: "vol1", ReadOnly: true},
					{MountPath: "/path3", Name: "vol3", ReadOnly: true},
				},
			},
			expectError: false,
		},
		{
			name: "Pod Found with Multiple Containers",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: true},
								},
							},
							{
								Name: "another-container",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol2", MountPath: "/path2", ReadOnly: true},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{
				"loki-promtail-1/promtail": {
					{MountPath: "/path1", Name: "vol1", ReadOnly: true},
				},
			},
			expectError: false,
		},
		{
			name: "Pod Found with Containers Having No ReadOnly Volumes",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: false},
								},
							},
							{
								Name: "another-container",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol2", MountPath: "/path2", ReadOnly: false},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{},
			expectError:    false,
		},
		{
			name: "Pod Found with Containers Having Mixed ReadOnly Volumes",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: true},
									{Name: "vol2", MountPath: "/path2", ReadOnly: false},
								},
							},
							{
								Name: "another-container",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol3", MountPath: "/path3", ReadOnly: true},
									{Name: "vol4", MountPath: "/path4", ReadOnly: false},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{
				"loki-promtail-1/promtail": {
					{MountPath: "/path1", Name: "vol1", ReadOnly: true},
				},
			},
			expectError: false,
		},
		{
			name: "Pod Found with Promtail Container Having No VolumeMounts",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:         "promtail",
								VolumeMounts: []v1.VolumeMount{},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{},
			expectError:    false,
		},
		{
			name: "Pod Found with No Promtail Container",
			pods: []v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "loki-promtail-1"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "not-promtail",
								VolumeMounts: []v1.VolumeMount{
									{Name: "vol1", MountPath: "/path1", ReadOnly: true},
								},
							},
						},
					},
				},
			},
			expectedResult: map[string][]VolumeMount{},
			expectError:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()

			// Add the provided pods to the fake client
			for _, pod := range tc.pods {
				_, err := client.CoreV1().Pods("").Create(context.TODO(), &pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create pod: %v", err)
				}
			}

			handler := &Handler{PodClient: client.CoreV1().Pods("")}

			// Call the function
			result, err := handler.getReadOnlyVolumesByNode()

			// Check the results
			if tc.expectError {
				if err == nil {
					t.Errorf("expected error, but got none")
				} else if err.Error() != tc.expectedErrMsg {
					t.Errorf("expected error message '%s', but got '%s'", tc.expectedErrMsg, err.Error())
				} else {
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if !equalVolumeMountMaps(result, tc.expectedResult) {
					t.Errorf("expected result %v, but got %v", tc.expectedResult, result)
				}
			}
		})
	}
}

func equalVolumeMountMaps(a, b map[string][]VolumeMount) bool {
	if len(a) != len(b) {
		return false
	}
	for key, aValue := range a {
		bValue, ok := b[key]
		if !ok || len(aValue) != len(bValue) {
			return false
		}
		for i, v := range aValue {
			if v != bValue[i] {
				return false
			}
		}
	}
	return true
}
func TestDeterminePath(t *testing.T) {
	testCases := []struct {
		name           string
		scrapeConfig   ScrapeConfig
		expectedResult string
	}{
		{
			name: "Kubernetes Pods Job",
			scrapeConfig: ScrapeConfig{
				JobName: "kubernetes-pods",
			},
			expectedResult: "/var/log/pods/*/*/*.log",
		},
		{
			name: "StaticConfig with Path Label",
			scrapeConfig: ScrapeConfig{
				JobName: "custom-job",
				StaticConfigs: []StaticConfig{
					{Labels: map[string]string{"__path__": "/var/logs/custom.log"}},
				},
			},
			expectedResult: "/var/logs/custom.log",
		},
		{
			name: "StaticConfig without Path Label",
			scrapeConfig: ScrapeConfig{
				JobName: "custom-job",
				StaticConfigs: []StaticConfig{
					{Labels: map[string]string{"__other__": "/var/logs/other.log"}},
				},
			},
			expectedResult: "",
		},
		{
			name: "Multiple StaticConfigs with Path Label",
			scrapeConfig: ScrapeConfig{
				JobName: "custom-job",
				StaticConfigs: []StaticConfig{
					{Labels: map[string]string{"__other__": "/var/logs/other.log"}},
					{Labels: map[string]string{"__path__": "/var/logs/custom.log"}},
				},
			},
			expectedResult: "/var/logs/custom.log",
		},
		{
			name: "Empty StaticConfigs",
			scrapeConfig: ScrapeConfig{
				JobName:       "custom-job",
				StaticConfigs: []StaticConfig{},
			},
			expectedResult: "",
		},
		{
			name: "StaticConfig with Empty Path Label",
			scrapeConfig: ScrapeConfig{
				JobName: "custom-job",
				StaticConfigs: []StaticConfig{
					{Labels: map[string]string{"__path__": ""}},
				},
			},
			expectedResult: "",
		},
		{
			name: "Multiple StaticConfigs with Multiple Path Labels",
			scrapeConfig: ScrapeConfig{
				JobName: "custom-job",
				StaticConfigs: []StaticConfig{
					{Labels: map[string]string{"__path__": "/var/logs/first.log"}},
					{Labels: map[string]string{"__path__": "/var/logs/second.log"}},
				},
			},
			expectedResult: "/var/logs/first.log",
		},
		{
			name: "Job Name Not Kubernetes Pods and No Path Label",
			scrapeConfig: ScrapeConfig{
				JobName: "other-job",
			},
			expectedResult: "",
		},
		{
			name: "Job Name Kubernetes Pods with Empty StaticConfigs",
			scrapeConfig: ScrapeConfig{
				JobName:       "kubernetes-pods",
				StaticConfigs: []StaticConfig{},
			},
			expectedResult: "/var/log/pods/*/*/*.log",
		},
		{
			name: "Job Name Kubernetes Pods with Path Label",
			scrapeConfig: ScrapeConfig{
				JobName: "kubernetes-pods",
				StaticConfigs: []StaticConfig{
					{Labels: map[string]string{"__path__": "/var/logs/override.log"}},
				},
			},
			expectedResult: "/var/log/pods/*/*/*.log",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := determinePath(tc.scrapeConfig)

			if result != tc.expectedResult {
				t.Errorf("expected result '%s', but got '%s'", tc.expectedResult, result)
			}
		})
	}
}

func TestGetJobPaths(t *testing.T) {
	testCases := []struct {
		name           string
		promtailConfig PromtailConfig
		expectedResult map[string]string
	}{
		{
			name: "Single ScrapeConfig",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{JobName: "job1"},
				},
			},
			expectedResult: map[string]string{
				"job1": "",
			},
		},
		{
			name: "Multiple ScrapeConfigs",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{JobName: "job1"},
					{JobName: "job2"},
				},
			},
			expectedResult: map[string]string{
				"job1": "",
				"job2": "",
			},
		},
		{
			name: "No ScrapeConfigs",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{},
			},
			expectedResult: map[string]string{},
		},
		{
			name: "ScrapeConfig with Path Label",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{
						JobName: "custom-job",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__path__": "/var/logs/custom.log"}},
						},
					},
				},
			},
			expectedResult: map[string]string{
				"custom-job": "/var/logs/custom.log",
			},
		},
		{
			name: "ScrapeConfig with Empty Path Label",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{
						JobName: "custom-job",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__path__": ""}},
						},
					},
				},
			},
			expectedResult: map[string]string{
				"custom-job": "",
			},
		},
		{
			name: "ScrapeConfigs with Duplicate JobNames",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{JobName: "job1"},
					{JobName: "job1"},
				},
			},
			expectedResult: map[string]string{
				"job1": "",
			},
		},
		{
			name: "Kubernetes Pods Job",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{JobName: "kubernetes-pods"},
				},
			},
			expectedResult: map[string]string{
				"kubernetes-pods": "/var/log/pods/*/*/*.log",
			},
		},
		{
			name: "Multiple StaticConfigs with Path Label",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{
						JobName: "custom-job",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__path__": "/var/logs/other.log"}},
							{Labels: map[string]string{"__path__": "/var/logs/custom.log"}},
						},
					},
				},
			},
			expectedResult: map[string]string{
				"custom-job": "/var/logs/other.log",
			},
		},
		{
			name: "ScrapeConfig without Path Label",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{
						JobName: "custom-job",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__other__": "/var/logs/other.log"}},
						},
					},
				},
			},
			expectedResult: map[string]string{
				"custom-job": "",
			},
		},
		{
			name: "Multiple ScrapeConfigs with Mixed Path Labels",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{
						JobName: "job1",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__path__": "/var/logs/job1.log"}},
						},
					},
					{
						JobName: "job2",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__path__": "/var/logs/job2.log"}},
						},
					},
					{
						JobName: "job3",
						StaticConfigs: []StaticConfig{
							{Labels: map[string]string{"__other__": "/var/logs/other.log"}},
						},
					},
				},
			},
			expectedResult: map[string]string{
				"job1": "/var/logs/job1.log",
				"job2": "/var/logs/job2.log",
				"job3": "",
			},
		},
		{
			name: "Empty StaticConfigs",
			promtailConfig: PromtailConfig{
				ScrapeConfigs: []ScrapeConfig{
					{
						JobName:       "custom-job",
						StaticConfigs: []StaticConfig{},
					},
				},
			},
			expectedResult: map[string]string{
				"custom-job": "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := getJobPaths(tc.promtailConfig)

			if !equalMaps(result, tc.expectedResult) {
				t.Errorf("expected result %v, but got %v", tc.expectedResult, result)
			}
		})
	}
}

func equalMaps(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for key, aValue := range a {
		bValue, ok := b[key]
		if !ok || aValue != bValue {
			return false
		}
	}
	return true
}
func TestDeletePod(t *testing.T) {
	testCases := []struct {
		name          string
		existingPods  []v1.Pod
		podName       string
		expectedError bool
	}{
		{
			name: "Pod Exists",
			existingPods: []v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}},
			},
			podName:       "test-pod",
			expectedError: false,
		},
		{
			name:          "Pod Does Not Exist",
			existingPods:  []v1.Pod{},
			podName:       "nonexistent-pod",
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()

			for _, pod := range tc.existingPods {
				_, err := client.CoreV1().Pods("").Create(context.TODO(), &pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create pod: %v", err)
				}
			}

			handler := &Handler{PodClient: client.CoreV1().Pods("")}

			err := handler.deletePod(tc.podName)

			if tc.expectedError {
				if err == nil {
					t.Errorf("expected error, but got none")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}

				_, err = client.CoreV1().Pods("").Get(context.TODO(), tc.podName, metav1.GetOptions{})
				if err == nil {
					t.Errorf("expected pod to be deleted, but it still exists")
				}
			}
		})
	}
}

func TestCalculateTimeRange(t *testing.T) {

	testCases := []struct {
		name          string
		pivotTime     string
		interval      string
		isFirstHalf   bool
		expectedStart string
		expectedEnd   string
		expectError   bool
	}{
		{
			name:          "Valid Input - First Half",
			pivotTime:     "1638316800000000000",
			interval:      "1h",
			isFirstHalf:   true,
			expectedStart: "1638313200",
			expectedEnd:   "1638316800",
			expectError:   false,
		},
		{
			name:          "Valid Input - Second Half",
			pivotTime:     "1638316800000000000",
			interval:      "1h",
			isFirstHalf:   false,
			expectedStart: "1638316800",
			expectedEnd:   "1638320400",
			expectError:   false,
		},
		{
			name:          "Invalid Pivot Time",
			pivotTime:     "invalid",
			interval:      "1h",
			isFirstHalf:   true,
			expectedStart: "0",
			expectedEnd:   "0",
			expectError:   true,
		},
		{
			name:          "Invalid Interval",
			pivotTime:     "1638316800000000000",
			interval:      "invalid",
			isFirstHalf:   true,
			expectedStart: "0",
			expectedEnd:   "0",
			expectError:   true,
		},
		{
			name:          "Empty Interval",
			pivotTime:     "1638316800000000000",
			interval:      "",
			isFirstHalf:   true,
			expectedStart: "0",
			expectedEnd:   "0",
			expectError:   true,
		},
		{
			name:          "Empty Pivot Time",
			pivotTime:     "",
			interval:      "1h",
			isFirstHalf:   true,
			expectedStart: "0",
			expectedEnd:   "0",
			expectError:   true,
		},
		{
			name:          "Zero Interval",
			pivotTime:     "1638316800000000000",
			interval:      "0s",
			isFirstHalf:   true,
			expectedStart: "1638316800",
			expectedEnd:   "1638316800",
			expectError:   false,
		},
		{
			name:          "Negative Interval - First Half",
			pivotTime:     "1638316800000000000",
			interval:      "-1h",
			isFirstHalf:   true,
			expectedStart: "1638320400",
			expectedEnd:   "1638316800",
			expectError:   false,
		},
		{
			name:          "Negative Interval - Second Half",
			pivotTime:     "1638316800000000000",
			interval:      "-1h",
			isFirstHalf:   false,
			expectedStart: "1638316800",
			expectedEnd:   "1638313200",
			expectError:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			start, end := calculateTimeRange(tc.pivotTime, tc.interval, tc.isFirstHalf)
			if start != tc.expectedStart || end != tc.expectedEnd {
				t.Errorf("expected start %s and end %s, but got start %s and end %s", tc.expectedStart, tc.expectedEnd, start, end)
			}
		})
	}
}

func TestBuildQueryParams(t *testing.T) {
	testCases := []struct {
		name        string
		queryParams map[string]string
		expected    LokiQueryParams
	}{
		{
			name: "All query parameters present",
			queryParams: map[string]string{
				"fileName": "log.txt",
				"level":    "error",
				"keyword":  "failure",
			},
			expected: LokiQueryParams{
				Filename: "log.txt",
				LogLevel: "error",
				Keyword:  "failure",
			},
		},
		{
			name: "Missing filename",
			queryParams: map[string]string{
				"level":   "error",
				"keyword": "failure",
			},
			expected: LokiQueryParams{
				Filename: "",
				LogLevel: "error",
				Keyword:  "failure",
			},
		},
		{
			name: "Missing loglevel",
			queryParams: map[string]string{
				"fileName": "log.txt",
				"keyword":  "failure",
			},
			expected: LokiQueryParams{
				Filename: "log.txt",
				LogLevel: "",
				Keyword:  "failure",
			},
		},
		{
			name: "Missing keyword",
			queryParams: map[string]string{
				"fileName": "log.txt",
				"level":    "error",
			},
			expected: LokiQueryParams{
				Filename: "log.txt",
				LogLevel: "error",
				Keyword:  "",
			},
		},
		{
			name:        "No query parameters",
			queryParams: map[string]string{},
			expected: LokiQueryParams{
				Filename: "",
				LogLevel: "",
				Keyword:  "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a new restful.Request
			req := httptest.NewRequest("GET", "/dummy", nil)
			query := req.URL.Query()
			for key, value := range tc.queryParams {
				query.Add(key, value)
			}
			req.URL.RawQuery = query.Encode()
			restfulReq := restful.NewRequest(req)

			handler := &Handler{}
			result := handler.buildQueryParams(restfulReq)

			if result != tc.expected {
				t.Errorf("expected %+v, but got %+v", tc.expected, result)
			}
		})
	}
}

func TestSendHTTPRequest(t *testing.T) {
	testCases := []struct {
		name          string
		reqKey        string
		podName       string
		requestsMap   map[string]*HTTPRequest
		expectedError string
	}{
		{
			name:    "Valid request",
			reqKey:  "testRequest",
			podName: "testPod",
			requestsMap: map[string]*HTTPRequest{
				"testRequest": {
					Method: "GET",
					URL:    "http://example.com/",
					Body:   nil,
					Headers: map[string]string{
						"Content-Type": "application/json",
					},
				},
			},
			expectedError: "",
		},
		{
			name:          "Request key not found",
			reqKey:        "invalidRequest",
			podName:       "testPod",
			requestsMap:   map[string]*HTTPRequest{},
			expectedError: "request configuration for 'invalidRequest' not found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handler := &Handler{
				RequestsMap: tc.requestsMap,
			}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte(`{"message": "success"}`)); err != nil {
					t.Fatalf("failed to write response: %v", err)
				}
			}))
			defer server.Close()

			if tc.name == "Valid request" {
				handler.RequestsMap[tc.reqKey].URL = server.URL + "/"
			}

			resp, err := handler.sendHTTPRequest(tc.reqKey, tc.podName)

			if tc.expectedError != "" {
				if err == nil {
					t.Fatalf("expected error: %v, got: nil", tc.expectedError)
				}
				if !contains(err.Error(), tc.expectedError) {
					t.Fatalf("expected error: %v, got: %v", tc.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if resp.StatusCode != http.StatusOK {
					t.Fatalf("expected status code: %v, got: %v", http.StatusOK, resp.StatusCode)
				}
			}
		})
	}
}
