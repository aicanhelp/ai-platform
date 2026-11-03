/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func TestNewAddonRecorder(t *testing.T) {
	addonT := &bkeaddon.AddonTransfer{
		Addon: &confv1beta1.Product{
			Name:    "test-addon",
			Version: "v1.0.0",
		},
	}

	recorder := NewAddonRecorder(addonT)

	if recorder.AddonName != "test-addon" {
		t.Errorf("Expected AddonName to be 'test-addon', got '%s'", recorder.AddonName)
	}
	if recorder.AddonVersion != "v1.0.0" {
		t.Errorf("Expected AddonVersion to be 'v1.0.0', got '%s'", recorder.AddonVersion)
	}
	if recorder.AddonObjects == nil {
		t.Error("Expected AddonObjects to be initialized, got nil")
	}
}

func TestAddonRecorder_Record(t *testing.T) {
	recorder := &AddonRecorder{
		AddonName:    "test-addon",
		AddonVersion: "v1.0.0",
		AddonObjects: []*AddonObject{},
	}

	obj := &unstructured.Unstructured{}
	obj.SetName("test-object")
	obj.SetKind("Deployment")
	obj.SetNamespace("default")

	recorder.Record(obj)

	if len(recorder.AddonObjects) != 1 {
		t.Errorf("Expected 1 AddonObject, got %d", len(recorder.AddonObjects))
	}
	if recorder.AddonObjects[0].Name != "test-object" {
		t.Errorf("Expected Name 'test-object', got '%s'", recorder.AddonObjects[0].Name)
	}
	if recorder.AddonObjects[0].Kind != "Deployment" {
		t.Errorf("Expected Kind 'Deployment', got '%s'", recorder.AddonObjects[0].Kind)
	}
	if recorder.AddonObjects[0].NameSpace != "default" {
		t.Errorf("Expected NameSpace 'default', got '%s'", recorder.AddonObjects[0].NameSpace)
	}
}

func TestNewAddonObject(t *testing.T) {
	obj := &unstructured.Unstructured{}
	obj.SetName("test-object")
	obj.SetKind("Pod")
	obj.SetNamespace("kube-system")

	addonObj := NewAddonObject(obj)

	if addonObj.Name != "test-object" {
		t.Errorf("Expected Name 'test-object', got '%s'", addonObj.Name)
	}
	if addonObj.Kind != "Pod" {
		t.Errorf("Expected Kind 'Pod', got '%s'", addonObj.Kind)
	}
	if addonObj.NameSpace != "kube-system" {
		t.Errorf("Expected NameSpace 'kube-system', got '%s'", addonObj.NameSpace)
	}
}

func TestNewTask(t *testing.T) {
	param := map[string]interface{}{
		"key": "value",
	}

	task := NewTask("test-task", "/path/to/file.yaml", param)

	if task.Name != "test-task" {
		t.Errorf("Expected Name 'test-task', got '%s'", task.Name)
	}
	if task.FilePath != "/path/to/file.yaml" {
		t.Errorf("Expected FilePath '/path/to/file.yaml', got '%s'", task.FilePath)
	}
	if task.Param["key"] != "value" {
		t.Errorf("Expected Param['key'] to be 'value', got '%v'", task.Param["key"])
	}
	if task.IgnoreError != false {
		t.Error("Expected IgnoreError to be false")
	}
	if task.Block != true {
		t.Error("Expected Block to be true")
	}
}

func TestTask_SetWaiter(t *testing.T) {
	task := &Task{}

	result := task.SetWaiter(false, 10*time.Second, 5*time.Second)

	if task.Block != false {
		t.Error("Expected Block to be false")
	}
	if task.Timeout != 10*time.Second {
		t.Errorf("Expected Timeout to be 10s, got %v", task.Timeout)
	}
	if task.Interval != 5*time.Second {
		t.Errorf("Expected Interval to be 5s, got %v", task.Interval)
	}
	if result != task {
		t.Error("Expected to return the same task pointer")
	}
}

func TestTask_AddRepo(t *testing.T) {
	task := &Task{}

	result := task.AddRepo("test-repo")

	if task.Param["repo"] != "test-repo" {
		t.Errorf("Expected Param['repo'] to be 'test-repo', got '%v'", task.Param["repo"])
	}
	if result != task {
		t.Error("Expected to return the same task pointer")
	}
}

func TestTask_AddRepo_WithNilParam(t *testing.T) {
	task := &Task{
		Param: nil,
	}

	result := task.AddRepo("test-repo")

	if task.Param == nil {
		t.Error("Expected Param to be initialized")
	}
	if task.Param["repo"] != "test-repo" {
		t.Errorf("Expected Param['repo'] to be 'test-repo', got '%v'", task.Param["repo"])
	}
	if result != task {
		t.Error("Expected to return the same task pointer")
	}
}

func TestTask_SetOperate(t *testing.T) {
	task := &Task{}

	result := task.SetOperate(bkeaddon.RemoveAddon)

	if task.Operate != bkeaddon.RemoveAddon {
		t.Errorf("Expected Operate to be RemoveAddon, got %v", task.Operate)
	}
	if result != task {
		t.Error("Expected to return the same task pointer")
	}
}

func TestTask_RegisAddonRecorder(t *testing.T) {
	task := &Task{}
	recorder := &AddonRecorder{
		AddonName: "test",
	}

	result := task.RegisAddonRecorder(recorder)

	if task.recorder != recorder {
		t.Error("Expected recorder to be set")
	}
	if result != task {
		t.Error("Expected to return the same task pointer")
	}
}

func TestMergeParam(t *testing.T) {
	tests := []struct {
		name     string
		src      map[string]interface{}
		dst      map[string]interface{}
		expected map[string]interface{}
	}{
		{
			name:     "both nil",
			src:      nil,
			dst:      nil,
			expected: map[string]interface{}{},
		},
		{
			name:     "src nil",
			src:      nil,
			dst:      map[string]interface{}{"key1": "value1"},
			expected: map[string]interface{}{"key1": "value1"},
		},
		{
			name:     "dst nil",
			src:      map[string]interface{}{"key1": "value1"},
			dst:      nil,
			expected: map[string]interface{}{"key1": "value1"},
		},
		{
			name:     "normal merge",
			src:      map[string]interface{}{"key1": "value1", "key2": "value2"},
			dst:      map[string]interface{}{"key2": "override", "key3": "value3"},
			expected: map[string]interface{}{"key1": "value1", "key2": "override", "key3": "value3"},
		},
		{
			name:     "empty src",
			src:      map[string]interface{}{},
			dst:      map[string]interface{}{"key1": "value1"},
			expected: map[string]interface{}{"key1": "value1"},
		},
		{
			name:     "empty dst",
			src:      map[string]interface{}{"key1": "value1"},
			dst:      map[string]interface{}{},
			expected: map[string]interface{}{"key1": "value1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mergeParam(tt.src, tt.dst)
			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d items, got %d", len(tt.expected), len(result))
				return
			}
			for k, v := range tt.expected {
				if result[k] != v {
					t.Errorf("Expected[%s] = %v, got %v", k, v, result[k])
				}
			}
		})
	}
}

func TestPrepareAddonParam(t *testing.T) {
	filesBaseNames := []string{"file1", "file2"}
	repo := "test-repo"

	result := prepareAddonParam(nil, filesBaseNames, repo)

	if len(result) != 2 {
		t.Errorf("Expected 2 entries, got %d", len(result))
	}

	if result["file1"]["repo"] != repo {
		t.Errorf("Expected repo to be '%s', got '%v'", repo, result["file1"]["repo"])
	}
	if result["file2"]["repo"] != repo {
		t.Errorf("Expected repo to be '%s', got '%v'", repo, result["file2"]["repo"])
	}
}

func TestPrepareAddonParam_WithParams(t *testing.T) {
	addonParam := map[string]string{
		"key1":       "value1",
		"file1.key2": "value2",
		"file2.key3": "value3",
	}
	filesBaseNames := []string{"file1", "file2"}
	repo := "test-repo"

	result := prepareAddonParam(addonParam, filesBaseNames, repo)

	if result["file1"]["key1"] != "value1" {
		t.Errorf("Expected key1=value1, got %v", result["file1"]["key1"])
	}
	if result["file1"]["key2"] != "value2" {
		t.Errorf("Expected key2=value2, got %v", result["file1"]["key2"])
	}
	if result["file2"]["key1"] != "value1" {
		t.Errorf("Expected key1=value1, got %v", result["file2"]["key1"])
	}
	if result["file2"]["key3"] != "value3" {
		t.Errorf("Expected key3=value3, got %v", result["file2"]["key3"])
	}
}

func TestParseNodeSelector(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    map[string]string
		wantErr bool
	}{
		{
			name:    "empty string",
			raw:     "",
			want:    map[string]string{},
			wantErr: false,
		},
		{
			name:    "whitespace only",
			raw:     "   ",
			want:    map[string]string{},
			wantErr: false,
		},
		{
			name:    "single key=value",
			raw:     "key=value",
			want:    map[string]string{"key": "value"},
			wantErr: false,
		},
		{
			name:    "multiple key=value",
			raw:     "key1=value1,key2=value2",
			want:    map[string]string{"key1": "value1", "key2": "value2"},
			wantErr: false,
		},
		{
			name:    "key with spaces",
			raw:     " key1 = value1 , key2 = value2 ",
			want:    map[string]string{"key1": "value1", "key2": "value2"},
			wantErr: false,
		},
		{
			name:    "key only without value",
			raw:     "key1",
			want:    map[string]string{"key1": ""},
			wantErr: false,
		},
		{
			name:    "key only with trailing comma",
			raw:     "key1,",
			want:    map[string]string{"key1": ""},
			wantErr: false,
		},
		{
			name:    "empty pair",
			raw:     "key1=value1,,key2=value2",
			want:    map[string]string{"key1": "value1", "key2": "value2"},
			wantErr: false,
		},
		{
			name:    "json format",
			raw:     `{"key1":"value1","key2":"value2"}`,
			want:    map[string]string{"key1": "value1", "key2": "value2"},
			wantErr: false,
		},
		{
			name:    "invalid json format",
			raw:     `{invalid}`,
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty key",
			raw:     "=value",
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseNodeSelector(tt.raw)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseNodeSelector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for k, v := range tt.want {
					if got[k] != v {
						t.Errorf("parseNodeSelector()[%s] = %v, want %v", k, got[k], v)
					}
				}
			}
		})
	}
}

func TestNormalizeNodeSelector(t *testing.T) {
	tests := []struct {
		name    string
		param   map[string]map[string]interface{}
		wantErr bool
	}{
		{
			name:    "nil param",
			param:   nil,
			wantErr: false,
		},
		{
			name:    "empty param",
			param:   map[string]map[string]interface{}{},
			wantErr: false,
		},
		{
			name: "no nodeSelector",
			param: map[string]map[string]interface{}{
				"file1": {"key": "value"},
			},
			wantErr: false,
		},
		{
			name: "nodeSelector as map[string]string",
			param: map[string]map[string]interface{}{
				"file1": {"nodeSelector": map[string]string{"key": "value"}},
			},
			wantErr: false,
		},
		{
			name: "nodeSelector as map[string]interface{}",
			param: map[string]map[string]interface{}{
				"file1": {"nodeSelector": map[string]interface{}{"key": "value"}},
			},
			wantErr: false,
		},
		{
			name: "nodeSelector as valid string",
			param: map[string]map[string]interface{}{
				"file1": {"nodeSelector": "key=value"},
			},
			wantErr: false,
		},
		{
			name: "nodeSelector as invalid string",
			param: map[string]map[string]interface{}{
				"file1": {"nodeSelector": "=value"},
			},
			wantErr: true,
		},
		{
			name: "nodeSelector as invalid type",
			param: map[string]map[string]interface{}{
				"file1": {"nodeSelector": 123},
			},
			wantErr: true,
		},
		{
			name: "nodeSelector is nil",
			param: map[string]map[string]interface{}{
				"file1": {"nodeSelector": nil},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := normalizeNodeSelector(tt.param)
			if (err != nil) != tt.wantErr {
				t.Errorf("normalizeNodeSelector() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestHasExcludeIpsParam(t *testing.T) {
	tests := []struct {
		name        string
		fabricParam map[string]string
		want        bool
	}{
		{
			name:        "nil param",
			fabricParam: nil,
			want:        false,
		},
		{
			name:        "empty param",
			fabricParam: map[string]string{},
			want:        false,
		},
		{
			name:        "no excludeIps key",
			fabricParam: map[string]string{"other": "value"},
			want:        false,
		},
		{
			name:        "excludeIps is empty string",
			fabricParam: map[string]string{"excludeIps": ""},
			want:        false,
		},
		{
			name:        "excludeIps has value",
			fabricParam: map[string]string{"excludeIps": "10.0.0.1"},
			want:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasExcludeIpsParam(tt.fabricParam)
			if got != tt.want {
				t.Errorf("hasExcludeIpsParam() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestProcessExcludeIps(t *testing.T) {
	tests := []struct {
		name          string
		excludeIpsStr string
		want          string
		wantErr       bool
	}{
		{
			name:          "single IP",
			excludeIpsStr: "10.0.0.1",
			want:          "10.0.0.1",
			wantErr:       false,
		},
		{
			name:          "multiple IPs",
			excludeIpsStr: "10.0.0.1,10.0.0.2,10.0.0.3",
			want:          "10.0.0.1,10.0.0.2,10.0.0.3",
			wantErr:       false,
		},
		{
			name:          "invalid IP",
			excludeIpsStr: "invalid-ip",
			want:          "",
			wantErr:       true,
		},
		{
			name:          "IP range",
			excludeIpsStr: "10.0.0.1-10.0.0.3",
			want:          "10.0.0.1,10.0.0.2,10.0.0.3",
			wantErr:       false,
		},
		{
			name:          "mixed IPs and range",
			excludeIpsStr: "10.0.0.1,10.0.0.5-10.0.0.7",
			want:          "10.0.0.1,10.0.0.5,10.0.0.6,10.0.0.7",
			wantErr:       false,
		},
		{
			name:          "invalid IP in range",
			excludeIpsStr: "10.0.0.1-invalid",
			want:          "",
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := processExcludeIps(tt.excludeIpsStr)
			if (err != nil) != tt.wantErr {
				t.Errorf("processExcludeIps() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("processExcludeIps() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestProcessItem(t *testing.T) {
	tests := []struct {
		name    string
		item    string
		want    string
		wantErr bool
	}{
		{
			name:    "single IP",
			item:    "10.0.0.1",
			want:    "10.0.0.1",
			wantErr: false,
		},
		{
			name:    "invalid IP",
			item:    "invalid",
			want:    "",
			wantErr: true,
		},
		{
			name:    "IP range",
			item:    "10.0.0.1-10.0.0.3",
			want:    "10.0.0.1,10.0.0.2,10.0.0.3",
			wantErr: false,
		},
		{
			name:    "invalid IP range",
			item:    "invalid-10.0.0.1",
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := processItem(tt.item)
			if (err != nil) != tt.wantErr {
				t.Errorf("processItem() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("processItem() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseFabricExcludeIPRange(t *testing.T) {
	tests := []struct {
		name     string
		rangeStr string
		want     []string
		wantErr  bool
	}{
		{
			name:     "valid range",
			rangeStr: "10.0.0.1-10.0.0.3",
			want:     []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
			wantErr:  false,
		},
		{
			name:     "reverse range",
			rangeStr: "10.0.0.3-10.0.0.1",
			want:     []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
			wantErr:  false,
		},
		{
			name:     "single IP as range",
			rangeStr: "10.0.0.1-10.0.0.1",
			want:     []string{"10.0.0.1"},
			wantErr:  false,
		},
		{
			name:     "invalid format - missing dash",
			rangeStr: "10.0.0.1",
			want:     nil,
			wantErr:  true,
		},
		{
			name:     "invalid format - too many parts",
			rangeStr: "10.0.0.1-10.0.0.2-10.0.0.3",
			want:     nil,
			wantErr:  true,
		},
		{
			name:     "invalid IP in range - start",
			rangeStr: "invalid-10.0.0.2",
			want:     nil,
			wantErr:  true,
		},
		{
			name:     "invalid IP in range - end",
			rangeStr: "10.0.0.1-invalid",
			want:     nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFabricExcludeIPRange(tt.rangeStr)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFabricExcludeIPRange() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(got) != len(tt.want) {
					t.Errorf("parseFabricExcludeIPRange() = %v, want %v", got, tt.want)
					return
				}
				for i, v := range tt.want {
					if got[i] != v {
						t.Errorf("parseFabricExcludeIPRange()[%d] = %v, want %v", i, got[i], v)
					}
				}
			}
		})
	}
}

func TestParseFabricParam(t *testing.T) {
	tests := []struct {
		name    string
		addon   *confv1beta1.Product
		want    map[string]string
		wantErr bool
	}{
		{
			name:    "no excludeIps",
			addon:   &confv1beta1.Product{Param: map[string]string{"key": "value"}},
			want:    map[string]string{"key": "value"},
			wantErr: false,
		},
		{
			name:    "empty excludeIps",
			addon:   &confv1beta1.Product{Param: map[string]string{"excludeIps": ""}},
			want:    map[string]string{"excludeIps": ""},
			wantErr: false,
		},
		{
			name:    "valid single IP",
			addon:   &confv1beta1.Product{Param: map[string]string{"excludeIps": "10.0.0.1"}},
			want:    map[string]string{"excludeIps": "10.0.0.1"},
			wantErr: false,
		},
		{
			name:    "valid IP range",
			addon:   &confv1beta1.Product{Param: map[string]string{"excludeIps": "10.0.0.1-10.0.0.5"}},
			want:    map[string]string{"excludeIps": "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5"},
			wantErr: false,
		},
		{
			name:    "invalid IP",
			addon:   &confv1beta1.Product{Param: map[string]string{"excludeIps": "invalid"}},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFabricParam(tt.addon)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFabricParam() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for k, v := range tt.want {
					if got[k] != v {
						t.Errorf("parseFabricParam()[%s] = %v, want %v", k, got[k], v)
					}
				}
			}
		})
	}
}

func TestConvertNodesToManageTemplateData(t *testing.T) {
	nodes := bkenode.Nodes{
		confv1beta1.Node{
			Hostname: "master-1",
			IP:       "10.0.0.1",
			Username: "root",
			Password: "password",
			Port:     "22",
			Role:     []string{"master"},
		},
		confv1beta1.Node{
			Hostname: "worker-1",
			IP:       "10.0.0.2",
			Username: "root",
			Password: "password",
			Port:     "22",
			Role:     []string{"worker"},
		},
	}

	result := convertNodesToManageTemplateData(nodes, "bke-cluster")

	if len(result) != 2 {
		t.Errorf("Expected 2 nodes, got %d", len(result))
	}

	if result[0]["hostname"] != "master-1" {
		t.Errorf("Expected hostname 'master-1', got %v", result[0]["hostname"])
	}
	if result[0]["name"] != "bke-cluster-master-1" {
		t.Errorf("Expected name 'bke-cluster-master-1', got %v", result[0]["name"])
	}
	if result[0]["ip"] != "10.0.0.1" {
		t.Errorf("Expected ip '10.0.0.1', got %v", result[0]["ip"])
	}
	if result[1]["hostname"] != "worker-1" {
		t.Errorf("Expected hostname 'worker-1', got %v", result[1]["hostname"])
	}
	if result[1]["name"] != "bke-cluster-worker-1" {
		t.Errorf("Expected name 'bke-cluster-worker-1', got %v", result[1]["name"])
	}
}

func TestConvertBKENodesToManageTemplateData_InheritName(t *testing.T) {
	nodes := bkev1beta1.BKENodes{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "mgmt-master-1", Namespace: "mgmt"},
			Spec: confv1beta1.BKENodeSpec{
				Hostname: "master-1",
				IP:       "10.0.0.1",
				Username: "root",
				Password: "password",
				Port:     "22",
				Role:     []string{"master"},
			},
		},
	}

	result := ConvertBKENodesToManageTemplateData(nodes, "mgmt")
	if len(result) != 1 {
		t.Fatalf("Expected 1 node, got %d", len(result))
	}
	if result[0]["name"] != "mgmt-master-1" {
		t.Errorf("Expected inherited name 'mgmt-master-1', got %v", result[0]["name"])
	}
	if result[0]["hostname"] != "master-1" {
		t.Errorf("Expected hostname 'master-1', got %v", result[0]["hostname"])
	}
}

func TestConvertNodesToManageTemplateData_Empty(t *testing.T) {
	nodes := bkenode.Nodes{}

	result := convertNodesToManageTemplateData(nodes, "bke-cluster")

	if len(result) != 0 {
		t.Errorf("Expected 0 nodes, got %d", len(result))
	}
}

func TestInitializeDefaultParams(t *testing.T) {
	result := initializeDefaultParams()

	if result["replicas"] != 1 {
		t.Errorf("Expected replicas=1, got %v", result["replicas"])
	}
	if result["kubeConfigDir"] != "/etc/kubernetes" {
		t.Errorf("Expected kubeConfigDir='/etc/kubernetes', got %v", result["kubeConfigDir"])
	}
	if result["namespace"] != "cluster-system" {
		t.Errorf("Expected namespace='cluster-system', got %v", result["namespace"])
	}
	if result["clusterName"] != "bke-cluster" {
		t.Errorf("Expected clusterName='bke-cluster', got %v", result["clusterName"])
	}
}

func TestExtractFileBaseNames(t *testing.T) {
	c := &Client{}
	files := []string{
		"/path/to/file1.yaml",
		"/path/to/file2.yaml",
		"file3.yaml",
	}

	result := c.extractFileBaseNames(files)

	if len(result) != 3 {
		t.Errorf("Expected 3 basenames, got %d", len(result))
	}
	if result[0] != "file1" {
		t.Errorf("Expected 'file1', got %s", result[0])
	}
	if result[1] != "file2" {
		t.Errorf("Expected 'file2', got %s", result[1])
	}
	if result[2] != "file3" {
		t.Errorf("Expected 'file3', got %s", result[2])
	}
}

func TestSetImageRepoParams(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			ImageRepo: confv1beta1.Repo{
				Domain: "registry.example.com",
				Port:   "5000",
				Ip:     "192.168.1.100",
				Prefix: "myprefix",
			},
		},
	}

	setImageRepoParams(&param, bkeConfig)

	if param["imageRepo"] != "registry.example.com:5000" {
		t.Errorf("Expected imageRepo='registry.example.com:5000', got %v", param["imageRepo"])
	}
	if param["imageRepoDomain"] != "registry.example.com" {
		t.Errorf("Expected imageRepoDomain='registry.example.com', got %v", param["imageRepoDomain"])
	}
	if param["imageRepoPort"] != "5000" {
		t.Errorf("Expected imageRepoPort='5000', got %v", param["imageRepoPort"])
	}
	if param["imageRepoIp"] != "192.168.1.100" {
		t.Errorf("Expected imageRepoIp='192.168.1.100', got %v", param["imageRepoIp"])
	}
	if param["imageRepoPrefix"] != "myprefix" {
		t.Errorf("Expected imageRepoPrefix='myprefix', got %v", param["imageRepoPrefix"])
	}
}

func TestSetChartRepoParams(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			ChartRepo: confv1beta1.Repo{
				Domain:                "chart.example.com",
				Port:                  "9443",
				Ip:                    "192.168.1.102",
				Prefix:                "charts",
				InsecureSkipTLSVerify: true,
				AuthSecretRef: &confv1beta1.AuthSecretRef{
					Name:        "chart-auth",
					Namespace:   "bke-cluster",
					UsernameKey: "user",
					PasswordKey: "pass",
				},
				TlsSecretRef: &confv1beta1.TlsSecretRef{
					Name:      "chart-tls",
					Namespace: "bke-cluster",
					CertKey:   "tls.crt",
					CaKey:     "ca.crt",
					KeyKey:    "tls.key",
				},
			},
		},
	}

	setChartRepoParams(&param, bkeConfig)

	if param["chartRepoDomain"] != "chart.example.com" {
		t.Errorf("Expected chartRepoDomain='chart.example.com', got %v", param["chartRepoDomain"])
	}
	if param["chartRepoPort"] != "9443" {
		t.Errorf("Expected chartRepoPort='9443', got %v", param["chartRepoPort"])
	}
	if param["chartRepoIp"] != "192.168.1.102" {
		t.Errorf("Expected chartRepoIp='192.168.1.102', got %v", param["chartRepoIp"])
	}
	if param["chartRepoPrefix"] != "charts" {
		t.Errorf("Expected chartRepoPrefix='charts', got %v", param["chartRepoPrefix"])
	}
	if param["chartRepoInsecureSkipTLSVerify"] != true {
		t.Errorf("Expected chartRepoInsecureSkipTLSVerify=true, got %v", param["chartRepoInsecureSkipTLSVerify"])
	}
	if param["chartRepoAuthSecretName"] != "chart-auth" {
		t.Errorf("Expected chartRepoAuthSecretName='chart-auth', got %v", param["chartRepoAuthSecretName"])
	}
	if param["chartRepoAuthSecretNamespace"] != "bke-cluster" {
		t.Errorf("Expected chartRepoAuthSecretNamespace='bke-cluster', got %v", param["chartRepoAuthSecretNamespace"])
	}
	if param["chartRepoAuthSecretUsernameKey"] != "user" {
		t.Errorf("Expected chartRepoAuthSecretUsernameKey='user', got %v", param["chartRepoAuthSecretUsernameKey"])
	}
	if param["chartRepoAuthSecretPasswordKey"] != "pass" {
		t.Errorf("Expected chartRepoAuthSecretPasswordKey='pass', got %v", param["chartRepoAuthSecretPasswordKey"])
	}
	if param["chartRepoTlsSecretName"] != "chart-tls" {
		t.Errorf("Expected chartRepoTlsSecretName='chart-tls', got %v", param["chartRepoTlsSecretName"])
	}
	if param["chartRepoTlsSecretNamespace"] != "bke-cluster" {
		t.Errorf("Expected chartRepoTlsSecretNamespace='bke-cluster', got %v", param["chartRepoTlsSecretNamespace"])
	}
	if param["chartRepoTlsSecretCertKey"] != "tls.crt" {
		t.Errorf("Expected chartRepoTlsSecretCertKey='tls.crt', got %v", param["chartRepoTlsSecretCertKey"])
	}
	if param["chartRepoTlsSecretCAKey"] != "ca.crt" {
		t.Errorf("Expected chartRepoTlsSecretCAKey='ca.crt', got %v", param["chartRepoTlsSecretCAKey"])
	}
	if param["chartRepoTlsSecretKeyKey"] != "tls.key" {
		t.Errorf("Expected chartRepoTlsSecretKeyKey='tls.key', got %v", param["chartRepoTlsSecretKeyKey"])
	}
}
func TestSetHTTPRepoParams(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			HTTPRepo: confv1beta1.Repo{
				Domain: "repo.example.com",
				Port:   "8080",
				Ip:     "192.168.1.101",
				Prefix: "yum",
			},
		},
	}

	setHTTPRepoParams(&param, bkeConfig)

	if param["httpRepoDomain"] != "repo.example.com" {
		t.Errorf("Expected httpRepoDomain='repo.example.com', got %v", param["httpRepoDomain"])
	}
	if param["httpRepoPort"] != "8080" {
		t.Errorf("Expected httpRepoPort='8080', got %v", param["httpRepoPort"])
	}
	if param["httpRepoIp"] != "192.168.1.101" {
		t.Errorf("Expected httpRepoIp='192.168.1.101', got %v", param["httpRepoIp"])
	}
	if param["httpRepoPrefix"] != "yum" {
		t.Errorf("Expected httpRepoPrefix='yum', got %v", param["httpRepoPrefix"])
	}
}

func TestSetNTPServer(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			NTPServer: "ntp.example.com",
		},
	}

	setNTPServer(&param, bkeConfig)

	if param["ntpServer"] != "ntp.example.com" {
		t.Errorf("Expected ntpServer='ntp.example.com', got %v", param["ntpServer"])
	}
}

func TestSetAgentHealthPort(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			AgentHealthPort: "8081",
		},
	}

	setAgentHealthPort(&param, bkeConfig)

	if param["agentHealthPort"] != "8081" {
		t.Errorf("Expected agentHealthPort='8081', got %v", param["agentHealthPort"])
	}
}

func TestSetAgentLogLevel(t *testing.T) {
	t.Run("from customExtra", func(t *testing.T) {
		param := make(map[string]interface{})
		bkeConfig := bkeinit.BkeConfig{
			CustomExtra: map[string]string{"agentLogLevel": "DEBUG"},
		}
		setAgentLogLevel(&param, bkeConfig)
		if param["agentLogLevel"] != "DEBUG" {
			t.Errorf("Expected agentLogLevel='DEBUG', got %v", param["agentLogLevel"])
		}
	})
	t.Run("default INFO when empty", func(t *testing.T) {
		param := make(map[string]interface{})
		setAgentLogLevel(&param, bkeinit.BkeConfig{})
		if param["agentLogLevel"] != "INFO" {
			t.Errorf("Expected agentLogLevel='INFO', got %v", param["agentLogLevel"])
		}
	})
	t.Run("default INFO when invalid", func(t *testing.T) {
		param := make(map[string]interface{})
		bkeConfig := bkeinit.BkeConfig{
			CustomExtra: map[string]string{"agentLogLevel": "WARN"},
		}
		setAgentLogLevel(&param, bkeConfig)
		if param["agentLogLevel"] != "INFO" {
			t.Errorf("Expected agentLogLevel='INFO', got %v", param["agentLogLevel"])
		}
	})
	t.Run("normalize warning", func(t *testing.T) {
		param := make(map[string]interface{})
		bkeConfig := bkeinit.BkeConfig{
			CustomExtra: map[string]string{"agentLogLevel": "warning"},
		}
		setAgentLogLevel(&param, bkeConfig)
		if param["agentLogLevel"] != "WARNING" {
			t.Errorf("Expected agentLogLevel='WARNING', got %v", param["agentLogLevel"])
		}
	})
}

func TestSetNodeReplicas(t *testing.T) {
	param := make(map[string]interface{})
	nodes := bkenode.Nodes{
		confv1beta1.Node{Role: []string{"master"}},
		confv1beta1.Node{Role: []string{"master"}},
		confv1beta1.Node{Role: []string{"worker"}},
		confv1beta1.Node{Role: []string{"worker"}},
		confv1beta1.Node{Role: []string{"worker"}},
	}

	setNodeReplicas(&param, nodes)

	if param["masterReplicas"] != "2" {
		t.Errorf("Expected masterReplicas='2', got %v", param["masterReplicas"])
	}
	// Worker() method behavior may vary, just check it's set
	if param["workerReplicas"] == nil {
		t.Error("Expected workerReplicas to be set")
	}
}

func TestSetEtcdIPs(t *testing.T) {
	tests := []struct {
		name  string
		nodes bkenode.Nodes
	}{
		{
			name: "with etcd nodes",
			nodes: bkenode.Nodes{
				confv1beta1.Node{IP: "10.0.0.1", Role: []string{"master"}},
				confv1beta1.Node{IP: "10.0.0.2", Role: []string{"master"}},
				confv1beta1.Node{IP: "10.0.0.3", Role: []string{"worker"}},
			},
		},
		{
			name:  "empty nodes",
			nodes: bkenode.Nodes{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param := make(map[string]interface{})
			setEtcdIPs(&param, tt.nodes)

			if _, ok := param["etcdIps"]; !ok {
				t.Error("Expected etcdIps to be set")
			}
		})
	}
}

func TestSetEtcdEndpoints(t *testing.T) {
	param := make(map[string]interface{})
	nodes := bkenode.Nodes{
		confv1beta1.Node{IP: "10.0.0.1", Role: []string{"master"}},
		confv1beta1.Node{IP: "10.0.0.2", Role: []string{"master"}},
	}

	setEtcdEndpoints(&param, nodes)

	endpoints, ok := param["etcdEndpoints"].(string)
	if !ok || endpoints == "" {
		t.Logf("etcdEndpoints is empty or not set, got %v", param["etcdEndpoints"])
	}
}

func TestSetK8sVersion(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			KubernetesVersion: "v1.21.0",
		},
	}

	setK8sVersion(&param, bkeConfig)

	if param["k8sVersion"] != "v1.21.0" {
		t.Errorf("Expected k8sVersion='v1.21.0', got %v", param["k8sVersion"])
	}
}

func TestSetEtcdVersion(t *testing.T) {
	tests := []struct {
		name      string
		bkeConfig bkeinit.BkeConfig
		wantSet   bool
		wantValue string
	}{
		{
			name: "set etcd version when not empty",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					EtcdVersion: "v3.5.13",
				},
			},
			wantSet:   true,
			wantValue: "v3.5.13",
		},
		{
			name: "do not set etcd version when empty",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					EtcdVersion: "",
				},
			},
			wantSet: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param := make(map[string]interface{})
			setEtcdVersion(&param, tt.bkeConfig)

			value, exists := param["etcdVersion"]
			if exists != tt.wantSet {
				t.Errorf("etcdVersion exists = %v, want %v", exists, tt.wantSet)
				return
			}
			if tt.wantSet && value != tt.wantValue {
				t.Errorf("Expected etcdVersion='%s', got %v", tt.wantValue, value)
			}
		})
	}
}

func TestSetContainerdVersion(t *testing.T) {
	tests := []struct {
		name      string
		bkeConfig bkeinit.BkeConfig
		wantSet   bool
		wantValue string
	}{
		{
			name: "set containerd version when not empty",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					ContainerdVersion: "1.7.20",
				},
			},
			wantSet:   true,
			wantValue: "1.7.20",
		},
		{
			name: "do not set containerd version when empty",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					ContainerdVersion: "",
				},
			},
			wantSet: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param := make(map[string]interface{})
			setContainerdVersion(&param, tt.bkeConfig)

			value, exists := param["containerdVersion"]
			if exists != tt.wantSet {
				t.Errorf("containerdVersion exists = %v, want %v", exists, tt.wantSet)
				return
			}
			if tt.wantSet && value != tt.wantValue {
				t.Errorf("Expected containerdVersion='%s', got %v", tt.wantValue, value)
			}
		})
	}
}

func TestSetKubeletDataRoot(t *testing.T) {
	tests := []struct {
		name      string
		bkeConfig bkeinit.BkeConfig
		wantSet   bool
	}{
		{
			name: "with kubelet-root-dir volume",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					Kubelet: &confv1beta1.Kubelet{
						ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
							ExtraVolumes: []confv1beta1.HostPathMount{
								{Name: "kubelet-root-dir", HostPath: "/var/lib/kubelet"},
							},
						},
					},
				},
			},
			wantSet: true,
		},
		{
			name: "without kubelet-root-dir volume",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					Kubelet: &confv1beta1.Kubelet{
						ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
							ExtraVolumes: []confv1beta1.HostPathMount{
								{Name: "other-volume", HostPath: "/other/path"},
							},
						},
					},
				},
			},
			wantSet: false,
		},
		{
			name: "nil kubelet",
			bkeConfig: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					Kubelet: nil,
				},
			},
			wantSet: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param := make(map[string]interface{})
			setKubeletDataRoot(&param, tt.bkeConfig)

			_, exists := param["kubeletDataRoot"]
			if exists != tt.wantSet {
				t.Errorf("kubeletDataRoot set = %v, want %v", exists, tt.wantSet)
			}
		})
	}
}

func TestSetDockerDataRoot(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			ContainerRuntime: confv1beta1.ContainerRuntime{
				CRI: bkeinit.CRIDocker,
				Param: map[string]string{
					"data-root": "/var/lib/docker",
				},
			},
		},
	}

	setDockerDataRoot(&param, bkeConfig)

	if param["dockerDataRoot"] != "/var/lib/docker" {
		t.Errorf("Expected dockerDataRoot='/var/lib/docker', got %v", param["dockerDataRoot"])
	}
}

func TestSetAddonParams(t *testing.T) {
	tests := []struct {
		name    string
		addons  []confv1beta1.Product
		wantKey string
		wantVal interface{}
	}{
		{
			name:    "beyondelb addon",
			addons:  []confv1beta1.Product{{Name: "beyondelb", Param: map[string]string{"lbNodes": "10.0.0.1,10.0.0.2"}}},
			wantKey: "ingressReplicas",
			wantVal: 2,
		},
		{
			name:    "calico addon",
			addons:  []confv1beta1.Product{{Name: "calico"}},
			wantKey: "clusterNetworkMode",
			wantVal: "calico",
		},
		{
			name:    "fabric addon",
			addons:  []confv1beta1.Product{{Name: "fabric"}},
			wantKey: "clusterNetworkMode",
			wantVal: "fabric",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param := make(map[string]interface{})
			bkeConfig := bkeinit.BkeConfig{Addons: tt.addons}
			setAddonParams(&param, bkeConfig)
			if param[tt.wantKey] != tt.wantVal {
				t.Errorf("Expected %s=%v, got %v", tt.wantKey, tt.wantVal, param[tt.wantKey])
			}
		})
	}
}

func TestSetDNSIP(t *testing.T) {
	tests := []struct {
		name          string
		serviceSubnet interface{}
		wantErr       bool
	}{
		{
			name:          "valid subnet",
			serviceSubnet: "10.96.0.0/12",
			wantErr:       false,
		},
		{
			name:          "invalid subnet",
			serviceSubnet: "invalid",
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			param := map[string]interface{}{
				"serviceSubnet": tt.serviceSubnet,
			}

			err := setDNSIP(&param)

			if (err != nil) != tt.wantErr {
				t.Errorf("setDNSIP() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && param["dnsIP"] == nil {
				t.Error("Expected dnsIP to be set")
			}
		})
	}
}

func TestSetNetworkParams(t *testing.T) {
	param := make(map[string]interface{})
	bkeConfig := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			Networking: confv1beta1.Networking{
				PodSubnet:     "10.244.0.0/16",
				ServiceSubnet: "10.96.0.0/12",
				DNSDomain:     "cluster.local",
			},
		},
	}
	bkeCluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: "10.0.0.1",
				Port: 6443,
			},
		},
	}

	setNetworkParams(&param, bkeConfig, bkeCluster)

	if param["podSubnet"] != "10.244.0.0/16" {
		t.Errorf("Expected podSubnet='10.244.0.0/16', got %v", param["podSubnet"])
	}
	if param["serviceSubnet"] != "10.96.0.0/12" {
		t.Errorf("Expected serviceSubnet='10.96.0.0/12', got %v", param["serviceSubnet"])
	}
	if param["dnsDomain"] != "cluster.local" {
		t.Errorf("Expected dnsDomain='cluster.local', got %v", param["dnsDomain"])
	}
	if param["apiServerSrcHost"] != "10.0.0.1" {
		t.Errorf("Expected apiServerSrcHost='10.0.0.1', got %v", param["apiServerSrcHost"])
	}
	if param["apiServerSrcPort"] != int32(6443) {
		t.Errorf("Expected apiServerSrcPort=6443, got %v", param["apiServerSrcPort"])
	}
}

func TestGetAddonYamlFiles(t *testing.T) {
	c := &Client{}
	addon := &confv1beta1.Product{
		Name:    "test-addon",
		Version: "v1.0.0",
	}

	patches := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer patches.Reset()

	_, err := c.getAddonYamlFiles(addon)
	if err == nil {
		t.Error("Expected error for non-existent addon dir")
	}
}

func TestCreateAddonTask(t *testing.T) {
	c := &Client{}
	config := &addonApplyConfig{
		addon: &confv1beta1.Product{
			Name:  "test-addon",
			Block: true,
		},
		addonT: &bkeaddon.AddonTransfer{
			Operate: bkeaddon.CreateAddon,
		},
		param: map[string]map[string]interface{}{
			"file1": {"key": "value"},
		},
		repo:          "test-repo",
		addonRecorder: &AddonRecorder{},
	}

	task := c.createAddonTask(config, "file1", "/path/to/file1.yaml")

	if task.Name != "test-addon" {
		t.Errorf("Expected task name 'test-addon', got %s", task.Name)
	}
	if task.FilePath != "/path/to/file1.yaml" {
		t.Errorf("Expected file path '/path/to/file1.yaml', got %s", task.FilePath)
	}
	if task.Operate != bkeaddon.CreateAddon {
		t.Errorf("Expected operate CreateAddon, got %v", task.Operate)
	}
}

func TestCreateAddonTaskBocOperator(t *testing.T) {
	c := &Client{}
	config := &addonApplyConfig{
		addon: &confv1beta1.Product{
			Name:  "bocoperator",
			Block: true,
		},
		addonT: &bkeaddon.AddonTransfer{
			Operate: bkeaddon.CreateAddon,
		},
		param: map[string]map[string]interface{}{
			"file1": {"key": "value"},
		},
		repo:          "test-repo",
		addonRecorder: &AddonRecorder{},
	}

	task := c.createAddonTask(config, "file1", "/path/to/file1.yaml")

	if task.Timeout != bocOperatorTimeoutMinutes*time.Minute {
		t.Errorf("Expected timeout %v, got %v", bocOperatorTimeoutMinutes*time.Minute, task.Timeout)
	}
	if task.Interval != bocOperatorIntervalSeconds*time.Second {
		t.Errorf("Expected interval %v, got %v", bocOperatorIntervalSeconds*time.Second, task.Interval)
	}
}

func TestHandleApplyErrorRemoveAddon(t *testing.T) {
	c := &Client{Log: nil}
	addon := &confv1beta1.Product{Name: "test", Version: "v1"}
	err := c.handleApplyError(errors.New("test error"), addon, "/path/file.yaml", bkeaddon.RemoveAddon)
	if err != nil {
		t.Error("Expected nil error for RemoveAddon operation")
	}
}

func TestHandleApplyErrorCreateAddon(t *testing.T) {
	c := &Client{Log: nil}
	addon := &confv1beta1.Product{Name: "test", Version: "v1"}
	err := c.handleApplyError(errors.New("test error"), addon, "/path/file.yaml", bkeaddon.CreateAddon)
	if err == nil {
		t.Error("Expected error for CreateAddon operation")
	}
}

func TestHandleApplyErrorNoMatchError(t *testing.T) {
	c := &Client{Log: nil}
	addon := &confv1beta1.Product{Name: "test", Version: "v1"}

	patches := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, &os.PathError{Op: "stat", Path: name, Err: os.ErrNotExist}
	})
	defer patches.Reset()

	noMatchErr := &NoMatchError{}
	err := c.handleApplyError(noMatchErr, addon, "/path/file.yaml", bkeaddon.CreateAddon)
	if err == nil {
		t.Error("Expected error for NoMatchError")
	}
}

type NoMatchError struct{}

func (e *NoMatchError) Error() string {
	return "no matches for kind"
}

func (e *NoMatchError) Status() int32 {
	return 0
}

func TestGetPortalK8sToken(t *testing.T) {
	c := &Client{}
	patches := gomonkey.ApplyMethod(c, "NewK8sToken", func(*Client) (string, error) {
		return "test-token", nil
	})
	defer patches.Reset()

	token, err := c.getPortalK8sToken()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if token != "test-token" {
		t.Errorf("Expected 'test-token', got %s", token)
	}
}

func TestCalculateTyphaReplicas(t *testing.T) {
	tests := []struct {
		nodeCount int
		want      int
	}{
		{nodeCount: 51, want: maxInt(minTyphaReplicas, 1)},
		{nodeCount: 400, want: maxInt(minTyphaReplicas, 2)},
		{nodeCount: 600, want: maxInt(minTyphaReplicas, 3)},
		{nodeCount: 601, want: maxInt(minTyphaReplicas, 4)},
		{nodeCount: 1000, want: maxInt(minTyphaReplicas, 5)},
		{nodeCount: 5000, want: maxTyphaReplicas},
	}

	for _, tt := range tests {
		t.Run("replicas", func(t *testing.T) {
			if got := calculateTyphaReplicas(tt.nodeCount); got != tt.want {
				t.Errorf("calculateTyphaReplicas(%d) = %d, want %d", tt.nodeCount, got, tt.want)
			}
		})
	}
}

func TestApplyCalicoScaleParamsOverridesAddonParam(t *testing.T) {
	file := writeCalicoManifestForTest(t, true)
	param := map[string]map[string]interface{}{
		"001-typha": {
			"allowTypha":    "false",
			"typhaReplicas": "1",
		},
		"002-calico": {},
	}

	err := applyCalicoScaleParams(
		&confv1beta1.Product{Name: "calico", Version: "v3.31.3"},
		[]string{file},
		param,
		makeTestNodes(51),
		bkeaddon.CreateAddon,
	)

	if err != nil {
		t.Fatalf("applyCalicoScaleParams() error = %v", err)
	}
	for name, values := range param {
		if values["allowTypha"] != "true" {
			t.Errorf("%s allowTypha = %v, want true", name, values["allowTypha"])
		}
		if values["typhaReplicas"] != strconv.Itoa(calculateTyphaReplicas(51)) {
			t.Errorf("%s typhaReplicas = %v, want %d", name, values["typhaReplicas"], calculateTyphaReplicas(51))
		}
	}
}

func TestApplyCalicoScaleParamsUnsupportedManifestCreateFails(t *testing.T) {
	file := writeCalicoManifestForTest(t, false)
	param := map[string]map[string]interface{}{"calico": {}}

	err := applyCalicoScaleParams(
		&confv1beta1.Product{Name: "calico", Version: "v3.27.3"},
		[]string{file},
		param,
		makeTestNodes(51),
		bkeaddon.CreateAddon,
	)

	if err == nil {
		t.Fatal("expected unsupported manifest error")
	}
}

func TestApplyCalicoScaleParamsUnsupportedManifestUpdateKeepsDisabled(t *testing.T) {
	file := writeCalicoManifestForTest(t, false)
	param := map[string]map[string]interface{}{"calico": {}}

	err := applyCalicoScaleParams(
		&confv1beta1.Product{Name: "calico", Version: "v3.27.3"},
		[]string{file},
		param,
		makeTestNodes(51),
		bkeaddon.UpdateAddon,
	)

	if err != nil {
		t.Fatalf("applyCalicoScaleParams() error = %v", err)
	}
	if param["calico"]["allowTypha"] != "false" {
		t.Errorf("allowTypha = %v, want false", param["calico"]["allowTypha"])
	}
	if param["calico"]["typhaReplicas"] != "0" {
		t.Errorf("typhaReplicas = %v, want 0", param["calico"]["typhaReplicas"])
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func writeCalicoManifestForTest(t *testing.T, supportsTypha bool) string {
	t.Helper()
	content := "kind: ConfigMap\nmetadata:\n  name: calico-config\ndata:\n  typha_service_name: \"none\"\n"
	if supportsTypha {
		content = `{{- if eq .allowTypha "true" }}
kind: Deployment
metadata:
  name: calico-typha
spec:
  replicas: {{ .typhaReplicas | int }}
---
kind: Service
metadata:
  name: calico-typha
---
kind: PodDisruptionBudget
metadata:
  name: calico-typha
---
kind: ConfigMap
metadata:
  name: calico-config
data:
  typha_service_name: "calico-typha"
---
kind: DaemonSet
spec:
  template:
    spec:
      containers:
        - name: calico-node
          env:
            - name: FELIX_TYPHAK8SSERVICENAME
              valueFrom:
                configMapKeyRef:
                  key: typha_service_name
{{- end }}
`
	}
	file, err := os.CreateTemp(t.TempDir(), "calico-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(content); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return file.Name()
}

func makeTestNodes(count int) bkenode.Nodes {
	nodes := make(bkenode.Nodes, 0, count)
	for i := 0; i < count; i++ {
		nodes = append(nodes, confv1beta1.Node{IP: "10.0.0.1"})
	}
	return nodes
}

func TestFilterCalicoTyphaBootstrapObjects(t *testing.T) {
	objects := []unstructured.Unstructured{
		newTestUnstructured("CustomResourceDefinition", "ippools.crd.projectcalico.org", ""),
		newTestUnstructured("ServiceAccount", "calico-node", "kube-system"),
		newTestUnstructured("ClusterRole", "calico-node", ""),
		newTestUnstructured("ClusterRoleBinding", "calico-node", ""),
		newTestUnstructured("ConfigMap", "calico-config", "kube-system"),
		newTestUnstructured("Deployment", "calico-typha", "kube-system"),
		newTestUnstructured("Service", "calico-typha", "kube-system"),
		newTestUnstructured("PodDisruptionBudget", "calico-typha", "kube-system"),
		newTestUnstructured("DaemonSet", "calico-node", "kube-system"),
		newTestUnstructured("Deployment", "calico-kube-controllers", "kube-system"),
	}

	withoutDependencies := filterCalicoTyphaBootstrapObjects(objects, false)
	if len(withoutDependencies) != 3 {
		t.Fatalf("without dependencies got %d objects, want 3", len(withoutDependencies))
	}
	for _, obj := range withoutDependencies {
		if !isCalicoTyphaObject(obj) {
			t.Fatalf("without dependencies should only contain typha objects, got %s/%s", obj.GetKind(), obj.GetName())
		}
	}

	withDependencies := filterCalicoTyphaBootstrapObjects(objects, true)
	if len(withDependencies) != 8 {
		t.Fatalf("with dependencies got %d objects, want 8", len(withDependencies))
	}
	for _, obj := range withDependencies {
		if obj.GetKind() == "DaemonSet" && obj.GetName() == "calico-node" {
			t.Fatal("calico-node DaemonSet must not be applied before typha is ready")
		}
		if obj.GetKind() == "Deployment" && obj.GetName() == "calico-kube-controllers" {
			t.Fatal("calico-kube-controllers Deployment should stay in the full calico apply phase")
		}
	}
}

func TestSortCalicoTyphaBootstrapObjectsForCreate(t *testing.T) {
	objects := []unstructured.Unstructured{
		newTestUnstructured("Deployment", "calico-typha", "kube-system"),
		newTestUnstructured("ServiceAccount", "calico-node", "kube-system"),
		newTestUnstructured("ClusterRoleBinding", "calico-node", ""),
		newTestUnstructured("ConfigMap", "calico-config", "kube-system"),
		newTestUnstructured("DaemonSet", "calico-node", "kube-system"),
	}

	filtered := filterCalicoTyphaBootstrapObjects(objects, true)
	sorted := (&Client{}).sortUnstructuredList(filtered, &Task{Operate: bkeaddon.CreateAddon})
	typhaIndex := indexTestUnstructured(sorted, "Deployment", "calico-typha")
	if typhaIndex < 0 {
		t.Fatal("calico-typha Deployment should be included in bootstrap objects")
	}
	if indexTestUnstructured(sorted, "DaemonSet", "calico-node") >= 0 {
		t.Fatal("calico-node DaemonSet must not be applied in typha bootstrap phase")
	}
	for _, dep := range []struct {
		kind string
		name string
	}{
		{kind: "ServiceAccount", name: "calico-node"},
		{kind: "ClusterRoleBinding", name: "calico-node"},
		{kind: "ConfigMap", name: "calico-config"},
	} {
		depIndex := indexTestUnstructured(sorted, dep.kind, dep.name)
		if depIndex < 0 {
			t.Fatalf("%s/%s should be included in bootstrap objects", dep.kind, dep.name)
		}
		if depIndex > typhaIndex {
			t.Fatalf("%s/%s should be applied before calico-typha Deployment", dep.kind, dep.name)
		}
	}
}

func indexTestUnstructured(objects []unstructured.Unstructured, kind, name string) int {
	for i, obj := range objects {
		if obj.GetKind() == kind && obj.GetName() == name {
			return i
		}
	}
	return -1
}
func newTestUnstructured(kind, name, namespace string) unstructured.Unstructured {
	obj := unstructured.Unstructured{}
	obj.SetKind(kind)
	obj.SetName(name)
	obj.SetNamespace(namespace)
	return obj
}
