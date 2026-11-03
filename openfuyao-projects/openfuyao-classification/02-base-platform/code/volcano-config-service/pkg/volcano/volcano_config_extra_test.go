/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package volcano

import (
	"context"
	"net/http"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"

	"volcano-config-service/pkg/constant"
	"volcano-config-service/pkg/utils/k8sutil"
	"volcano-config-service/pkg/zlog"
)

func TestMethodBalanceGet(t *testing.T) {
	v := newTestVolcanoClient()

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	configContent := `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority
  - name: balance`
	configMap.Data = make(map[string]string)
	configMap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	_, err := k8sutil.CreateConfigMap(v.clientset, configMap)
	if err != nil {
		zlog.Debugf("Failed to create configMap, err: %v", err)
		return
	}

	_, code := v.MethodBalanceGet()
	if code != http.StatusOK {
		t.Errorf("Expected OK, got %d", code)
	}
}

func TestMethodBalancePut(t *testing.T) {
	v := newTestVolcanoClient()

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	configContent := `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`
	configMap.Data = make(map[string]string)
	configMap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	_, err := k8sutil.CreateConfigMap(v.clientset, configMap)
	if err != nil {
		zlog.Debugf("Failed to create configMap, err: %v", err)
		return
	}

	_, code := v.MethodBalancePut("open")
	if code != http.StatusOK {
		t.Errorf("Expected OK, got %d", code)
	}
}

func TestMethodNumaDistanceGet_NoConfig(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodNumaDistanceGet()
	// Without NodeConfig, should return error status
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodNumaDistancePut_NoConfig(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodNumaDistancePut("open")
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodNumaFastGet_NoConfig(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodNumaFastGet()
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodNumaFastPut_NoConfig(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodNumaFastPut("open")
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodCRInfo(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodCRInfo("test-cr")
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodUuidToName(t *testing.T) {
	v := newTestVolcanoClient()

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			UID:       types.UID("test-uid"),
		},
	}
	_, err := v.clientset.CoreV1().Pods("default").Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		zlog.Debugf("Failed to create pod, err: %v", err)
	}

	_, code := v.MethodUuidToName("test-uid")
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodNumaAwareGet_NoConfig(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodNumaAwareGet()
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestMethodNumaAwarePut_NoConfig(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := v.MethodNumaAwarePut("close")
	if code == 0 {
		t.Errorf("Expected non-zero status, got %d", code)
	}
}

func TestCheckKubeletSupport_NoNodes(t *testing.T) {
	v := newTestVolcanoClient()
	_, _, err := checkKubeletSupport(v.clientset)
	if err == nil {
		t.Errorf("Expected error with no nodes")
	}
}

func TestGetMethodPut_InvalidStatus(t *testing.T) {
	v := newTestVolcanoClient()

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	configContent := `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`
	configMap.Data = make(map[string]string)
	configMap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	_, err := k8sutil.CreateConfigMap(v.clientset, configMap)
	if err != nil {
		return
	}

	_, code := methodPut(v.clientset, "binpack", "invalid_status")
	if code != http.StatusInternalServerError {
		t.Errorf("Expected internal server error for invalid status, got %d", code)
	}
}

func TestGetPriorityPut_InvalidStatus(t *testing.T) {
	v := newTestVolcanoClient()

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	configContent := `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`
	configMap.Data = make(map[string]string)
	configMap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	_, err := k8sutil.CreateConfigMap(v.clientset, configMap)
	if err != nil {
		return
	}

	_, code := priorityPut(v.clientset, "invalid_status")
	if code != http.StatusInternalServerError {
		t.Errorf("Expected internal server error for invalid status, got %d", code)
	}
}

func TestGetGangPut_InvalidStatus(t *testing.T) {
	v := newTestVolcanoClient()

	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	configContent := `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`
	configMap.Data = make(map[string]string)
	configMap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	_, err := k8sutil.CreateConfigMap(v.clientset, configMap)
	if err != nil {
		return
	}

	_, code := gangPut(v.clientset, "invalid_status")
	if code != http.StatusInternalServerError {
		t.Errorf("Expected internal server error for invalid status, got %d", code)
	}
}

func TestGetMethodGet_NoConfigMap(t *testing.T) {
	v := newTestVolcanoClient()
	_, code := methodGet(v.clientset, "binpack")
	if code != http.StatusInternalServerError {
		t.Errorf("Expected internal server error, got %d", code)
	}
}

func TestNewVolcanoOperation(t *testing.T) {
	cfg := &rest.Config{Host: "https://example.com"}
	op, err := NewVolcanoOperation(cfg)
	assert.NoError(t, err)
	assert.NotNil(t, op)
}

func TestConfigVolcano_Disabled(t *testing.T) {
	t.Setenv("VCS_API_DISABLED", "true")
	container := restful.NewContainer()
	err := ConfigVolcano(container, &rest.Config{Host: "https://example.com"})
	assert.NoError(t, err)
}

func TestConfigVolcano_Disabled_One(t *testing.T) {
	t.Setenv("VCS_API_DISABLED", "1")
	container := restful.NewContainer()
	err := ConfigVolcano(container, &rest.Config{Host: "https://example.com"})
	assert.NoError(t, err)
}

func TestConfigVolcano_FullPath(t *testing.T) {
	t.Setenv("VCS_API_DISABLED", "")
	container := restful.NewContainer()
	err := ConfigVolcano(container, &rest.Config{Host: "https://example.com"})
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(container.RegisteredWebServices()), 1)
}
