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

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sfake "k8s.io/client-go/kubernetes/fake"

	"volcano-config-service/pkg/constant"
	"volcano-config-service/pkg/utils/k8sutil"
)

func fakeClientsetWithConfigMap(content string) *k8sfake.Clientset {
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	cm.Data = make(map[string]string)
	cm.Data[constant.VolcanoConfigServiceConfigmapName] = content
	return k8sfake.NewSimpleClientset(cm)
}

func fakeClientsetWithConfigMapAndPod(content string) *k8sfake.Clientset {
	cs := fakeClientsetWithConfigMap(content)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "volcano-scheduler-0", Namespace: constant.VolcanoConfigServiceDefaultNamespace},
	}
	_, _ = cs.CoreV1().Pods(constant.VolcanoConfigServiceDefaultNamespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	return cs
}

func TestFakeMethodGet_Binpack(t *testing.T) {
	cs := fakeClientsetWithConfigMap(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority
  - name: binpack`)
	_, code := methodGet(cs, "binpack")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeMethodGet_NotFound(t *testing.T) {
	cs := fakeClientsetWithConfigMap(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`)
	_, code := methodGet(cs, "binpack")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeMethodPut_Open(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`)
	_, code := methodPut(cs, "binpack", "open")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeMethodPut_Close(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority
  - name: binpack`)
	_, code := methodPut(cs, "binpack", "close")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeMethodPut_InvalidStatus(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`)
	_, code := methodPut(cs, "binpack", "invalid")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakePriorityGet(t *testing.T) {
	cs := fakeClientsetWithConfigMap(`actions: "allocate, backfill, preempt"
tiers:
- plugins:
  - name: priority
  - name: gang
    enablePreemptable: false
    enableJobStarving: false`)
	_, code := priorityGet(cs)
	assert.Equal(t, http.StatusOK, code)
}

func TestFakePriorityPut_Open(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`)
	_, code := priorityPut(cs, "open")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakePriorityPut_Close(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "allocate, backfill, preempt"
tiers:
- plugins:
  - name: priority
  - name: gang
    enablePreemptable: false
    enableJobStarving: false`)
	_, code := priorityPut(cs, "close")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeGangPut_Open(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`)
	_, code := gangPut(cs, "open")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeGangPut_Close(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority
  - name: gang
    enablePreemptable: false
    enableJobStarving: false`)
	_, code := gangPut(cs, "close")
	assert.Equal(t, http.StatusOK, code)
}

func TestFakeGangPut_InvalidStatus(t *testing.T) {
	cs := fakeClientsetWithConfigMapAndPod(`actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority`)
	_, code := gangPut(cs, "invalid")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakeMethodGet_NoConfigMap(t *testing.T) {
	cs := k8sfake.NewSimpleClientset()
	_, code := methodGet(cs, "binpack")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakeMethodPut_NoConfigMap(t *testing.T) {
	cs := k8sfake.NewSimpleClientset()
	_, code := methodPut(cs, "binpack", "open")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakeMethodGet_EmptyContent(t *testing.T) {
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	cm.Data = make(map[string]string)
	cs := k8sfake.NewSimpleClientset(cm)
	_, code := methodGet(cs, "binpack")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakeMethodPut_EmptyContent(t *testing.T) {
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constant.VolcanoConfigServiceConfigmap,
			Namespace: constant.VolcanoConfigServiceDefaultNamespace,
		},
	}
	cm.Data = make(map[string]string)
	cs := k8sfake.NewSimpleClientset(cm)
	_, code := methodPut(cs, "binpack", "open")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakePriorityGet_NoConfigMap(t *testing.T) {
	cs := k8sfake.NewSimpleClientset()
	_, code := priorityGet(cs)
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakePriorityPut_NoConfigMap(t *testing.T) {
	cs := k8sfake.NewSimpleClientset()
	_, code := priorityPut(cs, "open")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakeGangPut_NoConfigMap(t *testing.T) {
	cs := k8sfake.NewSimpleClientset()
	_, code := gangPut(cs, "open")
	assert.Equal(t, http.StatusInternalServerError, code)
}

func TestFakeK8sutilGetConfigMap(t *testing.T) {
	cs := fakeClientsetWithConfigMap("test")
	cm, err := k8sutil.GetConfigMap(cs, constant.VolcanoConfigServiceConfigmap, constant.VolcanoConfigServiceDefaultNamespace)
	assert.NoError(t, err)
	assert.NotNil(t, cm)
}
