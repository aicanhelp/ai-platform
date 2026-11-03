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
package v1

import (
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestContainerCheckpointDeepCopy(t *testing.T) {
	in := &ContainerCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "ck", Namespace: "ns"},
		Spec: ContainerCheckpointSpec{
			Namespace:     "ns",
			PodName:       "pod",
			ContainerName: "c",
		},
		Status: ContainerCheckpointStatus{Version: "v1", TotalSamplesCount: 3},
	}
	out := in.DeepCopy()
	assert.NotNil(t, out)
	assert.Equal(t, in.Spec, out.Spec)
	assert.Equal(t, in.Status.Version, out.Status.Version)

	// Mutating the copy must not affect the original (deep copy independence).
	out.Spec.PodName = "other"
	assert.Equal(t, "pod", in.Spec.PodName)
}

func TestContainerCheckpointDeepCopyNil(t *testing.T) {
	var in *ContainerCheckpoint
	assert.Nil(t, in.DeepCopy())
	assert.Nil(t, in.DeepCopyObject())
}

func TestContainerCheckpointDeepCopyObject(t *testing.T) {
	in := &ContainerCheckpoint{Spec: ContainerCheckpointSpec{PodName: "p"}}
	assert.NotNil(t, in.DeepCopyObject())
}

func TestContainerCheckpointDeepCopyInto(t *testing.T) {
	in := &ContainerCheckpoint{Spec: ContainerCheckpointSpec{PodName: "p"}}
	out := &ContainerCheckpoint{}
	in.DeepCopyInto(out)
	assert.Equal(t, in.Spec.PodName, out.Spec.PodName)
}

func TestContainerCheckpointSpecDeepCopy(t *testing.T) {
	in := &ContainerCheckpointSpec{PodName: "p", Namespace: "ns"}
	assert.Equal(t, in, in.DeepCopy())
	var nilSpec *ContainerCheckpointSpec
	assert.Nil(t, nilSpec.DeepCopy())
}

func TestContainerCheckpointStatusDeepCopy(t *testing.T) {
	in := &ContainerCheckpointStatus{Version: "v", TotalSamplesCount: 2}
	out := in.DeepCopy()
	assert.Equal(t, in.Version, out.Version)
	var nilStatus *ContainerCheckpointStatus
	assert.Nil(t, nilStatus.DeepCopy())
}

func TestContainerCheckpointListDeepCopyWithItems(t *testing.T) {
	in := &ContainerCheckpointList{
		Items: []ContainerCheckpoint{
			{Spec: ContainerCheckpointSpec{PodName: "p1"}},
			{Spec: ContainerCheckpointSpec{PodName: "p2"}},
		},
	}
	out := in.DeepCopy()
	assert.Equal(t, len(in.Items), len(out.Items))
	assert.Equal(t, "p1", out.Items[0].Spec.PodName)
	assert.Equal(t, "p2", out.Items[1].Spec.PodName)
}

func TestContainerCheckpointListDeepCopyNilItems(t *testing.T) {
	empty := &ContainerCheckpointList{}
	out := empty.DeepCopy()
	assert.NotNil(t, out)
	assert.Nil(t, out.Items)
}

func TestContainerCheckpointListDeepCopyNil(t *testing.T) {
	var in *ContainerCheckpointList
	assert.Nil(t, in.DeepCopy())
	assert.Nil(t, in.DeepCopyObject())
}

func TestContainerCheckpointListDeepCopyObject(t *testing.T) {
	in := &ContainerCheckpointList{Items: []ContainerCheckpoint{{}}}
	assert.NotNil(t, in.DeepCopyObject())
}
