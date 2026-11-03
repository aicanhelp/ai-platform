/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package imagehelper

import (
	"os"
	"testing"
)

func TestImageHelper(t *testing.T) {
	t.Run("GetFullImageName", TestGetFullImageName)
	t.Run("GetImageNameWithTag", TestGetImageNameWithTag)
}

func TestGetImageNameWithTag(t *testing.T) {
	tests := []struct {
		name  string
		image string
		tag   string
		want  string
	}{
		{
			name:  "dirty image name tag",
			image: "/kube-apiserver",
			tag:   ":v1.21.1",
			want:  "kube-apiserver:v1.21.1",
		},
		{
			name:  "image name without tag",
			image: "/kube-apiserver:",
			tag:   ":v1.21.1",
			want:  "kube-apiserver:v1.21.1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetImageNameWithTag(tt.image, tt.tag); got != tt.want {
				t.Errorf("GetImageNameWithTag() = %v, want %v", got, tt.want)
			}
		})

	}
}

func TestGetFullImageName(t *testing.T) {
	tests := []struct {
		name  string
		repo  string
		image string
		tag   string
		want  string
	}{
		{
			name:  "dirty repo image tag",
			repo:  "registry.cn-hangzhou.aliyuncs.com/k8s/",
			image: "/kube-apiserver:",
			tag:   ":v1.21.1",
			want:  "registry.cn-hangzhou.aliyuncs.com/k8s/kube-apiserver:v1.21.1",
		},
		{
			name:  "dirty repo image tag",
			repo:  "registry.cn-hangzhou.aliyuncs.com/k8s",
			image: "/kube-apiserver",
			tag:   ":v1.21.1",
			want:  "registry.cn-hangzhou.aliyuncs.com/k8s/kube-apiserver:v1.21.1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetFullImageName(tt.repo, tt.image, tt.tag); got != tt.want {
				t.Errorf("GetFullImageName() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestRepoJoinImageName(t *testing.T) {
	t.Run("RepoJoinImageName", func(t *testing.T) {
		RepoJoinImageName(os.TempDir(), "ut")
	})
}

func TestHelper(t *testing.T) {
	t.Run("imageMapJoinRepo", func(t *testing.T) {
		imageMapJoinRepo("cr.openfuyao.cn/openfuyao",
			map[string]string{"kube-apiserver": "v1.29.1", "etcd": "etcd:3.5.6-0"})
	})
}
