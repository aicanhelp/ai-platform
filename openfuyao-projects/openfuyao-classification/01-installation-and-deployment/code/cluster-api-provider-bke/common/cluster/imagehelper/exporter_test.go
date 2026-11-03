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
package imagehelper

import (
	"reflect"
	"testing"

	common "gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
)

var mString = map[string]string{
	"etcd":                    "cr.openfuyao.cn/openfuyao/kubernetes/etcd:" + versions.EtcdImageTag(),
	"kube-apiserver":          "cr.openfuyao.cn/openfuyao/kubernetes/kube-apiserver:1.29.1-of.1",
	"kube-controller-manager": "cr.openfuyao.cn/openfuyao/kubernetes/kube-controller-manager:1.29.1-of.1",
	"kube-scheduler":          "cr.openfuyao.cn/openfuyao/kubernetes/kube-scheduler:1.29.1-of.1",
}

func TestNewImageExporter(t *testing.T) {
	t.Run("NewImageExporter", func(t *testing.T) {
		NewImageExporter("k8s.gcr.io", "v1.21.1", "")
	})
}

func TestImageExporter_ExportImageMap(t *testing.T) {
	tests := []struct {
		name     string
		exporter *ImageExporter
		want     map[string]string
		wantErr  bool
	}{
		{
			name:     "good exporter",
			exporter: NewImageExporter("cr.openfuyao.cn/openfuyao/kubernetes", "1.29.1-of.1", ""),
			want:     mString,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.exporter.ExportImageMap()
			if (err != nil) != tt.wantErr {
				t.Errorf("ImageExporter.ExportImageMap() error = %v wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ImageExporter.ExportImageMap() = %v \n want %v", got, tt.want)
			}
		})
	}

}

func TestImageExporter_ExportImageList(t *testing.T) {
	tests := []struct {
		name     string
		exporter *ImageExporter
		want     []string
		wantErr  bool
	}{
		{
			name:     "good exporter",
			exporter: NewImageExporter("cr.openfuyao.cn/openfuyao/kubernetes", "1.29.1-of.1", ""),
			want: []string{
				"cr.openfuyao.cn/openfuyao/kubernetes/kube-apiserver:1.29.1-of.1",
				"cr.openfuyao.cn/openfuyao/kubernetes/kube-controller-manager:1.29.1-of.1",
				"cr.openfuyao.cn/openfuyao/kubernetes/kube-scheduler:1.29.1-of.1",
				"cr.openfuyao.cn/openfuyao/kubernetes/etcd:" + versions.EtcdImageTag(),
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.exporter.ExportImageList()
			if (err != nil) != tt.wantErr {
				t.Errorf("ImageExporter.ExportImageList() error = %v wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) != len(tt.want) {
				t.Errorf("\nImageExporter.ExportImageList() = %v \nwant %v", got, tt.want)
				return
			}

			for _, v := range got {
				if !utils.SliceContainsString(tt.want, v) {
					t.Errorf("\nImageExporter.ExportImageList() = %v \nwant %v", got, tt.want)
				}
			}
		})
	}

}

func TestImageExporter_ExportImageMapWithBootStrapPhase(t *testing.T) {
	tests := []struct {
		name     string
		phase    string
		exporter *ImageExporter
		want     map[string]string
		wantErr  bool
	}{
		{
			name:     "init control plane 1.21.1",
			exporter: NewImageExporter("cr.openfuyao.cn/openfuyao/kubernetes", "1.29.1-of.1", ""),
			phase:    common.InitControlPlane,
			want:     mString,
			wantErr:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.exporter.ExportImageMapWithBootStrapPhase(tt.phase)
			if (err != nil) != tt.wantErr {
				t.Errorf("ImageExporter.ExportImageMapWithBootStrapPhase() error = %v wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ImageExporter.ExportImageMapWithBootStrapPhase() = %v \n want %v", got, tt.want)
			}
		})
	}
}
