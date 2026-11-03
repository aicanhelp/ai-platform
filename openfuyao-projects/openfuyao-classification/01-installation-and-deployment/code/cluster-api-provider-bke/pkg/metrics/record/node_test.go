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

package record

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkemetrics "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
)

func TestNodeBootstrapFailedCountRecord(t *testing.T) {
	tests := []struct {
		name        string
		metricsAddr string
	}{
		{
			name:        "metrics disabled",
			metricsAddr: "0",
		},
		{
			name:        "metrics enabled",
			metricsAddr: ":8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldAddr := config.MetricsAddr
			defer func() {
				config.MetricsAddr = oldAddr
			}()

			config.MetricsAddr = tt.metricsAddr

			obj := &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-cluster",
					Namespace: "default",
				},
			}

			if tt.metricsAddr != "0" {
				bkemetrics.DefaultRegister().Register("default/test-cluster")
				defer bkemetrics.DefaultRegister().Unregister("default/test-cluster")
			}

			NodeBootstrapFailedCountRecord(obj)
		})
	}
}

func TestNodeBootstrapSuccessCountRecord(t *testing.T) {
	tests := []struct {
		name        string
		metricsAddr string
	}{
		{
			name:        "metrics disabled",
			metricsAddr: "0",
		},
		{
			name:        "metrics enabled",
			metricsAddr: ":8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldAddr := config.MetricsAddr
			defer func() {
				config.MetricsAddr = oldAddr
			}()

			config.MetricsAddr = tt.metricsAddr

			obj := &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-cluster",
					Namespace: "default",
				},
			}

			if tt.metricsAddr != "0" {
				bkemetrics.DefaultRegister().Register("default/test-cluster")
				defer bkemetrics.DefaultRegister().Unregister("default/test-cluster")
			}

			NodeBootstrapSuccessCountRecord(obj)
		})
	}
}

func TestNodeBootstrapDurationRecord(t *testing.T) {
	tests := []struct {
		name        string
		metricsAddr string
	}{
		{
			name:        "metrics disabled",
			metricsAddr: "0",
		},
		{
			name:        "metrics enabled",
			metricsAddr: ":8080",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldAddr := config.MetricsAddr
			defer func() {
				config.MetricsAddr = oldAddr
			}()

			config.MetricsAddr = tt.metricsAddr

			obj := &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-cluster",
					Namespace: "default",
				},
			}

			if tt.metricsAddr != "0" {
				bkemetrics.DefaultRegister().Register("default/test-cluster")
				defer bkemetrics.DefaultRegister().Unregister("default/test-cluster")
			}

			node := confv1beta1.Node{
				IP:   "192.168.1.1",
				Role: []string{"master"},
			}

			startTime := time.Now().Add(-5 * time.Second)
			NodeBootstrapDurationRecord(obj, node, startTime, "test info")
		})
	}
}
