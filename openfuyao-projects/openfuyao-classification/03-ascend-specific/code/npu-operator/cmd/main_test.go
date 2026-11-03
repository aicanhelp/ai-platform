// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package main

import (
	"flag"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

func TestInitManagerOptions(t *testing.T) {
	tests := []struct {
		name               string
		enableHTTP2        bool
		secureMetrics      bool
		expectTLSOpts      bool
		expectFilterConfig bool
	}{
		{
			name:               "disable http2 and secure metrics",
			enableHTTP2:        false,
			secureMetrics:      true,
			expectTLSOpts:      true,
			expectFilterConfig: true,
		},
		{
			name:               "enable http2 and plain metrics",
			enableHTTP2:        true,
			secureMetrics:      false,
			expectTLSOpts:      false,
			expectFilterConfig: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originArgs := os.Args
			originFlags := flag.CommandLine
			defer func() {
				os.Args = originArgs
				flag.CommandLine = originFlags
			}()
			os.Args = []string{"npu-operator"}
			flag.CommandLine = flag.NewFlagSet("npu-operator-test", flag.ContinueOnError)

			var capturedOptions ctrl.Options
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			patches.ApplyFunc(ctrl.GetConfigOrDie, func() *rest.Config {
				return &rest.Config{Host: "https://127.0.0.1"}
			})
			patches.ApplyFunc(ctrl.NewManager, func(cfg *rest.Config, options ctrl.Options) (manager.Manager, error) {
				capturedOptions = options
				return nil, nil
			})

			mgr, err := initManager(tt.enableHTTP2, "0", tt.secureMetrics, ":8081", false)
			require.NoError(t, err)
			assert.Nil(t, mgr)
			assert.Equal(t, tt.expectTLSOpts, len(capturedOptions.Metrics.TLSOpts) > 0)
			assert.Equal(t, tt.expectFilterConfig, capturedOptions.Metrics.FilterProvider != nil)
			assert.False(t, capturedOptions.LeaderElection)
			assert.Equal(t, "348bca99.openfuyao.com", capturedOptions.LeaderElectionID)
		})
	}
}
