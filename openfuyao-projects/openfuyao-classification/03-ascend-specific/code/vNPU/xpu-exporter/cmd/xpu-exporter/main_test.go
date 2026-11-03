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

package main

import (
	"context"
	"errors"
	"flag"
	"os"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/collector"
	"huawei.com/xpu-exporter/collector/npuservice"
	"huawei.com/xpu-exporter/server"
)

type mockCollectorSvc struct{}

func (m *mockCollectorSvc) CreateCollector(_ time.Duration, _ time.Duration) prometheus.Collector {
	return &mockPromCollector{Desc: prometheus.NewDesc("mock_metric", "mock help", nil, nil)}
}

func (m *mockCollectorSvc) Start(_ context.Context, _ context.CancelFunc) { _ = m }

func (m *mockCollectorSvc) GetName() string { return "npu" }

type mockPromCollector struct {
	Desc *prometheus.Desc
}

func (m *mockPromCollector) Describe(ch chan<- *prometheus.Desc) {
	if m.Desc == nil {
		return
	}
	ch <- m.Desc
}

func (m *mockPromCollector) Collect(ch chan<- prometheus.Metric) {
	if m.Desc == nil {
		return
	}
	metric, err := prometheus.NewConstMetric(m.Desc, prometheus.GaugeValue, 1.0)
	if err != nil {
		return
	}
	ch <- metric
}

func TestCheckCommonParamValid(t *testing.T) {
	origUpdateTime := updateTime
	defer func() { updateTime = origUpdateTime }()

	tests := []struct {
		Name       string
		UpdateTime int
		WantErr    bool
	}{
		{Name: "valid_default", UpdateTime: updateTimeConst, WantErr: false},
		{Name: "valid_one", UpdateTime: 1, WantErr: false},
		{Name: "valid_sixty", UpdateTime: oneMinute, WantErr: false},
		{Name: "zero_invalid", UpdateTime: 0, WantErr: true},
		{Name: "too_large", UpdateTime: oneMinute + 1, WantErr: true},
		{Name: "negative", UpdateTime: -1, WantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			updateTime = tt.UpdateTime
			err := checkCommonParamValid()
			if tt.WantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestLoadCollectorService_NpuType(t *testing.T) {
	origXpuType := xpuType
	origServerHandler := serverHandler
	defer func() {
		xpuType = origXpuType
		serverHandler = origServerHandler
	}()

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(npuservice.New, func(_ string) collector.ICollectorService {
		return &mockCollectorSvc{}
	})

	xpuType = npuservice.CollectorName
	serverHandler = &server.ExporterServer{}
	err := loadCollectorService()
	assert.NoError(t, err)
}

func TestLoadCollectorService_NilCollector(t *testing.T) {
	origXpuType := xpuType
	origServerHandler := serverHandler
	defer func() {
		xpuType = origXpuType
		serverHandler = origServerHandler
	}()

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(npuservice.New, func(_ string) collector.ICollectorService {
		return nil
	})

	xpuType = npuservice.CollectorName
	serverHandler = &server.ExporterServer{}
	err := loadCollectorService()
	assert.Error(t, err)
}

func TestLoadCollectorService_InvalidType(t *testing.T) {
	origXpuType := xpuType
	origServerHandler := serverHandler
	defer func() {
		xpuType = origXpuType
		serverHandler = origServerHandler
	}()

	tests := []struct {
		Name    string
		XpuType string
	}{
		{Name: "empty_type", XpuType: ""},
		{Name: "gpu_type", XpuType: "gpu"},
		{Name: "random_type", XpuType: "invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			xpuType = tt.XpuType
			serverHandler = &server.ExporterServer{}
			err := loadCollectorService()
			assert.Error(t, err)
		})
	}
}

func setupMainPatches(fatallnTracker *bool) *gomonkey.Patches {
	patches := gomonkey.NewPatches()

	patches.ApplyFunc(flag.Parse, func() { _ = 0 })
	patches.ApplyFunc(syscall.Umask, func(_ int) int { return 0 })
	patches.ApplyFunc(log.Infof, func(_ string, _ ...interface{}) { _ = 0 })
	patches.ApplyFunc(log.Fatalln, func(_ ...interface{}) {
		if fatallnTracker != nil {
			*fatallnTracker = true
		}
	})
	patches.ApplyFunc(os.Exit, func(_ int) { _ = 0 })

	return patches
}

func applyMainMethodMocks(patches *gomonkey.Patches, mockColl prometheus.Collector) {
	patches.ApplyMethodFunc(&server.ExporterServer{}, "VerifyServerParams",
		func() error { return nil })
	patches.ApplyMethodFunc(&server.ExporterServer{}, "CreateCollector",
		func(_ time.Duration, _ time.Duration) prometheus.Collector {
			return mockColl
		})
	patches.ApplyMethodFunc(&server.ExporterServer{}, "StartCollect",
		func(_ context.Context, fn context.CancelFunc) { fn() })
	patches.ApplyMethodFunc(&server.ExporterServer{}, "StartServe",
		func(_ context.Context, fn context.CancelFunc, _ *prometheus.Registry) { fn() })
}

func TestMain_Success(t *testing.T) {
	fatallnCalled := false
	patches := setupMainPatches(&fatallnCalled)
	defer patches.Reset()

	patches.ApplyFunc(log.InitLogging, func(_ string) error { return nil })
	mockColl := &mockPromCollector{Desc: prometheus.NewDesc("mock_metric", "mock help", nil, nil)}
	patches.ApplyFunc(loadCollectorService, func() error { return nil })
	patches.ApplyFunc(checkCommonParamValid, func() error { return nil })
	applyMainMethodMocks(patches, mockColl)

	main()

	assert.False(t, fatallnCalled)
}

func TestMain_LoadCollectorServiceError(t *testing.T) {
	fatallnCalled := false
	patches := setupMainPatches(&fatallnCalled)
	defer patches.Reset()

	patches.ApplyFunc(log.InitLogging, func(_ string) error { return nil })
	mockColl := &mockPromCollector{Desc: prometheus.NewDesc("mock_metric", "mock help", nil, nil)}
	patches.ApplyFunc(loadCollectorService, func() error {
		return errors.New("load collector service error")
	})
	patches.ApplyFunc(checkCommonParamValid, func() error { return nil })
	applyMainMethodMocks(patches, mockColl)

	main()

	assert.True(t, fatallnCalled)
}

func TestMain_CheckParamError(t *testing.T) {
	fatallnCalled := false
	patches := setupMainPatches(&fatallnCalled)
	defer patches.Reset()

	patches.ApplyFunc(log.InitLogging, func(_ string) error { return nil })
	mockColl := &mockPromCollector{Desc: prometheus.NewDesc("mock_metric", "mock help", nil, nil)}
	patches.ApplyFunc(loadCollectorService, func() error { return nil })
	patches.ApplyFunc(checkCommonParamValid, func() error {
		return errors.New("check param error")
	})
	applyMainMethodMocks(patches, mockColl)

	main()

	assert.True(t, fatallnCalled)
}

func TestMain_VerifyServerParamsError(t *testing.T) {
	fatallnCalled := false
	patches := setupMainPatches(&fatallnCalled)
	defer patches.Reset()

	patches.ApplyFunc(log.InitLogging, func(_ string) error { return nil })
	mockColl := &mockPromCollector{Desc: prometheus.NewDesc("mock_metric", "mock help", nil, nil)}
	patches.ApplyFunc(loadCollectorService, func() error { return nil })
	patches.ApplyFunc(checkCommonParamValid, func() error { return nil })
	patches.ApplyMethodFunc(&server.ExporterServer{}, "VerifyServerParams",
		func() error { return errors.New("verify server params error") })
	patches.ApplyMethodFunc(&server.ExporterServer{}, "CreateCollector",
		func(_ time.Duration, _ time.Duration) prometheus.Collector {
			return mockColl
		})
	patches.ApplyMethodFunc(&server.ExporterServer{}, "StartCollect",
		func(_ context.Context, fn context.CancelFunc) { fn() })
	patches.ApplyMethodFunc(&server.ExporterServer{}, "StartServe",
		func(_ context.Context, fn context.CancelFunc, _ *prometheus.Registry) { fn() })

	main()

	assert.True(t, fatallnCalled)
}

func TestMain_InitLoggingError(t *testing.T) {
	fatallnCalled := false
	patches := setupMainPatches(&fatallnCalled)
	defer patches.Reset()

	patches.ApplyFunc(log.InitLogging, func(_ string) error {
		return errors.New("init logging error")
	})
	patches.ApplyFunc(loadCollectorService, func() error { return nil })
	patches.ApplyFunc(checkCommonParamValid, func() error { return nil })
	mockColl := &mockPromCollector{Desc: prometheus.NewDesc("mock_metric", "mock help", nil, nil)}
	applyMainMethodMocks(patches, mockColl)

	main()

	assert.True(t, fatallnCalled)
}
