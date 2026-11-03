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

package prometheus

import (
	"crypto/x509"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/common/model"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/zlog"
)

const (
	startTimeStamp = 1714995000
	endTimeStamp   = 1714995060
	timeStep       = 15
)

func TestQueryInstantMetrics(t *testing.T) {
	type args struct {
		expr string
		ts   time.Time
	}
	tests := []struct {
		name             string
		apiPattern       string
		fakeResponseData string
		args             args
		want             client.Metric
	}{
		{
			name:       "Valid Instant Query",
			apiPattern: "/api/v1/query",
			fakeResponseData: `{
    "status": "success",
    "data": {
      "resultType": "vector",
      "result": [
        {
          "metric": {
            "__name__": "up",
            "container": "alertmanager",
            "endpoint": "web",
            "instance": "172.17.177.126:9093",
            "job": "alertmanager-main",
            "namespace": "monitoring",
            "pod": "alertmanager-main-2",
            "service": "alertmanager-main"
          },
          "value": [1714995000, "1"]
        }
      ]
    }
  }`,
			args: args{
				expr: `up{container="alertmanager", endpoint="web", instance="172.17.177.126:9093", ` +
					`job="alertmanager-main", namespace="monitoring", pod="alertmanager-main-2", ` +
					`service="alertmanager-main"}`,
				ts: time.Unix(startTimeStamp, 0),
			},
			want: client.Metric{
				Code: func() *int { var c int = successErrCode; return &c }(),
				MetricOutput: client.MetricOutput{
					Type: "vector",
					MetricRecords: []client.MetricRecord{
						{
							Labels: map[string]string{
								"__name__":  "up",
								"container": "alertmanager",
								"endpoint":  "web",
								"instance":  "172.17.177.126:9093",
								"job":       "alertmanager-main",
								"namespace": "monitoring",
								"pod":       "alertmanager-main-2",
								"service":   "alertmanager-main",
							},
							InstantValue: &client.Point{Timestamp: startTimeStamp, Value: 1},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := mockPrometheusService(tt.apiPattern, tt.fakeResponseData)
			defer srv.Close()

			monitoringClient, _ := NewPrometheus(&Options{
				Endpoint: srv.URL,
			})
			got := monitoringClient.QueryInstantMetrics(tt.args.expr, tt.args.ts)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("QueryInstantMetrics() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestQueryRangeMetrics(t *testing.T) {
	type args struct {
		expr  string
		start time.Time
		end   time.Time
		step  time.Duration
	}
	tests := []struct {
		name             string
		apiPattern       string
		fakeResponseData string
		args             args
		want             client.Metric
	}{
		{
			name:       "Valid Range Query",
			apiPattern: "/api/v1/query_range",
			fakeResponseData: `{
    "status": "success",
    "data": {
      "resultType": "matrix",
      "result": [
        {
          "metric": {
            "__name__": "up",
            "container": "alertmanager",
            "endpoint": "web",
            "instance": "172.17.177.126:9093",
            "job": "alertmanager-main",
            "namespace": "monitoring",
            "pod": "alertmanager-main-2",
            "service": "alertmanager-main"
          },
        "values": [
          [1714995000, "1"],
          [1714995015, "1"],
          [1714995030, "1"],
          [1714995045, "1"],
          [1714995060, "1"]
        ]
        }
      ]
    }
  }`,
			args: args{
				expr: `up{container="alertmanager", endpoint="web", instance="172.17.177.126:9093", ` +
					`job="alertmanager-main", namespace="monitoring", pod="alertmanager-main-2", ` +
					`service="alertmanager-main"}`,
				start: time.Unix(startTimeStamp, 0),
				end:   time.Unix(endTimeStamp, 0),
				step:  time.Duration(timeStep * time.Second),
			},
			want: client.Metric{
				Code: func() *int { var c int = successErrCode; return &c }(),
				MetricOutput: client.MetricOutput{
					Type: "matrix",
					MetricRecords: []client.MetricRecord{
						{
							Labels: map[string]string{
								"__name__":  "up",
								"container": "alertmanager",
								"endpoint":  "web",
								"instance":  "172.17.177.126:9093",
								"job":       "alertmanager-main",
								"namespace": "monitoring",
								"pod":       "alertmanager-main-2",
								"service":   "alertmanager-main",
							},
							RangeValues: []client.Point{
								{
									Timestamp: 1714995000,
									Value:     1,
								},
								{
									Timestamp: 1714995015,
									Value:     1,
								},
								{
									Timestamp: 1714995030,
									Value:     1,
								},
								{
									Timestamp: 1714995045,
									Value:     1,
								},
								{
									Timestamp: 1714995060,
									Value:     1,
								},
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := mockPrometheusService(tt.apiPattern, tt.fakeResponseData)
			defer srv.Close()

			monitoringClient, _ := NewPrometheus(&Options{
				Endpoint: srv.URL,
			})
			got := monitoringClient.QueryRangeMetrics(tt.args.expr, tt.args.start, tt.args.end, tt.args.step)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("QueryInstantMetrics() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestQueryInstantCustomMetrics(t *testing.T) {
	type args struct {
		metrics []string
		ts      time.Time
		o       client.QueryOption
	}
	tests := []struct {
		name             string
		apiPattern       string
		fakeResponseData string
		args             args
		want             []client.Metric
	}{
		{
			name:       "Valid Instant Custom Query",
			apiPattern: "/api/v1/query",
			fakeResponseData: `{
    "status": "success",
    "data": {
      "resultType": "vector",
      "result": [
        {
          "value": [1714995000, "0.04920614035086065"]
        }
      ]
    }
  }`,
			args: args{
				metrics: []string{"cluster_cpu_utilisation"},
				ts:      time.Unix(startTimeStamp, 0),
				o:       client.WithClusterLevel(),
			},
			want: []client.Metric{
				{
					MetricName: "cluster_cpu_utilisation",
					MetricOutput: client.MetricOutput{
						Type: "vector",
						MetricRecords: []client.MetricRecord{
							{
								Labels:       map[string]string{},
								InstantValue: &client.Point{Timestamp: startTimeStamp, Value: 0.04920614035086065},
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := mockPrometheusService(tt.apiPattern, tt.fakeResponseData)
			defer srv.Close()

			monitoringClient, _ := NewPrometheus(&Options{
				Endpoint: srv.URL,
			})
			got := monitoringClient.QueryInstantCustomMetrics(tt.args.metrics, tt.args.ts, tt.args.o)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("QueryInstantMetrics() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestQueryRangeCustomMetrics(t *testing.T) {
	type args struct {
		metrics []string
		start   time.Time
		end     time.Time
		step    time.Duration
		o       client.QueryOption
	}
	tests := []struct {
		name             string
		apiPattern       string
		fakeResponseData string
		args             args
		want             []client.Metric
	}{
		{
			name:       "Valid Range Custom Query",
			apiPattern: "/api/v1/query_range",
			fakeResponseData: `{
    "status": "success",
    "data": {
      "resultType": "matrix",
      "result": [
        {
        "values": [
          [1714995000, "0.04920614035086065"],
          [1714995015, "0.05015570175439653"],
          [1714995030, "0.04922149122809006"],
          [1714995045, "0.050241228070158954"],
          [1714995060, "0.04941228070172199"]
        ]
        }
      ]
    }
  }`,
			args: args{
				metrics: []string{"cluster_cpu_utilisation"},
				start:   time.Unix(startTimeStamp, 0),
				end:     time.Unix(endTimeStamp, 0),
				step:    time.Duration(timeStep * time.Second),
				o:       client.WithClusterLevel(),
			},
			want: []client.Metric{
				{
					MetricName: "cluster_cpu_utilisation",
					MetricOutput: client.MetricOutput{
						Type: "matrix",
						MetricRecords: []client.MetricRecord{
							{
								Labels: map[string]string{},
								RangeValues: []client.Point{
									{
										Timestamp: 1714995000,
										Value:     0.04920614035086065,
									},
									{
										Timestamp: 1714995015,
										Value:     0.05015570175439653,
									},
									{
										Timestamp: 1714995030,
										Value:     0.04922149122809006,
									},
									{
										Timestamp: 1714995045,
										Value:     0.050241228070158954,
									},
									{
										Timestamp: 1714995060,
										Value:     0.04941228070172199,
									},
								},
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := mockPrometheusService(tt.apiPattern, tt.fakeResponseData)
			defer srv.Close()

			monitoringClient, _ := NewPrometheus(&Options{
				Endpoint: srv.URL,
			})
			got := monitoringClient.QueryRangeCustomMetrics(tt.args.metrics, tt.args.start, tt.args.end,
				tt.args.step, tt.args.o)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("QueryInstantMetrics() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetTargets(t *testing.T) {

	tests := []struct {
		name             string
		apiPattern       string
		fakeResponseData string
		want             []client.Target
		wantErr          bool
	}{
		{
			name:             "Valid Get Targets",
			apiPattern:       "/api/v1/targets",
			fakeResponseData: GetTargetsFakeResponse(),
			want: []client.Target{
				{
					TargetName:         "serviceMonitor/monitoring/alertmanager-main/0",
					Endpoint:           "https://prometheus/api/v1/metrics",
					Service:            "alertmanager-main",
					Health:             "up",
					Namespace:          "monitoring",
					LastError:          "",
					LastScrape:         parseTime("2025-08-06T09:28:50.270215847Z"),
					LastScrapeDuration: 0.00305425,
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := mockPrometheusService(tt.apiPattern, tt.fakeResponseData)
			defer srv.Close()

			monitoringClient, _ := NewPrometheus(&Options{
				Endpoint: srv.URL,
			})

			got, err := monitoringClient.GetTargets()
			if (err != nil) != tt.wantErr {
				t.Errorf("GetTargets() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetTargets() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetAlertingRules(t *testing.T) {

	test := struct {
		name             string
		apiPattern       string
		fakeResponseData string
		want             []client.AlertingRule
		wantErr          bool
	}{

		name:             "Valid Get Alerting Rules",
		apiPattern:       "/api/v1/rules",
		fakeResponseData: GetAlertingRulesFakeResponse(),
		want: []client.AlertingRule{
			{
				Group:       "alertmanager.rules",
				Name:        "AlertmanagerFailedReload",
				Severity:    "critical",
				Status:      "inactive",
				Duration:    600,
				Description: "Configuration has failed to load for {{ $labels.namespace }}/{{ $labels.pod}}.",
				Summary:     "Reloading an Alertmanager configuration has failed.",
				Query: `max_over_time(alertmanager_config_last_reload_successful{job="alertmanager-main",` +
					`namespace="monitoring"}[5m]) == 0`,
			},
		},
		wantErr: false,
	}

	t.Run(test.name, func(t *testing.T) {
		servie := mockPrometheusService(test.apiPattern, test.fakeResponseData)
		defer servie.Close()

		monitoringClient, _ := NewPrometheus(&Options{
			Endpoint: servie.URL,
		})

		got, err := monitoringClient.GetAlertingRules()
		if (err != nil) != test.wantErr {
			t.Errorf("GetAlertingRules() error = %v, wantErr %v", err, test.wantErr)
			return
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("GetAlertingRules() got = %v, want %v", got, test.want)
		}
	})

}

func TestGetMetricNames(t *testing.T) {

	tests := []struct {
		name             string
		apiPattern       string
		fakeResponseData string
		want             []client.NativeMetricName
		wantErr          bool
	}{
		{
			name:             "Valid Get Metric Names",
			apiPattern:       "/api/v1/label/__name__/values",
			fakeResponseData: GetMetricNamesFakeResponse(),
			want: []client.NativeMetricName{
				{":node_memory_MemAvailable_bytes:sum"},
				{"ALERTS"},
				{"ALERTS_FOR_STATE"},
				{"aggregator_discovery_aggregation_count_total"},
				{"aggregator_unavailable_apiservice"},
				{"aggregator_unavailable_apiservice_total"},
				{"alertmanager_alerts"},
				{"alertmanager_alerts_invalid_total"},
				{"alertmanager_alerts_received_total"},
				{"alertmanager_build_info"},
			},
			wantErr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			srv := mockPrometheusService(test.apiPattern, test.fakeResponseData)
			defer srv.Close()

			monitoringClient, _ := NewPrometheus(&Options{
				Endpoint: srv.URL,
			})

			got, err := monitoringClient.GetMetricNames()
			if (err != nil) != test.wantErr {
				t.Errorf("GetMetricNames() error = %v, wantErr %v", err, test.wantErr)
				return
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("GetMetricNames() got = %v, want %v", got, test.want)
			}
		})
	}
}

func TestParseQueryResp(t *testing.T) {
	tests := []struct {
		name    string
		value   model.Value
		params  parseParams
		want    client.MetricOutput
		wantErr bool
	}{
		{
			name: "Valid Vector Response",
			value: model.Vector{
				&model.Sample{
					Metric: model.Metric{
						"__name__": "up",
						"job":      "prometheus",
					},
					Timestamp: model.Time(1714995000000),
					Value:     model.SampleValue(1),
				},
			},
			params: parseParams{
				time: time.Unix(1714995000, 0),
			},
			want: client.MetricOutput{
				Type: client.MetricTypeVector,
				MetricRecords: []client.MetricRecord{
					{
						Labels: map[string]string{
							"__name__": "up",
							"job":      "prometheus",
						},
						InstantValue: &client.Point{
							Timestamp: 1714995000.0,
							Value:     1,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:  "Empty Vector Response",
			value: model.Vector{},
			params: parseParams{
				time: time.Unix(1714995000, 0),
			},
			want: client.MetricOutput{
				Type: client.MetricTypeVector,
				MetricRecords: []client.MetricRecord{
					{
						Labels:       map[string]string{},
						InstantValue: &client.Point{Timestamp: 1714995000.0, Value: -1},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Invalid Type (Scalar)",
			value: &model.Scalar{
				Value:     model.SampleValue(42),
				Timestamp: model.Time(1714995000000),
			},
			params: parseParams{
				time: time.Unix(1714995000, 0),
			},
			want:    client.MetricOutput{Type: client.MetricTypeVector},
			wantErr: true,
		},
		{
			name: "NaN Value in Vector",
			value: model.Vector{
				&model.Sample{
					Metric: model.Metric{
						"__name__": "up",
					},
					Timestamp: model.Time(1714995000000),
					Value:     model.SampleValue(math.NaN()),
				},
			},
			params: parseParams{
				time: time.Unix(1714995000, 0),
			},
			want: client.MetricOutput{
				Type: client.MetricTypeVector,
				MetricRecords: []client.MetricRecord{
					{
						Labels: map[string]string{
							"__name__": "up",
						},
						InstantValue: &client.Point{
							Timestamp: 1714995000.0,
							Value:     -1,
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseQueryResp(tt.value, tt.params)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseQueryResp() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got.Type != tt.want.Type {
					t.Errorf("parseQueryResp() Type = %v, want %v", got.Type, tt.want.Type)
				}
				if len(got.MetricRecords) != len(tt.want.MetricRecords) {
					t.Errorf("parseQueryResp() MetricRecords length = %v, want %v", len(got.MetricRecords), len(tt.want.MetricRecords))
				}
			}
		})
	}
}

func TestParseQueryRangeResp(t *testing.T) {
	tests := []struct {
		name    string
		value   model.Value
		params  parseParams
		want    client.MetricOutput
		wantErr bool
	}{
		{
			name: "Valid Matrix Response",
			value: model.Matrix{
				&model.SampleStream{
					Metric: model.Metric{
						"__name__": "up",
						"job":      "prometheus",
					},
					Values: []model.SamplePair{
						{Timestamp: model.Time(1714995000000), Value: model.SampleValue(1)},
						{Timestamp: model.Time(1714995015000), Value: model.SampleValue(1)},
					},
				},
			},
			params: parseParams{
				start: time.Unix(1714995000, 0),
				end:   time.Unix(1714995015, 0),
				step:  15 * time.Second,
			},
			want: client.MetricOutput{
				Type: client.MetricTypeMatrix,
				MetricRecords: []client.MetricRecord{
					{
						Labels: map[string]string{
							"__name__": "up",
							"job":      "prometheus",
						},
						RangeValues: []client.Point{
							{Timestamp: 1714995000.0, Value: 1},
							{Timestamp: 1714995015.0, Value: 1},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:  "Empty Matrix Response",
			value: model.Matrix{},
			params: parseParams{
				start: time.Unix(1714995000, 0),
				end:   time.Unix(1714995015, 0),
				step:  15 * time.Second,
			},
			want: client.MetricOutput{
				Type: client.MetricTypeMatrix,
				MetricRecords: []client.MetricRecord{
					{
						Labels: map[string]string{},
						RangeValues: []client.Point{
							{Timestamp: 1714995000.0, Value: -1},
							{Timestamp: 1714995015.0, Value: -1},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Invalid Type (Vector)",
			value: model.Vector{
				&model.Sample{
					Metric: model.Metric{"__name__": "up"},
					Value:  model.SampleValue(1),
				},
			},
			params: parseParams{
				start: time.Unix(1714995000, 0),
				end:   time.Unix(1714995015, 0),
				step:  15 * time.Second,
			},
			want:    client.MetricOutput{Type: client.MetricTypeMatrix},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseQueryRangeResp(tt.value, tt.params)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseQueryRangeResp() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got.Type != tt.want.Type {
					t.Errorf("parseQueryRangeResp() Type = %v, want %v", got.Type, tt.want.Type)
				}
			}
		})
	}
}

func TestProcessSingleValue(t *testing.T) {
	type args struct {
		timestamp model.Time
		value     model.SampleValue
	}
	tests := []struct {
		name string
		args args
		want client.Point
	}{
		{
			name: "Normal Value",
			args: args{
				timestamp: model.Time(1714995000000), // milliseconds
				value:     model.SampleValue(42.5),
			},
			want: client.Point{
				Timestamp: 1714995000.0,
				Value:     42.5,
			},
		},
		{
			name: "Zero Value",
			args: args{
				timestamp: model.Time(1714995000000),
				value:     model.SampleValue(0),
			},
			want: client.Point{
				Timestamp: 1714995000.0,
				Value:     0,
			},
		},
		{
			name: "NaN Value",
			args: args{
				timestamp: model.Time(int64(time.Second / time.Millisecond)),
				value:     model.SampleValue(math.NaN()),
			},
			want: client.Point{
				Timestamp: 1.0,
				Value:     -1.0,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := processSingleValue(tt.args.timestamp, tt.args.value)

			if math.IsNaN(float64(tt.args.value)) {
				if got.Value != tt.want.Value || got.Timestamp != tt.want.Timestamp {
					t.Errorf("processSingleValue() = %v, want %v", got, tt.want)
				}
			} else {
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("processSingleValue() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestGenerateTimestamps(t *testing.T) {
	tests := []struct {
		name  string
		start time.Time
		end   time.Time
		step  time.Duration
		want  map[float64]bool
	}{
		{
			name:  "Single Step",
			start: time.Unix(1714995000, 0),
			end:   time.Unix(1714995000, 0),
			step:  15 * time.Second,
			want:  map[float64]bool{1714995000.0: false},
		},
		{
			name:  "Multiple Steps",
			start: time.Unix(1714995000, 0),
			end:   time.Unix(1714995060, 0),
			step:  15 * time.Second,
			want: map[float64]bool{
				1714995000.0: false,
				1714995015.0: false,
				1714995030.0: false,
				1714995045.0: false,
				1714995060.0: false,
			},
		},
		{
			name:  "One Second Step",
			start: time.Unix(1714995000, 0),
			end:   time.Unix(1714995002, 0),
			step:  1 * time.Second,
			want: map[float64]bool{
				1714995000.0: false,
				1714995001.0: false,
				1714995002.0: false,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := generateTimestamps(tt.start, tt.end, tt.step)
			if len(got) != len(tt.want) {
				t.Errorf("generateTimestamps() length = %v, want %v", len(got), len(tt.want))
			}
			for ts := range tt.want {
				if _, exists := got[ts]; !exists {
					t.Errorf("generateTimestamps() missing timestamp %v", ts)
				}
			}
		})
	}
}

func TestInitTimeRangeAndParams(t *testing.T) {
	start := time.Unix(1714995000, 0)
	end := time.Unix(1714995060, 0)
	step := 15 * time.Second

	timeRange, params := initTimeRangeAndParams(start, end, step)

	if !timeRange.Start.Equal(start) {
		t.Errorf("initTimeRangeAndParams() timeRange.Start = %v, want %v", timeRange.Start, start)
	}
	if !timeRange.End.Equal(end) {
		t.Errorf("initTimeRangeAndParams() timeRange.End = %v, want %v", timeRange.End, end)
	}
	if timeRange.Step != step {
		t.Errorf("initTimeRangeAndParams() timeRange.Step = %v, want %v", timeRange.Step, step)
	}
	if !params.start.Equal(start) {
		t.Errorf("initTimeRangeAndParams() params.start = %v, want %v", params.start, start)
	}
	if !params.end.Equal(end) {
		t.Errorf("initTimeRangeAndParams() params.end = %v, want %v", params.end, end)
	}
	if params.step != step {
		t.Errorf("initTimeRangeAndParams() params.step = %v, want %v", params.step, step)
	}
}

func mockPrometheusService(apiPattern, fakeResponseData string) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc(apiPattern, func(res http.ResponseWriter, req *http.Request) {
		if _, err := res.Write([]byte(fakeResponseData)); err != nil {
			zlog.FormatError("Failed to write mock response for %s: %v", apiPattern, err)
		}
	})
	return httptest.NewServer(mux)
}

func parseTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		zlog.Error(err)
		return time.Time{}
	}
	return t
}

func GetTargetsFakeResponse() string {
	return `{
    "status": "success",
    "data": {
        "activeTargets": [
            {
                "discoveredLabels": {
                    "__meta_kubernetes_namespace": "monitoring",
                    "__meta_kubernetes_service_name": "alertmanager-main",
                    "job": "serviceMonitor/monitoring/alertmanager-main/0"
                },
                "scrapeUrl": "https://prometheus/api/v1/metrics",
                "lastError": "",
                "lastScrape": "2025-08-06T09:28:50.270215847Z",
                "lastScrapeDuration": 0.00305425,
                "health": "up"
            }
		]
	}
}`
}

func GetAlertingRulesFakeResponse() string {
	const alertRuleFilePath = "/etc/prometheus/rules/prometheus-k8s-rulefiles-0/" +
		"monitoring-alertmanager-main-rules-d9299149-6454-4158-a8f4-dbca4dfeb74b.yaml"

	return fmt.Sprintf(`{
	"status": "success",
	"data": {
		"groups": [{
			"name": "alertmanager.rules",
			"file": %q,
			"rules": [{
				"state": "inactive",
				"name": "AlertmanagerFailedReload",
				"query": "max_over_time(alertmanager_config_last_reload_successful{job=\"alertmanager-main\",`+
		`namespace=\"monitoring\"}[5m]) == 0",
				"duration": 600,
				"keepFiringFor": 0,
				"labels": {
					"severity": "critical"
				},
				"annotations": {
					"description": "Configuration has failed to load for {{ $labels.namespace }}/{{ $labels.pod}}.",
					"runbook_url": "https://runbooks.prometheus-operator.dev/runbooks/alertmanager/alertmanagerfailedreload",
					"summary": "Reloading an Alertmanager configuration has failed."
				},
				"alerts": [],
				"health": "ok",
				"evaluationTime": 0.000145934,
				"lastEvaluation": "2025-08-07T01:40:32.77060431Z",
				"type": "alerting"
			}],
			"interval": 30,
			"limit": 0,
			"evaluationTime": 0.003455247,
			"lastEvaluation": "2025-08-07T01:40:32.770595498Z"
		}]
	}
}`, alertRuleFilePath)
}

func GetMetricNamesFakeResponse() string {
	return `{
    "status": "success",
    "data": [
        ":node_memory_MemAvailable_bytes:sum",
        "ALERTS",
        "ALERTS_FOR_STATE",
        "aggregator_discovery_aggregation_count_total",
        "aggregator_unavailable_apiservice",
        "aggregator_unavailable_apiservice_total",
        "alertmanager_alerts",
        "alertmanager_alerts_invalid_total",
        "alertmanager_alerts_received_total",
        "alertmanager_build_info"
    ]
}`
}

func TestNewPrometheusTLSWithMock(t *testing.T) {
	// mock os.ReadFile
	readFilePatch := gomonkey.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte("mock cert data"), nil
	})
	defer readFilePatch.Reset()

	// mock AppendCertsFromPEM
	appendCertsPatch := gomonkey.ApplyMethod(reflect.TypeOf(&x509.CertPool{}), "AppendCertsFromPEM",
		func(_ *x509.CertPool, _ []byte) bool {
			return true
		})
	defer appendCertsPatch.Reset()

	options := &Options{
		Endpoint:           "https://mock-prometheus",
		InsecureSkipVerify: false, // 严格验证模式，触发 CA 加载逻辑
	}

	client, err := NewPrometheus(options)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewPrometheusHTTP(t *testing.T) {
	options := &Options{
		Endpoint:           "http://mock-prometheus",
		InsecureSkipVerify: false,
	}

	client, err := NewPrometheus(options)
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

func TestNewPrometheusInvalidEndpoint(t *testing.T) {
	options := &Options{
		Endpoint:           "://invalid-url",
		InsecureSkipVerify: false,
	}

	client, err := NewPrometheus(options)
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestNewPrometheusTLSWithCAError(t *testing.T) {
	// mock os.ReadFile to return error
	readFilePatch := gomonkey.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return nil, fmt.Errorf("file not found")
	})
	defer readFilePatch.Reset()

	options := &Options{
		Endpoint:           "https://mock-prometheus",
		InsecureSkipVerify: false,
	}

	client, err := NewPrometheus(options)
	assert.Error(t, err)
	assert.Nil(t, client)
}

func TestNewPrometheusTLSWithInvalidCA(t *testing.T) {
	// mock os.ReadFile
	readFilePatch := gomonkey.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte("invalid cert data"), nil
	})
	defer readFilePatch.Reset()

	// mock AppendCertsFromPEM to return false
	appendCertsPatch := gomonkey.ApplyMethod(reflect.TypeOf(&x509.CertPool{}), "AppendCertsFromPEM",
		func(_ *x509.CertPool, _ []byte) bool {
			return false
		})
	defer appendCertsPatch.Reset()

	options := &Options{
		Endpoint:           "https://mock-prometheus",
		InsecureSkipVerify: false,
	}

	client, err := NewPrometheus(options)
	assert.Error(t, err)
	assert.Nil(t, client)
}
