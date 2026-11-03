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

// Package prometheus provides a robust interface for interacting with the Prometheus API,
// allowing for the retrieval and management of monitoring data. This package abstracts
// the complexity involved in querying Prometheus servers and handling the responses.
package prometheus

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/api"
	"github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"

	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/zlog"
)

const (
	millisecondsToSeconds = 1000
	queryErrCode          = 4101
	successErrCode        = 0

	prometheusCAFile = "/ssl/prometheusCA.pem"
)

type prometheus struct {
	client v1.API
}

type parseParams struct {
	start time.Time
	end   time.Time
	step  time.Duration
	time  time.Time
}

// NewPrometheus initializes a new Prometheus client with the given configuration options.
func NewPrometheus(options *Options) (client.Interface, error) {
	parseEndpoint, err := url.Parse(options.Endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Prometheus endpoint: %v", err)
	}

	cfg := api.Config{
		Address: options.Endpoint,
	}

	if parseEndpoint.Scheme == "https" {
		tlsConfig := &tls.Config{
			InsecureSkipVerify: options.InsecureSkipVerify,
			MinVersion:         tls.VersionTLS12,
			MaxVersion:         tls.VersionTLS13,
		}

		// 只有在严格验证时才加载 CA
		if !options.InsecureSkipVerify {
			caCert, err := os.ReadFile(prometheusCAFile)
			if err != nil {
				return nil, fmt.Errorf("failed to read CA file %s: %v", prometheusCAFile, err)
			}

			caCertPool := x509.NewCertPool()
			if ok := caCertPool.AppendCertsFromPEM(caCert); !ok {
				return nil, fmt.Errorf("failed to append CA cert from %s", prometheusCAFile)
			}

			tlsConfig.RootCAs = caCertPool
		}

		roundTripper := &http.Transport{
			TLSClientConfig: tlsConfig,
		}

		cfg.RoundTripper = roundTripper
	}

	prometheusClient, err := api.NewClient(cfg)
	return prometheus{client: v1.NewAPI(prometheusClient)}, err
}

// QueryInstantMetrics performs an instant query against Prometheus using the provided expression and timestamp.
func (p prometheus) QueryInstantMetrics(expr string, ts time.Time) client.Metric {
	params := parseParams{
		time: ts,
	}

	instantQueryFunc := func() (model.Value, v1.Warnings, error) {
		return p.client.Query(context.Background(), expr, ts)
	}

	res := p.executeQuery(instantQueryFunc, parseQueryResp, params)

	if res.Msg != "" {
		code := queryErrCode
		res.Code = &code
	} else {
		code := successErrCode
		res.Code = &code
	}
	return res
}

// QueryRangeMetrics performs a range query against Prometheus using the provided expression,
// start and end time, and step.
func (p prometheus) QueryRangeMetrics(expr string, start, end time.Time, step time.Duration) client.Metric {
	timeRange, params := initTimeRangeAndParams(start, end, step)

	rangeQueryFunc := func() (model.Value, v1.Warnings, error) {
		return p.client.QueryRange(context.Background(), expr, timeRange)
	}

	res := p.executeQuery(rangeQueryFunc, parseQueryRangeResp, params)

	if res.Msg != "" {
		code := queryErrCode
		res.Code = &code
	} else {
		code := successErrCode
		res.Code = &code
	}
	return res
}

// QueryInstantCustomMetrics performs instant queries for custom metrics based on the given query options and timestamp.
func (p prometheus) QueryInstantCustomMetrics(metrics []string, ts time.Time, o client.QueryOption) []client.Metric {
	var wg sync.WaitGroup

	res := make([]client.Metric, len(metrics))
	opts := client.NewQueryOptions(o)

	params := parseParams{
		time: ts,
	}

	for i, metric := range metrics {
		wg.Add(1)
		go func(i int, metric string) {
			defer wg.Done()
			instantCustomQueryFunc := func() (model.Value, v1.Warnings, error) {
				return p.client.Query(context.Background(), makeExpr(metric, *opts), ts)
			}

			if i >= len(res) {
				zlog.FormatError("Invalid index %d out of range for metrics array", i)
				return
			}

			tempResult := p.executeQuery(instantCustomQueryFunc, parseQueryResp, params)
			tempResult.MetricName = metric

			res[i] = tempResult

		}(i, metric)
	}

	wg.Wait()
	return res
}

// QueryRangeCustomMetrics performs range queries for custom metrics based on the given query options,
// start, end, and step.
func (p prometheus) QueryRangeCustomMetrics(metrics []string, start, end time.Time, step time.Duration,
	o client.QueryOption) []client.Metric {
	var wg sync.WaitGroup

	res := make([]client.Metric, len(metrics))
	opts := client.NewQueryOptions(o)

	timeRange, params := initTimeRangeAndParams(start, end, step)

	for i, metric := range metrics {
		wg.Add(1)
		go func(i int, metric string) {
			defer wg.Done()
			rangeCustomQueryFunc := func() (model.Value, v1.Warnings, error) {
				return p.client.QueryRange(context.Background(), makeExpr(metric, *opts), timeRange)
			}

			if i >= len(res) {
				zlog.FormatError("Invalid index %d out of range for metrics array", i)
				return
			}

			tempResult := p.executeQuery(rangeCustomQueryFunc, parseQueryRangeResp, params)
			tempResult.MetricName = metric

			res[i] = tempResult

		}(i, metric)
	}
	wg.Wait()
	return res
}

// GetTargets retrieves the current set of targets that Prometheus is scraping.
func (p prometheus) GetTargets() ([]client.Target, error) {
	var res []client.Target

	items, err := p.client.Targets(context.Background())
	if err != nil {
		zlog.Error(err)
		return res, err
	}

	for _, item := range items.Active {
		target := client.Target{
			TargetName:         item.DiscoveredLabels["job"],
			Endpoint:           item.ScrapeURL,
			Service:            item.DiscoveredLabels["__meta_kubernetes_service_name"],
			Health:             string(item.Health),
			Namespace:          item.DiscoveredLabels["__meta_kubernetes_namespace"],
			LastError:          item.LastError,
			LastScrape:         item.LastScrape,
			LastScrapeDuration: item.LastScrapeDuration,
		}
		res = append(res, target)
	}

	return res, nil

}

// GetAlertingRules retrieves the set of alerting rules currently configured in Prometheus.
func (p prometheus) GetAlertingRules() ([]client.AlertingRule, error) {
	var res []client.AlertingRule

	items, err := p.client.Rules(context.Background())
	if err != nil {
		zlog.Error(err)
		return res, err
	}

	for _, group := range items.Groups {
		for _, rule := range group.Rules {
			if alertRule, ok := rule.(v1.AlertingRule); ok {
				alertingRule := client.AlertingRule{
					Group:       group.Name,
					Name:        alertRule.Name,
					Severity:    string(alertRule.Labels["severity"]),
					Status:      alertRule.State,
					Duration:    alertRule.Duration,
					Description: string(alertRule.Annotations["description"]),
					Summary:     string(alertRule.Annotations["summary"]),
					Query:       alertRule.Query,
				}
				res = append(res, alertingRule)
			}
		}
	}
	return res, nil
}

// GetMetricNames retrieves all metric names available in the Prometheus server.
func (p prometheus) GetMetricNames() ([]client.NativeMetricName, error) {
	var res []client.NativeMetricName

	metricNames, _, err := p.client.LabelValues(context.Background(),
		"__name__", []string{}, time.Time{}, time.Time{})
	if err != nil {
		zlog.Error(err)
		return res, err
	}

	for _, metricName := range metricNames {
		name := client.NativeMetricName{
			NativeMetricName: string(metricName),
		}
		res = append(res, name)
	}

	return res, nil

}

func parseQueryResp(value model.Value, params parseParams) (client.MetricOutput, error) {
	res := client.MetricOutput{Type: client.MetricTypeVector}

	data, ok := value.(model.Vector)
	if !ok {
		errorMsg := fmt.Sprintf("Type assertion failed: expected model.Vector, got %s", reflect.TypeOf(value))
		zlog.Error(errorMsg)
		return res, errors.New(errorMsg)
	}

	singleTimestamp := float64(params.time.Unix())

	for _, v := range data {
		mv := client.MetricRecord{
			Labels: make(map[string]string),
		}

		for k, v := range v.Metric {
			mv.Labels[string(k)] = string(v)
		}

		point := processSingleValue(v.Timestamp, v.Value)
		mv.InstantValue = &point
		res.MetricRecords = append(res.MetricRecords, mv)
	}

	if len(data) == 0 {
		mv := client.MetricRecord{
			Labels: make(map[string]string),
		}
		mv.InstantValue = &client.Point{
			Timestamp: singleTimestamp,
			Value:     float64(-1),
		}
		res.MetricRecords = append(res.MetricRecords, mv)
	}

	return res, nil
}

func parseQueryRangeResp(value model.Value, params parseParams) (client.MetricOutput, error) {
	res := client.MetricOutput{Type: client.MetricTypeMatrix}

	data, ok := value.(model.Matrix)
	if !ok {
		errorMsg := fmt.Sprintf("Type assertion failed: expected model.Matrix, got %s", reflect.TypeOf(value))
		zlog.Error(errorMsg)
		return res, errors.New(errorMsg)
	}

	timestamps := generateTimestamps(params.start, params.end, params.step)

	for _, v := range data {
		mv := client.MetricRecord{
			Labels: make(map[string]string),
		}

		for k, v := range v.Metric {
			mv.Labels[string(k)] = string(v)
		}

		for _, k := range v.Values {
			timestamp := float64(k.Timestamp) / millisecondsToSeconds
			timestamps[timestamp] = true
			mv.RangeValues = append(mv.RangeValues, processSingleValue(k.Timestamp, k.Value))
		}

		res.MetricRecords = append(res.MetricRecords, mv)
	}

	for i := range res.MetricRecords {
		for timestamp, exists := range timestamps {
			if !exists {
				res.MetricRecords[i].RangeValues = append(res.MetricRecords[i].RangeValues, client.Point{
					Timestamp: timestamp,
					Value:     float64(-1),
				})
			}
		}
		sort.Slice(res.MetricRecords[i].RangeValues, func(a, b int) bool {
			return res.MetricRecords[i].RangeValues[a].Timestamp < res.MetricRecords[i].RangeValues[b].Timestamp
		})
	}

	if len(data) == 0 {
		mv := client.MetricRecord{
			Labels: make(map[string]string),
		}
		for ts := range timestamps {
			mv.RangeValues = append(mv.RangeValues, client.Point{
				Timestamp: ts,
				Value:     float64(-1),
			})
		}
		res.MetricRecords = append(res.MetricRecords, mv)
	}

	return res, nil
}

func processSingleValue(timestamp model.Time, value model.SampleValue) client.Point {
	ts := float64(timestamp) / millisecondsToSeconds
	if math.IsNaN(float64(value)) {
		return client.Point{
			Timestamp: ts,
			Value:     float64(-1),
		}
	}
	return client.Point{
		Timestamp: ts,
		Value:     float64(value),
	}
}

func generateTimestamps(start, end time.Time, step time.Duration) map[float64]bool {
	timestamps := make(map[float64]bool)
	for ts := start; ts.Before(end) || ts.Equal(end); ts = ts.Add(step) {
		timestamps[float64(ts.Unix())] = false
	}
	return timestamps
}

func initTimeRangeAndParams(start, end time.Time, step time.Duration) (v1.Range, parseParams) {
	timeRange := v1.Range{
		Start: start,
		End:   end,
		Step:  step,
	}

	params := parseParams{
		start: start,
		end:   end,
		step:  step,
	}

	return timeRange, params
}

func (p prometheus) executeQuery(queryFunc func() (model.Value, v1.Warnings, error),
	parseFunc func(model.Value, parseParams) (client.MetricOutput, error), params parseParams) client.Metric {
	value, _, err := queryFunc()
	if err != nil {
		return client.Metric{Msg: err.Error()}
	}

	metricData, parseErr := parseFunc(value, params)

	if parseErr != nil {
		return client.Metric{Msg: parseErr.Error()}
	}
	return client.Metric{MetricOutput: metricData}
}
