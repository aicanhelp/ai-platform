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

package metrics

import (
	"fmt"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/util/json"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// UTCOffsetHours is the offset for UTC+8 timezone (Asia/Shanghai)
	UTCOffsetHours = 8
)

type HttpResponse struct {
	Code int         `json:"code"`
	Msg  string      `json:"msg"`
	Data interface{} `json:"data,omitempty"`
}

func (res HttpResponse) bytes() ([]byte, error) {
	return json.Marshal(res)
}

func newResp(data interface{}, msg string, code int) HttpResponse {
	return HttpResponse{
		Code: code,
		Msg:  msg,
		Data: data,
	}
}

// Ok creates a successful HTTP response
func Ok(data interface{}, msg string) ([]byte, error) {
	return newResp(data, msg, http.StatusOK).bytes()
}

// Error creates an error HTTP response
func Error(msg string) ([]byte, error) {
	return newResp(nil, msg, http.StatusInternalServerError).bytes()
}

// Bad creates a bad request HTTP response
func Bad(msg string) ([]byte, error) {
	return newResp(nil, msg, http.StatusBadRequest).bytes()
}

type httpExportResponse struct {
	Name      string `json:"name"`
	StartTime string `json:"start_time"`
	EndTime   string `json:"end_time"`
	Describe  string `json:"describe"`
	Duration  int    `json:"duration"`
}

// ParamsResult encapsulates the result of extractAndValidateParams function
type ParamsResult struct {
	Cluster string
	From    string
	To      string
	Err     error
}

// extractAndValidateParams extracts and validates the query parameters from the request
func (r *BKEClusterMetricRegister) extractAndValidateParams(req *http.Request) ParamsResult {
	cluster := req.URL.Query().Get("cluster")
	if cluster == "" {
		return ParamsResult{
			Cluster: "",
			From:    "",
			To:      "",
			Err:     fmt.Errorf("param cluster is required"),
		}
	}
	from := req.URL.Query().Get("from")
	to := req.URL.Query().Get("to")
	return ParamsResult{
		Cluster: cluster,
		From:    from,
		To:      to,
		Err:     nil,
	}
}

// validateTimeRange validates the time range parameters
func (r *BKEClusterMetricRegister) validateTimeRange(from, to string) (time.Time, time.Time, error) {
	startTime, endTime, err := ParseTimeFromTo(from, to)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	return startTime, endTime, nil
}

// collectAndFilterMetrics collects metrics and filters them based on time range
func (r *BKEClusterMetricRegister) collectAndFilterMetrics(cluster string,
	startTime, endTime time.Time) (map[string][]map[string]string, error) {
	metricsGather, err := r.Gather(cluster, PhaseDurationSeconds, NodeBootstrapDurationSeconds)
	if err != nil {
		return nil, err
	}
	return filterMetricsData(metricsGather, startTime, endTime), nil
}

// formatMetricsResponse formats the metrics data into the expected response structure
func (r *BKEClusterMetricRegister) formatMetricsResponse(metricsGather map[string][]map[string]string) []httpExportResponse {
	return ProcessMetrics(metricsGather)
}

// HttpExportFunc returns an HTTP handler function for exporting metrics
func (r *BKEClusterMetricRegister) HttpExportFunc() http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// Extract and validate query parameters
		paramsResult := r.extractAndValidateParams(req)
		if paramsResult.Err != nil {
			r.writeErrorResponse(w, Bad, paramsResult.Err.Error())
			return
		}
		cluster := paramsResult.Cluster
		from := paramsResult.From
		to := paramsResult.To

		// Validate time range
		startTime, endTime, err := r.validateTimeRange(from, to)
		if err != nil {
			log.Errorf("parse time error: %v", err)
			r.writeErrorResponse(w, Bad, fmt.Sprintf("parse time error: %v", err))
			return
		}
		log.Debugf("gather cluster: %q from %q to %q", cluster, startTime, endTime)

		// Collect and filter metrics
		metricsGather, err := r.collectAndFilterMetrics(cluster, startTime, endTime)
		if err != nil {
			r.writeErrorResponse(w, Error, fmt.Sprintf("gather cluster: %s, error: %v", cluster, err))
			return
		}

		// Format response
		resp := r.formatMetricsResponse(metricsGather)

		// Write success response
		res, err := Ok(resp, "export success")
		if err != nil {
			log.Errorf("failed to create Ok response: %v", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
		if _, err = w.Write(res); err != nil {
			log.Errorf("write resp error: %v", err)
			return
		}
	}
}

// writeErrorResponse is a helper function to write error responses
func (r *BKEClusterMetricRegister) writeErrorResponse(w http.ResponseWriter,
	responseFunc func(string) ([]byte, error), message string) {
	res, err := responseFunc(message)
	if err != nil {
		log.Errorf("failed to create error response: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	if _, err := w.Write(res); err != nil {
		log.Errorf("write error response error: %v", err)
		return
	}
}

// HttpClusterFunc returns an HTTP handler function for listing clusters
func (r *BKEClusterMetricRegister) HttpClusterFunc() http.HandlerFunc {
	return func(writer http.ResponseWriter, request *http.Request) {
		var resp []string

		r.mux.RLock()
		for clusterName := range r.collectors {
			resp = append(resp, clusterName)
		}
		r.mux.RUnlock()

		res, err := Ok(resp, "success")
		if err != nil {
			log.Errorf("failed to create Ok response: %v", err)
			http.Error(writer, "Internal Server Error", http.StatusInternalServerError)
			return
		}
		if _, err := writer.Write(res); err != nil {
			log.Errorf("write response error: %v", err)
			return
		}
	}
}

// calculateTimeDifference 计算时间差值单位为s，不足1s为0
func calculateTimeDifference(startTimeStr, endTimeStr string) int {

	// 解析开始时间和结束时间
	startTime, err := time.Parse(TimeFormat, startTimeStr)
	if err != nil {
		log.Warnf("Error parsing start time: %v", err)
		return 0
	}

	endTime, err := time.Parse(TimeFormat, endTimeStr)
	if err != nil {
		log.Warnf("Error parsing end time: %v", err)
		return 0
	}

	// 计算时间差值（单位为秒）
	duration := endTime.Sub(startTime)
	seconds := int(duration.Seconds())

	// 如果时间差值小于1秒，则返回0
	if seconds < 1 {
		return 0
	}

	return seconds
}

func filterMetricsData(metricsGather map[string][]map[string]string,
	start, end time.Time) map[string][]map[string]string {
	res := make(map[string][]map[string]string)

	for name, gatherMetrics := range metricsGather {
		resItem := make([]map[string]string, 0)
		for _, metric := range gatherMetrics {

			startTime := metric["start_time"]
			endTime := metric["end_time"]

			if startTime == "" || endTime == "" {
				// 如果没有相关时间字段直接跳过就好
				continue
			}
			// 判断是否在时间范围内

			startTimeTime, err := time.Parse(TimeFormat, startTime)
			if err != nil {
				log.Warnf("Error parsing start time %s: %v", startTime, err)
				continue
			}
			endTimeTime, err := time.Parse(TimeFormat, endTime)
			if err != nil {
				log.Warnf("Error parsing end time %s: %v", endTime, err)
				continue
			}
			// time.Parse() 解析出来的时间time.Time 不带时区（UTC），
			// 但是这个时间（startTime\endTime）本身是加了时区偏移后的时间，程序中所有的时间都是上海时区
			// 所以有这奇怪的一坨
			if startTimeTime.Local().Add(-UTCOffsetHours*time.Hour).After(start) &&
				endTimeTime.Local().Add(-UTCOffsetHours*time.Hour).Before(end) {
				resItem = append(resItem, metric)
			}
		}
		res[name] = resItem
		log.Debugf("gather vec: %q %d match time range", name, len(resItem))
	}

	return res
}
