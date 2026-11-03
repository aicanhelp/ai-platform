/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package report

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/metric"
	"openfuyao.com/colocation-management/pkg/common"
)

// UsageCollector is used to collect node usage report and send to server.
type UsageCollector struct {
	config          *apps.Configuration
	usageReportChan chan *metric.NodeUsageReport
	clusterState    *aggregate.ClusterState

	nodes    map[string]v1.Time // key is node name, value is lastUsageTimestampCollected
	nodeLock sync.RWMutex
}

const (
	// milliScale is the scale of millicores in a resource.Quantity.
	milliScale = -3
)

// NewUsageCollector creates a new UsageCollector instance.
func NewUsageCollector(config *apps.Configuration, clusterState *aggregate.ClusterState) *UsageCollector {
	collector := &UsageCollector{
		config:          config,
		usageReportChan: make(chan *metric.NodeUsageReport, config.UsageBufCapacity),
		clusterState:    clusterState,
		nodes:           make(map[string]v1.Time),
	}
	clusterState.RegisterNodeCallback(collector.addNodeCallback, collector.deleteNodeCallback)
	return collector
}

// Run starts the usage collector.
func (c *UsageCollector) Run(ctx context.Context) error {
	go c.processUsageReports(ctx)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/metrics/usage", c.receiveUsageReport)

	if err := http.ListenAndServe(c.config.UsageReportAddr, mux); err != nil {
		klog.Errorf("fail to start usage collector server. err: %v", err)
		return err
	}

	close(c.usageReportChan)
	return nil
}

func (c *UsageCollector) processUsageReports(ctx context.Context) {
	var (
		addSampleSuc  = 0
		addSampleFail = 0
	)

	for {
		select {
		case <-ctx.Done():
			klog.Info("usage report processing loop exited due to context cancellation")
			return
		case usageReport := <-c.usageReportChan:
			addSampleSuc, addSampleFail = c.processSingleUsageReport(usageReport, addSampleSuc, addSampleFail)
		}
	}
}

func (c *UsageCollector) processSingleUsageReport(usageReport *metric.NodeUsageReport,
	sucCount, failCount int) (int, int) {
	nodeState := c.clusterState.FindNodeState(usageReport.Node)
	if nodeState == nil {
		klog.Warningf("the node (%s) of sample container to be added does not existed", usageReport.Node)
		return sucCount, failCount
	}

	samples := newContainerUsageSamplesWithKey(usageReport)
	for _, sample := range samples {
		if err := nodeState.AddSample(sample); err != nil {
			failCount++
			klog.Warningf("add sample of container:%v is failed. %v, %v", sample.Container, failCount, err)
		} else {
			sucCount++
			klog.V(common.AdvanceDebugLog).Infof("add sample of container:%v is succeed. %v", sample.Container, sucCount)
		}
	}

	return sucCount, failCount
}
func (c *UsageCollector) receiveUsageReport(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, fmt.Sprintf("Method %v is not supported!", r.Method), http.StatusMethodNotAllowed)
		return
	}

	reports, err := c.parseRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	errCode, errMsg, lastTimestamp := c.processReports(reports)
	c.sendResponse(w, errCode, errMsg, lastTimestamp)
}

func (c *UsageCollector) parseRequest(r *http.Request) ([]metric.NodeUsageReport, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read usage report body failed")
	}
	defer r.Body.Close()

	var reports []metric.NodeUsageReport
	if err = json.Unmarshal(body, &reports); err != nil {
		return nil, fmt.Errorf("unmarshal request data in json")
	}
	return reports, nil
}

func (c *UsageCollector) processReports(reports []metric.NodeUsageReport) (int, string, *v1.Time) {
	if len(reports) == 0 {
		return codeSuccess, "", nil
	}

	var node string
	var lastUsageTimestampCollected *v1.Time

	for _, report := range reports {
		if errCode, errMsg := c.validateReport(report, node); errCode != codeSuccess {
			return errCode, errMsg, nil
		}
		node = report.Node

		lastUsageTimestampCollected = c.findNodeLastUsageTimestampCollected(node)
		if lastUsageTimestampCollected == nil {
			return codeNodeNotInRange, fmt.Sprintf("Node:%v is not in usage collector range.", node), nil
		}

		if !report.Timestamp.Time.After(lastUsageTimestampCollected.Time) {
			return codeUsageTimestampExpired,
				fmt.Sprintf("Usage report's timestamp for node:%v is not after last collected timestamp", node),
				nil
		}

		if !c.sendToChannel(&report, node) {
			return codeReportBufferFull, "report buffer is full. retry send", nil
		}
	}

	return codeSuccess, "", lastUsageTimestampCollected
}

func (c *UsageCollector) validateReport(report metric.NodeUsageReport, expectedNode string) (int, string) {
	if expectedNode != "" && expectedNode != report.Node {
		return codeInvalidFormat,
			fmt.Sprintf("Invalid usage report format, node:%v is not same in this batch of reports", report.Node)
	}
	return codeSuccess, ""
}

func (c *UsageCollector) sendToChannel(report *metric.NodeUsageReport, node string) bool {
	select {
	case c.usageReportChan <- report:
		c.updateNodeLastUsageTimestampCollected(node, report.Timestamp)
		return true
	default:
		klog.Errorf("Usage channel is full! retry send!")
		return false
	}
}

func (c *UsageCollector) sendResponse(w http.ResponseWriter, errCode int, errMsg string, lastTimestamp *v1.Time) {
	response := &metric.NodeUsageResponse[v1.Time]{
		Code: errCode,
		Msg:  errMsg,
	}
	if lastTimestamp != nil {
		response.Data = *lastTimestamp
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		klog.Errorf("encode http response fail. %v", err)
	}
}
func newContainerUsageSamplesWithKey(usage *metric.NodeUsageReport) []*aggregate.ContainerUsageSampleWithKey {
	var samples []*aggregate.ContainerUsageSampleWithKey

	for podRef, containersResource := range usage.Containers {
		podID, err := getPodIDFromRef(podRef)
		if err != nil {
			continue
		}

		for containerName, resources := range containersResource {
			containerSamples := createContainerSamples(podID, containerName, resources, usage.Timestamp.Time)
			samples = append(samples, containerSamples...)
		}
	}
	return samples
}

func getPodIDFromRef(podRef metric.PodRef) (aggregate.PodID, error) {
	namespace, name, err := podRef.NamespaceAndName()
	if err != nil {
		return aggregate.PodID{}, err
	}
	return aggregate.PodID{Namespace: namespace, PodName: name}, nil
}

func createContainerSamples(podID aggregate.PodID, containerName string,
	resources corev1.ResourceList, timestamp time.Time) []*aggregate.ContainerUsageSampleWithKey {
	var samples []*aggregate.ContainerUsageSampleWithKey

	for resourceName, resourceQuantity := range resources {
		sample := createSingleSample(podID, containerName, resourceName, resourceQuantity, timestamp)
		if sample != nil {
			samples = append(samples, sample)
		}
	}

	return samples
}

func createSingleSample(podID aggregate.PodID, containerName string, resourceName corev1.ResourceName,
	quantity resource.Quantity, timestamp time.Time) *aggregate.ContainerUsageSampleWithKey {
	res := aggregate.ResourceName(resourceName)
	amount, valid := convertResourceAmount(res, quantity)
	if !valid {
		return nil
	}

	sample := &aggregate.ContainerUsageSampleWithKey{
		Container: aggregate.ContainerID{
			PodID:         podID,
			ContainerName: containerName,
		},
		ContainerUsageSample: aggregate.ContainerUsageSample{
			MeasureStart: timestamp,
			Resource:     res,
			Usage:        amount,
		},
	}

	klog.V(common.TraceDebugLog).InfoS("CollectLastUsage:",
		"pod.namespace", podID.Namespace,
		"pod.name", podID.PodName,
		"container.name", containerName,
		"MeasureStart", sample.ContainerUsageSample.MeasureStart,
		"Resource", sample.ContainerUsageSample.Resource,
		"usage", sample.ContainerUsageSample.Usage,
		"resourceQuantity", quantity,
	)
	return sample
}

func convertResourceAmount(res aggregate.ResourceName, quantity resource.Quantity) (aggregate.ResourceAmount, bool) {
	switch res {
	case aggregate.ResourceCPU:
		return aggregate.ResourceAmount(quantity.ScaledValue(milliScale)), true
	case aggregate.ResourceMemory:
		return aggregate.ResourceAmount(quantity.Value()), true
	default:
		klog.Warningf("invalid resource name:%v in usage report.", res)
		return 0, false
	}
}

func (c *UsageCollector) addNodeCallback(node string) {
	c.nodeLock.Lock()
	defer c.nodeLock.Unlock()

	if _, ok := c.nodes[node]; !ok {
		c.nodes[node] = v1.Time{}
	}
}

func (c *UsageCollector) deleteNodeCallback(node string) {
	c.nodeLock.Lock()
	defer c.nodeLock.Unlock()

	delete(c.nodes, node)
}

func (c *UsageCollector) findNodeLastUsageTimestampCollected(node string) *v1.Time {
	c.nodeLock.RLock()
	defer c.nodeLock.RUnlock()

	lastTime, ok := c.nodes[node]
	if ok {
		return &lastTime
	}
	return nil
}

func (c *UsageCollector) updateNodeLastUsageTimestampCollected(node string, lastTime v1.Time) {
	c.nodeLock.Lock()
	defer c.nodeLock.Unlock()

	c.nodes[node] = lastTime
}
