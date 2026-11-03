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
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/cmd/colocation-overquota-agent/apps"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/exporter"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/metric"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

const (
	defaultShutdownTimeout = 5 * time.Second
)

// MetricsReporter is a metrics reporter.
type MetricsReporter struct {
	ctx             context.Context
	config          *apps.Configuration
	kubeletClient   *metric.KubeletClient
	apiserverClient *kubernetes.Clientset
	storage         metric.PodStorage
	reportBuffer    *Buffer
	buffers         sync.Pool
	exporterServer  *http.Server

	lastUsageTimestampCollected metav1.Time
}

// NewMetricsReporter creates a new metrics reporter.
func NewMetricsReporter(
	ctx context.Context,
	cfg *rest.Config,
	config *apps.Configuration,
) (*MetricsReporter, error) {
	client, err := newKubeletClient(config, cfg)
	if err != nil {
		return nil, err
	}

	apiserverClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("create config failed. %v", err)
	}
	reporter := &MetricsReporter{
		ctx:             ctx,
		config:          config,
		kubeletClient:   client,
		apiserverClient: apiserverClient,
		storage:         metric.NewPodStorage(config.Node, config.MetricsScrapePeriod),
		reportBuffer:    NewReportBuffer(config.ReportsBufCapacity),
		buffers: sync.Pool{
			New: func() interface{} {
				buf := make([]byte, 10e3)
				return &buf
			},
		},
	}

	go func() {
		err = reporter.runServe()
	}()
	if err != nil {
		return nil, fmt.Errorf("create metrics reporter failed. %v", err)
	}

	go reporter.runScrape()

	go reporter.runUpload()

	if config.EnableIncludeExporter {
		go reporter.runExport()
		go reporter.reclaimMetricsReporter()
		server, err := exporter.NewIncludeExporter(config.ExporterPort)
		if err != nil {
			return nil, fmt.Errorf("create include exporter failed. %v", err)
		}
		reporter.exporterServer = server
	}
	return reporter, nil
}

func newKubeletClient(config *apps.Configuration, cfg *rest.Config) (*metric.KubeletClient, error) {
	scheme := "https"
	defaultPort := config.KubeletSecureMetricsPort

	if config.InsecureKubeletMetricsTLS {
		klog.V(common.AdvanceDebugLog).Info("insecure kubelet metrics port is enabled")
		scheme = "http"
		defaultPort = config.KubeletInsecureMetricsPort
	}
	client, err := metric.NewForConfig(&metric.KubeletClientConfig{
		Client:              cfg,
		AddressTypePriority: []corev1.NodeAddressType{corev1.NodeHostName},
		Scheme:              scheme,
		DefaultPort:         defaultPort,
		UseNodeStatusPort:   false,
	})
	if err != nil {
		return nil, fmt.Errorf("create config failed. %v", err)
	}
	return client, nil
}

func (mr *MetricsReporter) runScrape() {
	ticker := time.NewTicker(mr.config.MetricsScrapePeriod)
	defer ticker.Stop()
	mr.scrapeTick(time.Now())

	for {
		select {
		case startTime := <-ticker.C:
			mr.scrapeTick(startTime)
		case <-mr.ctx.Done():
			return
		}
	}
}

func (mr *MetricsReporter) scrapeTick(startTime time.Time) {
	ctx, cancelTimeout := context.WithTimeout(mr.ctx, mr.config.MetricsScrapePeriod)
	defer cancelTimeout()
	node := &corev1.Node{
		Status: corev1.NodeStatus{Addresses: []corev1.NodeAddress{{Type: corev1.NodeHostName, Address: mr.config.KubeletIP}}},
	}
	currentMetrics, err := mr.kubeletClient.GetMetrics(ctx, node)
	if err != nil {
		klog.Errorf("scrapeTick: can not scrape metrics from kubelet. %v", err)
		return
	}
	if len(currentMetrics.Pods) != len(currentMetrics.Containers) {
		klog.V(common.TraceDebugLog).InfoS("Scraping metrics",
			"metrics.pods.len", len(currentMetrics.Pods), "metrics.containers.len", len(currentMetrics.Containers))
		klog.V(common.MicroscopicDebugLog).Infof("metrics.pods:\n%v", metric.PrintNamespaceNamesForPods(currentMetrics.Pods))
		klog.V(common.MicroscopicDebugLog).Infof("metrics.containers:\n%v",
			metric.PrintNamespaceNamesForContainers(currentMetrics.Containers))
	} else {
		klog.V(common.TraceDebugLog).InfoS("Scraping metrics", "metrics.pods.len", len(currentMetrics.Pods))
		klog.V(common.MicroscopicDebugLog).Infof("metrics.pods:\n%v", metric.PrintNamespaceNamesForPods(currentMetrics.Pods))
	}
	currentPods, err := mr.kubeletClient.GetOnlinePods(ctx, node)
	if err != nil {
		klog.Errorf("Metric reporter scrape pods from kubelet failed. %v", err)
		return
	}
	klog.V(common.TraceDebugLog).InfoS("Scraping pods", "kubelet.pods.len", len(currentPods))
	klog.V(common.MicroscopicDebugLog).Infof("kubelet.pods:\n%v", metric.PrintObjMeta(currentPods))
	mr.storage.Store(currentMetrics)
	klog.V(common.MicroscopicDebugLog).InfoS("Stored metrics")
	report, err := mr.storage.GetLastMetrics(currentPods...)
	if err != nil {
		klog.Errorf("Metric reporter get last metrics failed. %v", err)
		return
	}
	klog.V(common.TraceDebugLog).InfoS("Transform metrics to report", "report.Timestamp",
		report.Timestamp, "report.Window",
		report.Window, "report.pods.len", len(report.Pods), "report.containers.len", len(report.Containers))
	if len(report.Containers) != 0 || len(report.Pods) != 0 {
		klog.V(common.MicroscopicDebugLog).Infof("report.pods:\n%v",
			metric.PrintNamespaceNamesForResList2(report.Pods))
		klog.V(common.MicroscopicDebugLog).Infof("report.containers:\n%v",
			metric.PrintNamespaceNamesForResList(report.Containers))
		mr.reportBuffer.Put(*report)
		klog.V(common.MicroscopicDebugLog).InfoS("Buffered report")
	}
	scrapeTime := time.Since(startTime)
	klog.V(common.AdvanceDebugLog).Infof("scrapeTick: Scrape metrics cycle complete, elapsed: %v", scrapeTime)
}

func (mr *MetricsReporter) runServe() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/usage", mr.getLastReporter)
	mux.HandleFunc("/debug/pods", mr.getKubeletPods)
	if err := http.ListenAndServe(mr.config.ServerAddress, mux); err != nil {
		klog.Errorf("fail to start metrics reporter server in %v. err: %v", mr.config.ServerAddress, err)
		return err
	}
	return nil
}

func (mr *MetricsReporter) getKubeletPods(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		pods, err := mr.kubeletClient.GetPods(mr.ctx, &corev1.Node{
			Status: corev1.NodeStatus{
				Addresses: []corev1.NodeAddress{
					{Type: corev1.NodeHostName, Address: mr.config.KubeletIP},
				},
			},
		})
		if err != nil {
			klog.Errorf("Fail to et pods from kubelet. %v", err)
			http.Error(w, "Fail to get pods from kubelet", http.StatusInternalServerError)
			return
		}

		type Metadata struct {
			Namespace string `json:"namespace"`
			Name      string `json:"name"`
		}
		var podsResponse []*Metadata
		for _, pod := range pods {
			podsResponse = append(podsResponse, &Metadata{Namespace: pod.Namespace, Name: pod.Name})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(podsResponse)
		w.WriteHeader(http.StatusOK)
	default:
		http.Error(w, fmt.Sprintf("Method %v is not supported!", r.Method), http.StatusMethodNotAllowed)
		return
	}
}

func (mr *MetricsReporter) getLastReporter(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		lastReport, err := mr.reportBuffer.Peek()
		if err != nil {
			http.Error(w, "usage metrics are empty", http.StatusOK)
			return
		}
		klog.V(common.GeneralDebugLog).Infof("%#v", lastReport)

		type NodeMetricsSummary struct {
			Timestamp metav1.Time                           `json:"timestamp"`
			Window    metav1.Duration                       `json:"window"`
			Pods      map[metric.PodRef]corev1.ResourceList `json:"pods"`
		}
		var summary = &NodeMetricsSummary{
			Timestamp: lastReport.Timestamp,
			Window:    lastReport.Window,
			Pods:      make(map[metric.PodRef]corev1.ResourceList),
		}
		for podRef, containers := range lastReport.Containers {
			var podRes = corev1.ResourceList{}
			for _, resource := range containers {
				podRes = metric.SumResourceList(podRes, resource)
			}
			summary.Pods[podRef] = podRes
		}
		data, err := json.Marshal(summary)
		if err != nil {
			klog.Errorf("%v", err)
			return
		}
		klog.V(common.GeneralDebugLog).Infof("data=%v", string(data))

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(summary)
		w.WriteHeader(http.StatusOK)
	default:
		http.Error(w, fmt.Sprintf("Method %v is not supported!", r.Method), http.StatusMethodNotAllowed)
		return
	}
}

func (mr *MetricsReporter) runUpload() {
	ticker := time.NewTicker(mr.config.ReportsUploadPeriod)
	defer ticker.Stop()
	mr.uploadTick(time.Now())

	for {
		select {
		case startTime := <-ticker.C:
			mr.uploadTick(startTime)
		case <-mr.ctx.Done():
			return
		}
	}
}

func (mr *MetricsReporter) isColocationNode() bool {
	node, err := mr.apiserverClient.CoreV1().Nodes().Get(mr.ctx, mr.config.Node, metav1.GetOptions{})
	if err != nil {
		klog.Errorf("runUpload: can not get node info from apiserver. %v", err)
		return false
	}

	// 仅在开启混部的节点执行上报工作
	return utils.IsColocationNode(node)
}

func (mr *MetricsReporter) uploadTick(startTime time.Time) {
	if !mr.isColocationNode() {
		return
	}
	resp, success := mr.uploadMetrics(startTime)
	if !success {
		return
	}
	defer resp.Body.Close()
	bp, ok := mr.buffers.Get().(*[]byte)
	if !ok {
		klog.Errorf("buffers.Get() is not *[]byte,got type to %T", mr.buffers.Get())
		return
	}
	b := *bp
	defer func() {
		*bp = b
		mr.buffers.Put(bp)
	}()
	buf := bytes.NewBuffer(b)
	buf.Reset()
	_, err := io.Copy(buf, resp.Body)
	if err != nil {
		klog.Errorf("uploadTick: failed to read http response body. %v", err)
		return
	}
	b = buf.Bytes()

	var r = &metric.NodeUsageResponse[metav1.Time]{}
	if err := json.Unmarshal(b, r); err != nil {
		klog.Errorf("uploadTick: can not unmarshal http response body to NodeUsageResponse. %v", err)
		return
	}
	if r.Code != codeSuccess {
		if r.Code >= codeUsageTimestampExpired {
			mr.lastUsageTimestampCollected = r.Data
			klog.Errorf("uploadTick: receive report response from server. code=%v, msg=%v, last collected usage timestamp:%v",
				r.Code, r.Msg, mr.lastUsageTimestampCollected)
		} else {
			klog.Errorf("uploadTick: receive report response from server. code=%v, msg=%v", r.Code, r.Msg)
		}
		return
	}

	mr.lastUsageTimestampCollected = r.Data
	uploadTime := time.Since(startTime)
	klog.V(common.VerboseDebugLog).Infof("upload cycle complete, last ts:%v, elapsed:%v",
		mr.lastUsageTimestampCollected, uploadTime)
}

func (mr *MetricsReporter) uploadMetrics(startTime time.Time) (*http.Response, bool) {
	reportsData, err := mr.peekReportsInBytes()
	if err != nil {
		return nil, false
	}

	resp := mr.doMetricsUpload(err, reportsData)
	if resp == nil {
		return nil, false
	}
	if resp.StatusCode != http.StatusOK {
		klog.Errorf("uploadTick: http request failed, status: %q", resp.Status)
		return resp, false
	}
	return resp, true
}
func (mr *MetricsReporter) peekReportsInBytes() ([]byte, error) {
	allReports, err := mr.reportBuffer.PeekAll(mr.lastUsageTimestampCollected)
	if err != nil {
		klog.Errorf("uploadTick: can not peek stored reports. %v", err)
		return nil, err
	}
	if len(allReports) == 0 {
		return nil, fmt.Errorf("the length of reportBuffer is 0")
	}

	reportsData, err := json.Marshal(allReports)
	if err != nil {
		klog.Errorf("uploadTick: marshal all reports data failed. %v", err)
		return nil, err
	}
	return reportsData, nil
}

func (mr *MetricsReporter) doMetricsUpload(err error, reportsData []byte) *http.Response {
	client := http.Client{Timeout: common.DefaultHttpClientTimeout}
	req, err := http.NewRequest("POST", (&url.URL{
		Scheme: "http",
		Host:   mr.config.UsageReportAddr,
		Path:   "/v1/metrics/usage",
	}).String(), bytes.NewBuffer(reportsData))
	if err != nil {
		klog.Errorf("uploadTick: build http request failed. %v", err)
		return nil
	}
	resp, err := client.Do(req.WithContext(mr.ctx))
	if err != nil {
		klog.Errorf("uploadTick: do http post request failed. %v", err)
		return nil
	}
	return resp
}

func (mr *MetricsReporter) runExport() {
	ticker := time.NewTicker(mr.config.ExporterPeriod)
	defer ticker.Stop()
	mr.exportTick()

	for {
		select {
		case <-ticker.C:
			mr.exportTick()
		case <-mr.ctx.Done():
			return
		}
	}
}

func (mr *MetricsReporter) exportTick() {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: mr.config.Node,
		},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{
				{Type: corev1.NodeHostName, Address: mr.config.KubeletIP},
			},
		},
	}

	pods, err := mr.kubeletClient.GetPodMetrics(mr.ctx, node)
	if err != nil {
		klog.Errorf("runExport: can not get offline pods. %v", err)
		return
	}
	exporter.CollectPodMetrics(pods)

	get, err := mr.apiserverClient.CoreV1().Nodes().Get(mr.ctx, mr.config.Node, metav1.GetOptions{})
	if err != nil {
		klog.Errorf("runExport: can not get node info from apiserver. %v", err)
		return
	}
	if get != nil {
		exporter.CollectNodeMetrics(get)
	}
}

// reclaimMetricsReporter 回收已删除pod指标
func (mr *MetricsReporter) reclaimMetricsReporter() {

	// 等待pod信息采集
	time.Sleep(mr.config.ExporterPeriod)
	klog.V(common.GeneralDebugLog).Infof("waiting pods info collected")
	factory := informers.NewSharedInformerFactoryWithOptions(mr.apiserverClient,
		mr.config.ExporterPeriod,
		informers.WithTweakListOptions(func(options *metav1.ListOptions) {
			// 设置字段选择器只监听当前node上的Pod
			fieldSelector := fields.OneTermEqualSelector(common.SpecNodeNameField, mr.config.Node).String()
			options.FieldSelector = fieldSelector
		}))
	factory.Core().V1().Pods().Informer().AddEventHandler(&cache.ResourceEventHandlerFuncs{
		DeleteFunc: mr.DeleteMetricsReporter,
	})
	factory.Start(mr.ctx.Done())
}

// DeleteMetricsReporter 删除pod指标
func (mr *MetricsReporter) DeleteMetricsReporter(obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		klog.Errorf("DeleteMetricsReporter: can not convert to pod. %v", obj)
		return
	}
	qoslevel := utils.GetQosLevelByPod(pod)

	pe := &exporter.PodExport{
		Namespace: pod.Namespace,
		Name:      pod.Name,
		NodeName:  mr.config.Node,
		QosLevel:  string(qoslevel),
	}
	exporter.DeletePodMetrics(pe)
}

// Shutdown 关闭exporter server
func (r *MetricsReporter) Shutdown() {
	if r.exporterServer != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), defaultShutdownTimeout)
		defer cancel()
		if err := r.exporterServer.Shutdown(shutdownCtx); err != nil {
			klog.Errorf("Failed to shutdown exporter server: %v", err)
		}
	}
}
