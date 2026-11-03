/*
// Copyright 2021 The Kubernetes Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


Original file: https://github.com/kubernetes-sigs/metrics-server/tree/helm-chart-3.6.0/pkg/scraper/client/resource/client.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
Modifications:
Modified function GetMetrics
Added function GetOnlinePods GetPodMetrics getPodMetrics getPodsSummary buildPodExporter

*/

package metric

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"sigs.k8s.io/metrics-server/pkg/utils"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/exporter"
	"openfuyao.com/colocation-management/pkg/common"
	colocationutils "openfuyao.com/colocation-management/pkg/utils"
)

// KubeletClient is a client to query kubelet metrics.
type KubeletClient struct {
	defaultPort       int
	useNodeStatusPort bool
	client            *http.Client
	scheme            string
	addrResolver      utils.NodeAddressResolver
	buffers           sync.Pool

	store *podStorage
}

// KubeletClientConfig is the configuration for a KubeletClient.
type KubeletClientConfig struct {
	Client              *rest.Config
	AddressTypePriority []corev1.NodeAddressType
	Scheme              string
	DefaultPort         int
	UseNodeStatusPort   bool
}

// NewForConfig creates a new KubeletClient for the given config.
func NewForConfig(config *KubeletClientConfig) (*KubeletClient, error) {
	var transport http.RoundTripper
	var err error
	if config.Client != nil {
		// 因为各种使用场景下kubelet本身的证书签发方各有不同，此处选择跳过对kubelet证书的校验
		// kubelet认证本客户端通过 bearerToken 的形式，获取到serviceaccount之后依据k8s的rbac进行鉴权
		config.Client.TLSClientConfig.Insecure = true
		config.Client.TLSClientConfig.CAData = nil
		config.Client.TLSClientConfig.CAFile = ""
		transport, err = rest.TransportFor(config.Client)
		if err != nil {
			return nil, fmt.Errorf("unable to construct transport: %v", err)
		}
	} else {
		transport = http.DefaultTransport
	}

	c := &http.Client{
		Transport: transport,
		Timeout:   common.DefaultHttpClientTimeout,
	}
	resolver := utils.NewPriorityNodeAddressResolver(config.AddressTypePriority)
	return newClient(c, resolver, config.DefaultPort, config.Scheme, config.UseNodeStatusPort), nil
}

func newClient(c *http.Client, resolver utils.NodeAddressResolver,
	defaultPort int, scheme string, useNodeStatusPort bool) *KubeletClient {
	return &KubeletClient{
		addrResolver:      resolver,
		defaultPort:       defaultPort,
		client:            c,
		scheme:            scheme,
		useNodeStatusPort: useNodeStatusPort,
		buffers: sync.Pool{
			New: func() interface{} {
				buf := make([]byte, 10e3)
				return &buf
			},
		},
	}
}

// GetMetrics implements client.KubeletMetricsGetter
func (kc *KubeletClient) GetMetrics(ctx context.Context, node *corev1.Node) (*MetricsBatch, error) {
	port := kc.defaultPort

	url, err := kc.urlBuild("/metrics/resource", node, port)
	if err != nil {
		return nil, err
	}
	return kc.getMetrics(ctx, url.String(), node.Name)
}

func (kc *KubeletClient) getMetrics(ctx context.Context, url, nodeName string) (*MetricsBatch, error) {
	requestTime := time.Now()
	b, err := kc.doRequest(ctx, url)
	if err != nil {
		return nil, err
	}
	ms, err := decodeBatch(b, requestTime, nodeName)
	if err != nil {
		return nil, err
	}
	return ms, nil
}

// GetPods implements client.KubeletMetricsGetter
func (kc *KubeletClient) GetPods(ctx context.Context, node *corev1.Node) ([]*metav1.PartialObjectMetadata, error) {
	port := kc.defaultPort
	url, err := kc.urlBuild("/pods", node, port)
	if err != nil {
		return nil, err
	}
	return kc.getPods(ctx, url.String())
}

func (kc *KubeletClient) getPods(ctx context.Context, url string) ([]*metav1.PartialObjectMetadata, error) {
	b, err := kc.doRequest(ctx, url)
	if err != nil {
		return nil, err
	}
	podList := corev1.PodList{}
	if err := json.Unmarshal(b, &podList); err != nil {
		return nil, err
	}
	pods := make([]*metav1.PartialObjectMetadata, 0)
	for _, item := range podList.Items {
		pods = append(pods, &metav1.PartialObjectMetadata{
			TypeMeta:   item.TypeMeta,
			ObjectMeta: item.ObjectMeta,
		})
		klog.V(common.MicroscopicDebugLog).InfoS(
			"get pods request resource",
			"pod.namespace", item.Namespace,
			"pod.name", item.Name,
			"container.name", item.Spec.Containers[0].Name,
			"request", item.Spec.Containers[0].Resources.Requests,
			"limits", item.Spec.Containers[0].Resources.Limits)
	}
	return pods, nil
}

// GetOnlinePods get online pods
func (kc *KubeletClient) GetOnlinePods(ctx context.Context,
	node *corev1.Node) ([]*metav1.PartialObjectMetadata, error) {
	podsMeta, err := kc.GetPods(ctx, node)
	if err != nil {
		return nil, err
	}

	results := make([]*metav1.PartialObjectMetadata, 0)
	for _, podMeta := range podsMeta {
		if !(colocationutils.OfflinePodMeta(podMeta) || colocationutils.HLSPodMeta(podMeta)) {
			results = append(results, podMeta)
		}
	}
	return results, nil
}

// GetPodMetrics get pod metrics
func (kc *KubeletClient) GetPodMetrics(ctx context.Context, node *corev1.Node) ([]*exporter.PodExport, error) {
	port := kc.defaultPort
	url, err := kc.urlBuild("/pods", node, port)
	if err != nil {
		return nil, err
	}
	exporters, err := kc.getPodMetrics(ctx, url.String(), node.Name)
	if err != nil {
		return nil, err
	}

	url.Path = "/stats/summary"
	exporters, err = kc.getPodsSummary(ctx, url.String(), exporters)
	if err != nil {
		return nil, err
	}
	return exporters, nil
}

func (kc *KubeletClient) getPodMetrics(ctx context.Context, url, node string) ([]*exporter.PodExport, error) {
	b, err := kc.doRequest(ctx, url)
	if err != nil {
		return nil, err
	}
	podList := corev1.PodList{}
	if err := json.Unmarshal(b, &podList); err != nil {
		return nil, err
	}
	pods, err := buildPodExporter(podList, node)
	if err != nil {
		return nil, err
	}
	return pods, nil
}

func (kc *KubeletClient) getPodsSummary(ctx context.Context,
	url string, exporters []*exporter.PodExport) ([]*exporter.PodExport, error) {
	b, err := kc.doRequest(ctx, url)
	if err != nil {
		return nil, err
	}
	summary := exporter.KubeletStatsSummary{}
	if err := json.Unmarshal(b, &summary); err != nil {
		return nil, err
	}

	for _, exporter := range exporters {
		for _, pod := range summary.Pods {
			if exporter.Name == pod.PodRef.Name && exporter.Namespace == pod.PodRef.Namespace {
				// 实时值 单位核
				exporter.CpuUsage = float64(pod.CPU.UsageNanoCores) / common.NanoCoresPerCore
				// 实时值 单位字节
				exporter.MemoryUsage = float64(pod.Memory.UsageBytes)
			}
		}
	}
	return exporters, nil
}

func (kc *KubeletClient) doRequest(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	response, err := kc.client.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed, status: %q", response.Status)
	}
	bp, ok := kc.buffers.Get().(*[]byte)
	if !ok {
		return nil, fmt.Errorf("failed to get buffer from pool ")
	}
	b := *bp
	defer func() {
		*bp = b
		kc.buffers.Put(bp)
	}()
	buf := bytes.NewBuffer(b)
	buf.Reset()
	_, err = io.Copy(buf, response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body - %v", err)
	}
	b = buf.Bytes()
	return b, err
}

func (kc *KubeletClient) urlBuild(path string, node *corev1.Node, port int) (url.URL, error) {
	nodeStatusPort := int(node.Status.DaemonEndpoints.KubeletEndpoint.Port)
	if kc.useNodeStatusPort && nodeStatusPort != 0 {
		port = nodeStatusPort
	}
	addr, err := kc.addrResolver.NodeAddress(node)
	if err != nil {
		return url.URL{}, err
	}
	url := url.URL{
		Scheme: kc.scheme,
		Host:   net.JoinHostPort(addr, strconv.Itoa(port)),
		Path:   path,
	}
	return url, nil
}

func buildPodExporter(podList corev1.PodList, node string) ([]*exporter.PodExport, error) {
	pods := make([]*exporter.PodExport, 0)
	for _, pod := range podList.Items {
		namespace := pod.Namespace
		podName := pod.Name
		nodeName := node
		qoslevel := colocationutils.GetQosLevelByPod(&pod)
		tmpPodExport := exporter.PodExport{
			Namespace: namespace,
			Name:      podName,
			NodeName:  nodeName,
			QosLevel:  string(qoslevel),
		}
		switch qoslevel {
		case common.QosLs:
			pods = append(pods, &tmpPodExport)
		case common.QosHLS:
			pods = append(pods, &tmpPodExport)

		case common.QosBe:
			beCpuReq, beMemoryReq, err := getBeRequst(pod, namespace, podName)
			if err != nil {
				klog.Errorf("get be cpu and memory request in Pod %s/%s error:%v", namespace, podName, err)
				return pods, err
			}
			tmpPodExport.BeCpuRequest = beCpuReq
			tmpPodExport.BeMemoryRequest = beMemoryReq
			pods = append(pods, &tmpPodExport)
		case common.QosNone:
			pods = append(pods, &tmpPodExport)
		default:
			continue
		}
	}
	return pods, nil
}

func getBeRequst(pod corev1.Pod, namespace, podName string) (beCpuReq, beMemoryReq float64, err error) {
	beCpuReq = float64(0)    // 默认值
	beMemoryReq = float64(0) // 默认值
	// 遍历 Pod 的容器并收集指标
	for _, container := range pod.Spec.Containers {
		if req, ok := container.Resources.Requests[common.ExtenderResourceCPU]; ok {
			float := float64(req.MilliValue())
			beCpuReq += float / 1000 // 转为核
		}
		if req, ok := container.Resources.Requests[common.ExtenderResourceMemory]; ok {
			float := float64(req.Value())
			beMemoryReq += float
		}
	}
	return beCpuReq, beMemoryReq, nil
}
