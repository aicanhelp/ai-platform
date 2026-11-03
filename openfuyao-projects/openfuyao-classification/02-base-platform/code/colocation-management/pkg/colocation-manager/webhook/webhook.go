/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package webhook

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/webhook/helper"
	"openfuyao.com/colocation-management/pkg/colocation-manager/webhook/pod"
	"openfuyao.com/colocation-management/pkg/common"
)

// qos level
// batch cpu
// nodeAnnotations scheduler时候计算 allocate update nri的时候

// WebhookServer webhook server
type WebhookServer struct {
	Server *http.Server
	config *apps.Configuration
	contxt context.Context
}

const (
	waitForSvcInterval = 5 * time.Second
	waitForSvcTimeout  = 2 * time.Minute
)

// NewWebhookServer 创建WebhookServer
func NewWebhookServer(ctx context.Context, config *apps.Configuration) *WebhookServer {

	pair, err := tls.LoadX509KeyPair(config.WebhookTlsCertFile, config.WebhookTlsKeyFile)
	if err != nil {
		klog.Errorf("Failed to load key pair: %v", err)
	}

	whsvr := &WebhookServer{
		contxt: ctx,
		config: config,
		Server: &http.Server{
			Addr:      fmt.Sprintf(":%v", config.WebhookPort),
			TLSConfig: &tls.Config{Certificates: []tls.Certificate{pair}},
		},
	}
	return whsvr
}

// Run 启动webhook server
func (whsvr *WebhookServer) Run() {
	mux := http.NewServeMux()
	mux.HandleFunc("/mutate", whsvr.Serve)
	mux.HandleFunc("/validate", whsvr.Serve)
	whsvr.Server.Handler = mux

	serverErr := make(chan error, 1)
	go func() {
		klog.Info("Starting webhook server...")
		if err := whsvr.Server.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
			klog.Errorf("Failed to listen and serve webhook server: %v", err)
			serverErr <- err
		}
	}()

	klog.Infof("检查webhook service binding情况")
	err := WaitForSvcRunning(waitForSvcInterval, waitForSvcTimeout,
		whsvr.config.WebhookSvcNamespace, whsvr.config.WebhookSvcName)
	if err != nil {
		return
	}

	select {
	case <-whsvr.contxt.Done():
		err = whsvr.Server.Shutdown(whsvr.contxt)
		if err != nil {
			klog.Errorf("Failed to shutdown webhook server: %v", err)
		}
		klog.Info("webhook server is closed")
		return
	case err := <-serverErr:
		klog.Errorf("webhook server error: %v", err)
		return
	default:
		klog.Info("Webhook server started successfully")
	}
}

// Serve 核心业务逻辑实现
func (whsvr *WebhookServer) Serve(w http.ResponseWriter, r *http.Request) {
	// AdmissionReview describes an admission review request/response.
	admissionReviewReq := &admissionv1.AdmissionReview{}
	if err := json.NewDecoder(r.Body).Decode(admissionReviewReq); err != nil {
		klog.Errorf("Can't get request: %v", err)
		return
	}

	contentType := r.Header.Get("Content-Type")
	if contentType != "application/json" {
		klog.Errorf("Content-Type=%s, expect application/json", contentType)
		http.Error(w, "invalid Content-Type, expect `application/json`", http.StatusUnsupportedMediaType)
		return
	}

	// AdmissionReview describes an admission review request/response.
	admissionReviewResp := &admissionv1.AdmissionReview{}
	// 根据path区分
	if r.URL.Path == "/mutate" {
		admissionReviewResp = whsvr.mutate(admissionReviewReq)
	} else if r.URL.Path == "/validate" {
		admissionReviewResp = whsvr.validate(admissionReviewReq)
	}

	if admissionReviewResp == nil {
		klog.Errorf("admissionReviewResp is nil ")
		return
	}

	// 回写response
	if err := json.NewEncoder(w).Encode(admissionReviewResp); err != nil {
		klog.Errorf("Can't write response: %v", err)
		return
	}
}

// mutate
func (whsvr *WebhookServer) mutate(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionReview {
	primt := newAdmissionReview(ar.Request.UID)

	if ar.Request.Operation == admissionv1.Delete {
		return &primt
	}

	switch ar.Request.Kind.Kind {
	case "Pod":
		return whsvr.handlePodMutation(ar)
	case "Node":
		// 添加处理逻辑
		return &primt
	default:
		return &primt
	}
}

func (whsvr *WebhookServer) handlePodMutation(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionReview {
	// 基础响应
	baseResponse := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: &admissionv1.AdmissionResponse{
			UID:     ar.Request.UID,
			Allowed: true,
		},
	}

	// 反序列化Pod
	rawPod := v1.Pod{}
	if err := json.Unmarshal(ar.Request.Object.Raw, &rawPod); err != nil {
		klog.Errorf("Can't unmarshal request: %v", err)
		return baseResponse
	}

	// 获取QoS级别
	qosLevel := ""
	if rawPod.GetAnnotations() != nil {
		qosLevel = rawPod.Annotations[common.QosClassAnnotationKey]
	}

	klog.Info("start to handle ", qosLevel, "pod")

	// 根据QoS级别分发处理
	switch qosLevel {
	case "HLS":
		return whsvr.handleHLSPod(ar, &rawPod)
	case "LS":
		return whsvr.handleLSPod(ar, &rawPod)
	case "BE":
		return whsvr.handleBEPod(ar, &rawPod)
	default:
		// 默认情况放行Pod
		return baseResponse
	}
}

func (whsvr *WebhookServer) handleHLSPod(ar *admissionv1.AdmissionReview, rawPod *v1.Pod) *admissionv1.AdmissionReview {
	// HLS Pod的特定处理逻辑
	// 例如：检查资源限制、添加特定标签等
	klog.Info("- start to verify hls pod resource")
	if err := pod.VerifyPodResources(rawPod, "hls"); err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	// 校验limits==requests,且cpu为整数
	valid, err := pod.VerifyResourceEquality(rawPod, "hls")
	if err != nil {
		klog.Info("Validation failed:", err)
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}
	if !valid {
		klog.Info("Validation euqality failed:", err)
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	klog.Info("- hls pod verification passed")
	mutators, err := pod.NewHLSPodMutators()
	if err != nil {
		klog.InfoS("failed create mutators")
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	return whsvr.applyPodMutations(ar, mutators)
}

// LS Pod的特定处理逻辑
func (whsvr *WebhookServer) handleLSPod(ar *admissionv1.AdmissionReview, rawPod *v1.Pod) *admissionv1.AdmissionReview {

	// 校验申请的资源
	if err := pod.VerifyPodResources(rawPod, "ls"); err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	// 校验limits==requests,且cpu为整数
	valid, err := pod.VerifyResourceEquality(rawPod, "ls")
	if err != nil {
		fmt.Println("Validation failed:", err)
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}
	if !valid {
		fmt.Println("Validation euqality failed:", err)
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	// 获取 LS Pod 的 mutators
	mutators, err := pod.NewLSPodMutators()
	if err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	return whsvr.applyPodMutations(ar, mutators)
}

// BE Pod的特定处理逻辑
// 检查扩展资源，如果申请的不是kube.io资源就拦截
// 是否需要保证limis==requests?
func (whsvr *WebhookServer) handleBEPod(ar *admissionv1.AdmissionReview, rawPod *v1.Pod) *admissionv1.AdmissionReview {
	if err := pod.VerifyPodResources(rawPod, "be"); err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	mutators, err := pod.NewBEPodMutators()
	if err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	return whsvr.applyPodMutations(ar, mutators)
}
func (whsvr *WebhookServer) applyPodMutations(ar *admissionv1.AdmissionReview,
	mutators *pod.PodMutators) *admissionv1.AdmissionReview {

	mutatedJSON, err := pod.MutatingPodsJson(ar.Request.Object.Raw, mutators)
	klog.Info("get mutatedJSON after mutation")
	if err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	klog.Infof("Mutated JSON: %s", string(mutatedJSON))

	resp, err := pod.MutatingAdmissionResponse(ar, mutatedJSON)
	if err != nil {
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	return ToAdmissionReview(resp)
}

// validate处理 未使用3
func (whsvr *WebhookServer) validate(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionReview {
	primt := newAdmissionReview(ar.Request.UID)

	// 根据不同的资源类型做处理
	switch ar.Request.Kind.Kind {
	case "Pod":
		// 添加处理逻辑
		return &primt
	case "Node":
		// 添加处理逻辑
		return &primt
	default:
		return &primt
	}
}

// ToAdmissionErrorResponse 将错误转换为 AdmissionResponse
func ToAdmissionErrorResponse(ar *admissionv1.AdmissionReview, err error) *admissionv1.AdmissionResponse {
	return helper.ToAdmissionErrorResponse(ar.Request.UID, err)
}

// ToAdmissionReview 将 AdmissionResponse 转换为 AdmissionReview
func ToAdmissionReview(resp *admissionv1.AdmissionResponse) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: resp,
	}
}

func newAdmissionReview(uid types.UID) admissionv1.AdmissionReview {
	return admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: &admissionv1.AdmissionResponse{
			UID:     uid,
			Allowed: true,
		},
	}
}
