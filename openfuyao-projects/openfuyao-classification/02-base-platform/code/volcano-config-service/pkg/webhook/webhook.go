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

// Package webhook defined mutate and validate rules
package webhook

import (
	"encoding/json"
	"fmt"
	"net/http"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"volcano-config-service/pkg/zlog"
)

const (
	// NumaTopologyPolicyAnnotation Observerd Annotation
	NumaTopologyPolicyAnnotation = "volcano.sh/numa-topology-policy"
)

// WhServer indicate webhook server
type WhServer struct {
	Server *http.Server
}

// patchOperation 定义patch操作结构
type patchOperation struct {
	Op    string      `json:"op"`
	Path  string      `json:"path"`
	Value interface{} `json:"value"`
}

// Serve 核心业务逻辑实现
func (whsvr *WhServer) Serve(w http.ResponseWriter, r *http.Request) {
	// AdmissionReview describes an admission review request/response.
	admissionReviewReq := &admissionv1.AdmissionReview{}
	if err := json.NewDecoder(r.Body).Decode(admissionReviewReq); err != nil {
		zlog.Errorf("Can't get request: %v", err)
		return
	}

	contentType := r.Header.Get("Content-Type")
	if contentType != "application/json" {
		zlog.Errorf("Content-Type=%s, expect application/json", contentType)
		http.Error(w, "invalid Content-Type, expect `application/json`", http.StatusUnsupportedMediaType)
		return
	}

	// AdmissionReview describes an admission review request/response.
	admissionReviewResp := &admissionv1.AdmissionReview{}

	// 根据path区分
	if r.URL.Path == "/mutate" {
		admissionReviewResp = whsvr.mutate(admissionReviewReq)
	} else {
		zlog.Errorf("can't identify current path")
		return
	}

	if admissionReviewResp == nil {
		zlog.Errorf("admissionReviewResp is nil ")
		return
	}

	// 回写response
	if err := json.NewEncoder(w).Encode(admissionReviewResp); err != nil {
		zlog.Errorf("can't write response: %v", err)
		return
	}
}

// mutate 处理变更逻辑
func (whsvr *WhServer) mutate(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionReview {
	zlog.Infof("Mutating %s %s/%s", ar.Request.Kind.Kind, ar.Request.Namespace, ar.Request.Name)

	patches, err := whsvr.generatePatches(ar.Request)
	if err != nil {
		zlog.Errorf("Failed to mutate %s: %v", ar.Request.Kind.Kind, err)
		return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
	}

	return whsvr.buildAdmissionResponse(ar, patches)
}

// generatePatches 根据资源类型生成对应的patch操作
func (whsvr *WhServer) generatePatches(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	switch req.Kind.Kind {
	case "Pod":
		return whsvr.mutatePod(req)
	case "Deployment":
		return whsvr.mutateDeployment(req)
	case "StatefulSet":
		return whsvr.mutateStatefulSet(req)
	case "DaemonSet":
		return whsvr.mutateDaemonSet(req)
	case "Job":
		return whsvr.mutateJob(req)
	case "CronJob":
		return whsvr.mutateCronJob(req)
	default:
		zlog.Infof("Resource type %s not supported for mutation", req.Kind.Kind)
		return nil, nil // 返回nil表示不需要patch，但不是错误
	}
}

// buildAdmissionResponse 构建AdmissionReview响应
func (whsvr *WhServer) buildAdmissionResponse(ar *admissionv1.AdmissionReview,
	patches []patchOperation) *admissionv1.AdmissionReview {
	response := &admissionv1.AdmissionResponse{
		UID:     ar.Request.UID,
		Allowed: true,
	}

	// 如果有patch操作，序列化并添加到响应中
	if len(patches) > 0 {
		patchBytes, err := json.Marshal(patches)
		if err != nil {
			zlog.Errorf("Failed to marshal patches: %v", err)
			return ToAdmissionReview(ToAdmissionErrorResponse(ar, err))
		}

		patchType := admissionv1.PatchTypeJSONPatch
		response.Patch = patchBytes
		response.PatchType = &patchType
	}

	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: response,
	}
}

func (whsvr *WhServer) createAdmissionReview(uid types.UID) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
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

// mutatePod 处理Pod的mutation
func (whsvr *WhServer) mutatePod(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	var pod corev1.Pod
	if err := json.Unmarshal(req.Object.Raw, &pod); err != nil {
		return nil, fmt.Errorf("failed to unmarshal pod: %v", err)
	}

	return whsvr.mutatePodSpec(&pod.ObjectMeta, &pod.Spec)
}

// mutateDeployment 处理Deployment的mutation
func (whsvr *WhServer) mutateDeployment(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	var deployment appsv1.Deployment
	if err := json.Unmarshal(req.Object.Raw, &deployment); err != nil {
		return nil, fmt.Errorf("failed to unmarshal deployment: %v", err)
	}

	return whsvr.mutatePodTemplateSpec(&deployment.Spec.Template)
}

// mutateStatefulSet 处理StatefulSet的mutation
func (whsvr *WhServer) mutateStatefulSet(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	var statefulSet appsv1.StatefulSet
	if err := json.Unmarshal(req.Object.Raw, &statefulSet); err != nil {
		return nil, fmt.Errorf("failed to unmarshal statefulset: %v", err)
	}

	return whsvr.mutatePodTemplateSpec(&statefulSet.Spec.Template)
}

// mutateDaemonSet 处理DaemonSet的mutation
func (whsvr *WhServer) mutateDaemonSet(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	var daemonSet appsv1.DaemonSet
	if err := json.Unmarshal(req.Object.Raw, &daemonSet); err != nil {
		return nil, fmt.Errorf("failed to unmarshal daemonset: %v", err)
	}

	return whsvr.mutatePodTemplateSpec(&daemonSet.Spec.Template)
}

// mutateJob 处理Job的mutation
func (whsvr *WhServer) mutateJob(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	var Job batchv1.Job
	if err := json.Unmarshal(req.Object.Raw, &Job); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Job: %v", err)
	}

	return whsvr.mutatePodTemplateSpec(&Job.Spec.Template)
}

// mutateCronJob 处理Job的mutation
func (whsvr *WhServer) mutateCronJob(req *admissionv1.AdmissionRequest) ([]patchOperation, error) {
	var CronJob batchv1.CronJob
	if err := json.Unmarshal(req.Object.Raw, &CronJob); err != nil {
		return nil, fmt.Errorf("failed to unmarshal CronJob: %v", err)
	}

	return whsvr.mutatePodTemplateSpec(&CronJob.Spec.JobTemplate.Spec.Template)
}

// mutatePodTemplateSpec 处理PodTemplateSpec
func (whsvr *WhServer) mutatePodTemplateSpec(template *corev1.PodTemplateSpec) ([]patchOperation, error) {
	return whsvr.mutatePodSpec(&template.ObjectMeta, &template.Spec)
}

// mutatePodSpec 核心的Pod规格mutation逻辑 - 修改为只在没有业务类型注解时自动推断
func (whsvr *WhServer) mutatePodSpec(meta *metav1.ObjectMeta, spec *corev1.PodSpec) ([]patchOperation, error) {
	var patches []patchOperation

	// 确保annotations存在
	if meta.Annotations == nil {
		patches = append(patches, patchOperation{
			Op:    "add",
			Path:  "/metadata/annotations",
			Value: map[string]string{},
		})
		meta.Annotations = make(map[string]string)
	}

	// 检查是否已经有业务类型注解
	if numaType, exists := meta.Annotations[NumaTopologyPolicyAnnotation]; exists {
		// 存在numa亲和策略，不存在则无需处理
		zlog.Infof("Found numa-topology-policy annotation: %s", numaType)

		currentScheduler := spec.SchedulerName

		// 已经是volcano => 不处理
		if currentScheduler == "volcano" {
			zlog.Infof("Pod already uses volcano scheduler, skip modification")
			return patches, nil
		}

		// 检查是否已设置调度器且不是default-scheduler => 不覆盖
		if currentScheduler != "" && currentScheduler != "default-scheduler" {
			zlog.Infof("Pod uses custom scheduler %s, skip modification", currentScheduler)
			return patches, nil
		}

		// 需要设置为volcano的情况：
		// - schedulerName为空
		// - schedulerName为"default-scheduler"
		var op string
		if currentScheduler == "" {
			op = "add"
		} else {
			op = "replace"
		}

		patches = append(patches, patchOperation{
			Op:    op,
			Path:  "/spec/schedulerName",
			Value: "volcano",
		})

		zlog.Infof("Setting schedulerName from '%s' to 'volcano' for numa-aware scheduling", currentScheduler)
	}

	return patches, nil
}

// toAdmissionErrorResponse transforms an error into an admission response with error.
func toAdmissionErrorResponse(uid types.UID, err error) *admissionv1.AdmissionResponse {
	zlog.Errorf("admission webhook error: %s", err)
	return &admissionv1.AdmissionResponse{
		UID: uid,
		Result: &metav1.Status{
			Message: err.Error(),
			Code:    http.StatusBadRequest,
		},
	}
}

// ToAdmissionErrorResponse transforms an error into an admission response with error.
func ToAdmissionErrorResponse(ar *admissionv1.AdmissionReview, err error) *admissionv1.AdmissionResponse {
	return toAdmissionErrorResponse(ar.Request.UID, err)
}

// ToAdmissionReview indicate response type.
func ToAdmissionReview(resp *admissionv1.AdmissionResponse) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			Kind:       "AdmissionReview",
			APIVersion: "admission.k8s.io/v1",
		},
		Response: resp,
	}
}
