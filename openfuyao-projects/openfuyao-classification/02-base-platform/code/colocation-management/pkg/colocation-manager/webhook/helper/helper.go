/*
* Copyright 2022 Xabier Larrakoetxea
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Original file: https://github.com/slok/kubewebhook/blob/master/pkg/webhook/internal/helpers/helpers.go
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* Modifications:
* - By YuXiang Guo on 2024-11-25: Removed some codes and kept those would be used by this project
 */

// Package helper
package helper

import (
	"fmt"
	"reflect"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
)

var (
	// 错误信息
	errEmptyInput      = fmt.Errorf("rawJSON输入为空")
	errWhitespaceInput = fmt.Errorf("rawJSON输入只包含空白字符")
	errTypeAssertion   = fmt.Errorf("无法将对象转换为runtime.Object")
	errDeserialization = fmt.Errorf("反序列化失败")
)

// ObjectCreator 接口定义了从原始JSON数据创建特定类型对象的能力
type ObjectCreator interface {
	NewObject(rawJSON []byte) (runtime.Object, error)
}

// staticObjectCreator 结构体用于根据预定义的Kubernetes对象类型和反序列化器创建对象
type staticObjectCreator struct {
	// 存储Kubernetes对象的反射类型
	objType reflect.Type
	// 用于JSON反序列化的解码器
	deserializer runtime.Decoder
}

// NewStaticObjectCreator 创建一个静态对象创建器，无需推断类型，它会根据接收到的对象类型创建相同类型的新对象
func NewStaticObjectCreator(obj metav1.Object) ObjectCreator {
	scheme := runtime.NewScheme()
	serializationCodecs := serializer.NewCodecFactory(scheme)
	objectKind := extractK8sObjectType(obj)
	deserializer := serializationCodecs.UniversalDeserializer()
	return staticObjectCreator{
		// 获取Kubernetes对象的类型
		objType: objectKind,
		// 使用通用反序列化器
		deserializer: deserializer,
	}
}

// NewK8sObj 根据类型创建一个新的Kubernetes对象
func NewK8sObj(t reflect.Type) metav1.Object {
	newObj := reflect.New(t).Interface()
	return newObj.(metav1.Object)
}

// NewObject 从原始JSON数据创建指定类型的对象
func (s staticObjectCreator) NewObject(rawJSON []byte) (runtime.Object, error) {
	if err := validateInput(rawJSON); err != nil {
		return nil, err
	}

	// 创建指定类型的新对象
	runtimeObj, err := s.createRuntimeObject()
	if err != nil {
		return nil, err
	}

	// 反序列化JSON数据
	if err := s.deserialize(rawJSON, runtimeObj); err != nil {
		return nil, fmt.Errorf("%w: %v", errDeserialization, err)
	}

	return runtimeObj, nil
}

// extractK8sObjectType 返回Kubernetes对象的类型（非指针类型）
func extractK8sObjectType(obj metav1.Object) reflect.Type {
	// 获取对象的间接类型（即解引用后的类型）
	return reflect.Indirect(reflect.ValueOf(obj)).Type()
}

// createRuntimeObject 创建指定类型的runtime.Object
func (s staticObjectCreator) createRuntimeObject() (runtime.Object, error) {
	obj := NewK8sObj(s.objType)
	runtimeObj, ok := obj.(runtime.Object)
	if !ok {
		return nil, errTypeAssertion
	}
	return runtimeObj, nil
}

// deserialize 反序列化JSON数据到对象中
func (s staticObjectCreator) deserialize(rawJSON []byte, obj runtime.Object) error {
	_, _, err := s.deserializer.Decode(rawJSON, nil, obj)
	return err
}

// dynamicObjectCreator 结构体用于动态创建对象
type dynamicObjectCreator struct {
	// 通用解码器
	universalDecoder runtime.Decoder
	// 非结构化数据解码器
	unstructuredDecoder runtime.Decoder
}

// NewObject 从原始JSON数据动态创建对象
func (d dynamicObjectCreator) NewObject(rawJSON []byte) (runtime.Object, error) {
	if err := validateInput(rawJSON); err != nil {
		return nil, err
	}

	// 首先尝试通用解码器
	runtimeObj, _, err := d.universalDecoder.Decode(rawJSON, nil, nil)
	if err == nil {
		return runtimeObj, nil
	}

	// 回退到非结构化解码器
	runtimeObj, _, err = d.unstructuredDecoder.Decode(rawJSON, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errDeserialization, err)
	}

	return runtimeObj, nil
}

// NewDynamicObjectCreator 创建动态对象创建器
func NewDynamicObjectCreator() ObjectCreator {
	return dynamicObjectCreator{
		// 使用全局方案的通用反序列化器
		universalDecoder: scheme.Codecs.UniversalDeserializer(),
		// 使用非结构化JSON方案
		unstructuredDecoder: unstructured.UnstructuredJSONScheme,
	}
}

// validateInput 验证输入数据
func validateInput(rawJSON []byte) error {
	if len(rawJSON) == 0 {
		return errEmptyInput
	}

	if strings.TrimSpace(string(rawJSON)) == "" {
		return errWhitespaceInput
	}

	return nil
}

// ToAdmissionErrorResponse 将错误转换为准入响应
func ToAdmissionErrorResponse(uid types.UID, err error) *admissionv1.AdmissionResponse {
	klog.Errorf("准入webhook错误: %v", err)
	rs := &metav1.Status{
		Message: err.Error(),
		Status:  metav1.StatusFailure,
	}
	ar := &admissionv1.AdmissionResponse{
		UID:    uid,
		Result: rs,
	}
	return ar
}
