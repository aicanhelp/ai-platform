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

package utils

import (
	"encoding/base64"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

// UID 格式：uid/namespace/name
type UNN string

// Validate 验证UID是否有效
func (uid UNN) Validate() bool {
	parts := strings.Split(string(uid), "/")
	partsLen := 3
	if len(parts) != partsLen {
		return false
	}
	// 检查每个部分都不能为空
	for _, part := range parts {
		if part == "" {
			return false
		}
	}
	return true
}

// GetUid 获取UID
func (uid UNN) Uid() string {
	return strings.Split(string(uid), "/")[0]
}

// GetNameSpace 获取NameSpace
func (uid UNN) Namespace() string {
	return strings.Split(string(uid), "/")[1]
}

// GetName 获取Name
func (uid UNN) Name() string {
	return strings.Split(string(uid), "/")[2]
}

// GetAll 获取(UID,NameSpace,Name)
func (uid UNN) All() []string {
	return strings.Split(string(uid), "/")
}

// GetFormat 获取UNN的格式
func (uid UNN) Format() string {
	return "<uid>/<namespace>/<name>"
}

func (uid UNN) String() string {
	return string(uid)
}

func (uid UNN) Encode() string {
	return base64.StdEncoding.EncodeToString([]byte(uid.String()))
}

func NewUNN(uid, namespace, name string) UNN {
	return UNN(uid + "/" + namespace + "/" + name)
}

func NewUNNFromId(id string) UNN {
	data, err := base64.StdEncoding.DecodeString(id)
	if err != nil {
		klog.Error("failed to decode UNN from id:", err)
	}
	unn := UNN(data)
	return unn
}

func NewUNNFromObject(obj metav1.Object) UNN {
	uid := obj.GetUID()
	namespace := obj.GetNamespace()
	name := obj.GetName()
	return NewUNN(string(uid), namespace, name)
}
