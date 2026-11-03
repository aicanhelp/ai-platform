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

// Package fuyaouser defines the user crd and functions for openfuyao platform
package fuyaouser

import (
	"context"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"

	"openfuyao/oauth-server/pkg/zlog"
)

var (
	userMgmtGVR = schema.GroupVersionResource{
		Group:    "users.openfuyao.com",
		Version:  "v1alpha1",
		Resource: "users",
	}
)

// GetUserInfo returns the User with given plugin name
func GetUserInfo(c dynamic.Interface, name string) (*User, error) {
	dr, err := c.Resource(userMgmtGVR).Get(context.Background(), name, v1.GetOptions{})
	if err != nil {
		return nil, err
	}
	var userCR User
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(dr.Object, &userCR)
	if err != nil {
		zlog.LogErrorf("Error converting to %s: %s", userMgmtGVR.Resource, name)
		return nil, err
	}
	return &userCR, nil
}

// PatchUserInfo updates the User with given patch data
func PatchUserInfo(c dynamic.Interface, name string, data []byte) error {
	_, err := c.Resource(userMgmtGVR).
		Patch(context.Background(), name, "application/merge-patch+json", data, v1.PatchOptions{}, "")
	return err
}

// UpdateUserInfo updates the User with given usercr
func UpdateUserInfo(c dynamic.Interface, userCR *User) error {
	obj, err := StructToUnstructured(userCR)
	if err != nil {
		zlog.LogError("cannot convert to unstructured object")
		return err
	}
	_, err = c.Resource(userMgmtGVR).Update(context.Background(), obj, v1.UpdateOptions{})
	if err != nil {
		zlog.LogErrorf("update user %s failed", userCR.Name)
	}
	return err
}

// StructToUnstructured convert from any struct to Unstructured struct
// mainly for kubernetes usage
func StructToUnstructured(v interface{}) (*unstructured.Unstructured, error) {
	UnstructuredResult, err := runtime.DefaultUnstructuredConverter.ToUnstructured(v)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{
		Object: UnstructuredResult,
	}, nil
}
