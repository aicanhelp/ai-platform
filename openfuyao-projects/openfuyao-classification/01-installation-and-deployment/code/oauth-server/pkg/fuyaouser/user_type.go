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
	"k8s.io/apimachinery/pkg/apis/meta/v1"
)

// UserSpec represents the desired configuration for system users
type UserSpec struct {
	Username              string    `json:"Username,omitempty"`
	EncryptedPassword     []byte    `json:"EncryptedPassword,omitempty"`
	Description           string    `json:"Description,omitempty"`
	InvitedByClustersList []string  `json:"InvitedByClustersList,omitempty"`
	PlatformRole          string    `json:"PlatformRole,omitempty"`
	FailedLoginRecords    []v1.Time `json:"failedLoginRecords,omitempty"`
	FirstLogin            bool      `json:"FirstLogin,omitempty"`
}

// UserStatus reflects current operational state
type UserStatus struct {
	LockStatus      string   `json:"lockStatus,omitempty"`
	LockedTimestamp *v1.Time `json:"lockedTimestamp,omitempty"`
	RemainAttempts  int      `json:"RemainAttempts,omitempty"`
}

// User represents the core user resource schema
type User struct {
	v1.TypeMeta   `json:",inline"`
	v1.ObjectMeta `json:"metadata,omitempty"`

	Spec   UserSpec   `json:"spec,omitempty"`
	Status UserStatus `json:"status,omitempty"`
}

// UserList contains collections of User resources
type UserList struct {
	v1.TypeMeta `json:",inline"`
	v1.ListMeta `json:"metadata,omitempty"`
	Items       []User `json:"items"`
}
