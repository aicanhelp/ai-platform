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

// Package utils contains a collection of utility functions and helpers that provide
// common functionalities useful across the entire engineering project.
package utils

import "time"

const (
	// DefaultClusterRolePrefix is the prefix used for default roles applied to new cluster.
	DefaultClusterRolePrefix = "karmada-controller-manager"

	// DefaultSAPrefix is the default service account prefix used by karmada.
	DefaultSAPrefix = "karmada"

	// DefaultUserClusterRolePrefix is the clusterrole prefix which need to registered to karmada control plane.
	DefaultUserClusterRolePrefix = "openfuyao:multicluster:cluster:"

	// DefaultUserPlatformRolePrefix is the platformrole prefix which need to registered to karmada control plane.
	DefaultUserPlatformRolePrefix = "openfuyao:multicluster:platform:"

	// DefaultImpersonatorPrefix is the prefix used for impersonation accounts in karmada.
	DefaultImpersonatorPrefix = "impersonator"

	// DefaultKarmadaNamespace is the default namespace used by karmada system.
	DefaultKarmadaNamespace = "karmada-cluster"

	// DefaultKarmdaClusterAPIGroup is the karmada cluster resource APIgroup.
	DefaultKarmdaClusterAPIGroup = "cluster.karmada.io"

	// KarmadaSystemLabel is the label key applied to all system componets of karmada.
	KarmadaSystemLabel = "karmada.io/system"

	// CreatedByopenFuyaoLabel is the label kay indicating resources created by the openfuyao system.
	CreatedByopenFuyaoLabel = "openfuyao.com/created"

	// DefaultopenFuyaoLabel is the label key used for tagging resource managed by openFuyao.
	DefaultopenFuyaoLabel = "openfuyao.com/tagged"

	// DefaultopenFuyaoUserRefLabel is the label key used for targeting to the specifie user.
	DefaultopenFuyaoUserRefLabel = "openfuyao.com/user-ref"

	// DefaultopenFuyaoRoleRefLabel is the label key used for targeting to the specifie role.
	DefaultopenFuyaoRoleRefLabel = "openfuyao.com/role-ref"

	// DefaultopenFuyaoRoleType is the label key used for targeting to the specifie role type.
	DefaultopenFuyaoRoleType = "role-type"

	// KaramadaSystemLabelValue set default `true`.
	KaramadaSystemLabelValue = "true"
	// CreatedByopenFuyaoLabelValue set default `true`.
	CreatedByopenFuyaoLabelValue = "true"

	// DefaultPlatformRoleLabel set labels value `platform-admin-Role`.
	DefaultPlatformRoleLabel = "platform-admin-Role"
	// DefaultMemCusterRoleLabel set labels value `mem-cluster-Role`.
	DefaultMemCusterRoleLabel = "mem-cluster-Role"

	// TLSCAData defines tls ca prefix
	TLSCAData = "caBundle"
	// TLSTokenData defines tls token prefix
	TLSTokenData = "token"

	// KarmadaConfigPath is the path of karmada-apiserver kubeconfig.
	KarmadaConfigPath = "/etc/karmada/kubeconfig"

	// KarmadaEndPoints is the karmada APIserver endpoints.
	KarmadaEndPoints = "https://karmada-apiserver.karmada-system.svc.cluster.local:5443"
	// KarmadaAggreEndPoints is the karmada-aggregated-apiserver endpoints.
	KarmadaAggreEndPoints = "https://karmada-aggregated-apiserver.karmada-system.svc.cluster.local:443"

	// KarmadaClusterAPIPath is the API path for accessing cluster-specific resources in Karmada.
	KarmadaClusterAPIPath = "/apis/cluster.karmada.io/v1alpha1/clusters"
)

const (
	// WatchInterval is the interval at which are polled for changes.
	WatchInterval = 1 * time.Second
	// WatchTimeout is the timeout for watching changes.
	WatchTimeout = 10 * time.Second

	// WatchClusterResourceInterval is the interval for waitUntilTimeout
	WatchClusterResourceInterval = 10 * time.Millisecond

	// WatchClusterInterval set for karmada cluster object.
	WatchClusterInterval = 1 * time.Second
	// WatchClusterTimeout set for karmada cluster object.
	WatchClusterTimeout = 30 * time.Second

	// WatchKarmadaListInterval set for karmada list operation .
	WatchKarmadaListInterval = 2 * time.Second
	// WatchKarmadaListTimeout set for karmada list operation.
	WatchKarmadaListTimeout = 600 * time.Second

	// MaxLabelLength is the max length.
	MaxLabelLength = 64
	// MaxNameLength is the max length.
	MaxNameLength = 256

	// MegabytesPerGigabyte defines the number of megabytes in one gigabyte,
	// used for converting memory uints from MiB to GiB.
	MegabytesPerGigabyte = 1024

	// MilliCoresPerCore defines the number of milliseconds in one second.
	// It is used for converting time units from seconds to miliseconds.
	MilliCoresPerCore = 1000
)

const (
	// MetadataLabelPattern defines a regular expression used to validate metadata labels strings.
	MetadataLabelPattern = `([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]`
	// MetadataNamePattern defines a regular expression used to validate metadata name strings.
	MetadataNamePattern = `^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`
)
