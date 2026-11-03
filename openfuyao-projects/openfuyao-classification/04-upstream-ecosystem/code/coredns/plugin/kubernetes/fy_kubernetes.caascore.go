//go:build caascore
// +build caascore

// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package kubernetes plugin.
package kubernetes

import "k8s.io/client-go/tools/clientcmd/api"

func init() {
	fyAdjustAuthInfo = func(k *Kubernetes, authInfo *api.AuthInfo) error {
		if len(k.FyCRLFile) > 0 {
			authInfo.CRLFile = k.FyCRLFile
		}
		return nil
	}
}
