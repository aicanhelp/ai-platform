/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package pkiutil

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestConvertSecretCertToBKECert tests the convertSecretCertToBKECert function
// to ensure it correctly matches secret names to BKECert certificates.
func TestConvertSecretCertToBKECert(t *testing.T) {
	// Define test cases with secret name and expected certificate name
	testCases := []struct {
		name             string
		secretName       string
		expectedCertName string
		shouldMatch      bool
	}{
		{"match ca certificate", "cluster-ca", "ca", true},
		{"match apiserver certificate", "apiserver", "apiserver", true},
		{"match admin certificate (renamed from kubeconfig)", "cluster-admin", "admin", true},
		{"match kubelet certificate", "cluster-kubelet", "kubelet", true},
		{"match etcd certificate", "cluster-etcd", "etcd", true},
		{"match etcd-server certificate", "cluster-etcd-server", "etcd-server", true},
		{"no match returns nil", "unknown-certificate", "", false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			secret := &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: tc.secretName},
			}
			result := convertSecretCertToBKECert(secret)
			if tc.shouldMatch {
				if result == nil || result.Name != tc.expectedCertName {
					t.Errorf("Expected to match '%s' certificate, got %v", tc.expectedCertName, result)
				}
			} else {
				if result != nil {
					t.Errorf("Expected nil for non-matching secret, got %v", result)
				}
			}
		})
	}
}
