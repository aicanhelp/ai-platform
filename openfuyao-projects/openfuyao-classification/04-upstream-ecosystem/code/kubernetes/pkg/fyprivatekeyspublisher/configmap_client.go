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

// Package fyprivatekeyspublisher publish the hash of key file which signed service account.
package fyprivatekeyspublisher

import (
	"context"
	"net/http"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/klog/v2"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
)

const (
	resourceType = "configmaps"
)

type configmapClient interface {
	getConfigMap() (*v1.ConfigMap, bool)
	createConfigMap(configmapDate *v1.ConfigMap) error
	updateConfigMap(configmapDate *v1.ConfigMap) error
}

type realConfigmapClient struct {
	client corev1.CoreV1Interface
}

func (c realConfigmapClient) getConfigMap() (*v1.ConfigMap, bool) {
	var code int
	date := &v1.ConfigMap{}
	ctx := context.TODO()
	res := c.client.RESTClient().Get().
		Resource(resourceType).
		Namespace(kubeapi.NamespaceSystem).
		Name(TokenPrivateKeysConfigmap).
		Do(ctx)
	res.StatusCode(&code)
	if code == http.StatusNotFound {
		return nil, false
	}

	err := res.Into(date)
	if err != nil {
		klog.ErrorS(err, "fy get published data error!")
		return nil, false
	}

	return date, true
}

func (c realConfigmapClient) createConfigMap(configmapDate *v1.ConfigMap) error {
	ctx := context.TODO()
	_, err := c.client.ConfigMaps(kubeapi.NamespaceSystem).Create(ctx, configmapDate, metav1.CreateOptions{})
	return err
}

func (c realConfigmapClient) updateConfigMap(configmapDate *v1.ConfigMap) error {
	ctx := context.TODO()
	_, err := c.client.ConfigMaps(kubeapi.NamespaceSystem).Update(ctx, configmapDate, metav1.UpdateOptions{})
	return err
}
