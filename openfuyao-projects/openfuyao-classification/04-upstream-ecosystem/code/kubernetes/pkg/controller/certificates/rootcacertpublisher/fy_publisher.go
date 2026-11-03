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

// Package rootcacertpublisher manages certificate ConfigMap which stores certificates objects inside Namespaces
package rootcacertpublisher

import (
	"k8s.io/apimachinery/pkg/labels"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/klog/v2"
)

func (c *Publisher) SetConfigmapAndCA(currentCA []byte, nsInformer coreinformers.NamespaceInformer) {
	c.rootCA = currentCA
	c.updateConfigmap(nsInformer)
}

func (c *Publisher) updateConfigmap(nsInformer coreinformers.NamespaceInformer) {
	nsList, err := nsInformer.Lister().List(labels.Everything())
	if err != nil {
		klog.Errorf("[onchange] failed to list namespcae to update configmap root-ca-file: %v", err)
		return
	}
	for _, ns := range nsList {
		c.queue.Add(ns.Name)
	}
}
