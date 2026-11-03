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
	"crypto/sha256"
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

const (
	TokenPrivateKeysConfigmap    = "apiserver-private-keys-hash"
	TokenPrivateKeysConfigmapKey = "hash"
	publishInterval              = 10 * time.Second
)

var publisher fyPrivateKeyPublisher

type fyPrivateKeyPublisher struct {
	sync.Mutex

	client   configmapClient
	working  bool
	stopWork context.CancelFunc
	date     string
}

func Install(c *rest.Config) {
	kubeClient, err := clientset.NewForConfig(c)
	if err != nil {
		klog.ErrorS(err, "fy-private-keys-publisher install error!")
		return
	}

	publisher.client = &realConfigmapClient{client: kubeClient.CoreV1()}
}

func hash(data []byte) string {
	bytes := sha256.Sum256(data)
	return fmt.Sprintf("%x", bytes)
}

func Publish(privateKeys []byte) {
	newData := hash(privateKeys)

	if publisher.date == newData {
		return
	}

	publisher.date = newData
	go publisher.publish()
}

func (f *fyPrivateKeyPublisher) publish() {
	if f.working && f.stopWork != nil {
		klog.InfoS("fy stop publish work.")
		f.stopWork()
	}
	f.working = true

	ctx, cancel := context.WithCancel(context.TODO())
	f.stopWork = cancel
	wait.PollUntilContextCancel(
		ctx,
		publishInterval,
		true,
		func(ctx context.Context) (done bool, err error) {
			if f.client == nil {
				return false, nil
			}

			f.Lock()
			defer f.Unlock()

			_, isResourceExist := f.client.getConfigMap()

			newConfigMap := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: TokenPrivateKeysConfigmap,
				},
				Data: map[string]string{},
			}
			newConfigMap.Data[TokenPrivateKeysConfigmapKey] = f.date

			if isResourceExist {
				err := f.client.updateConfigMap(newConfigMap)
				if err != nil {
					klog.ErrorS(err, "fy update published data error!")
					return false, nil
				}
			} else {
				err := f.client.createConfigMap(newConfigMap)
				if err != nil {
					klog.ErrorS(err, "fy create published data error!")
					return false, nil
				}
			}

			f.working = false
			f.stopWork = nil
			return true, nil
		})
}
