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

// Package webhook defined mutate and validate rules
package webhook

import (
	"context"
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"

	"volcano-config-service/pkg/utils/k8sutil"
)

// Endpoint indicate api point
var Endpoint string

// IsNotFound return is not found error
func IsNotFound(err error) bool {
	if err == nil {
		return false
	}

	return errors.IsNotFound(err)
}

// WaitForSvcRunning indicate service running
func WaitForSvcRunning(interval, timeout time.Duration, svcNamespace, svcName string) error {
	var err error

	// create apiserver client
	client, err := k8sutil.InitClient()
	if err != nil {
		return fmt.Errorf("failed to init kubenetes client: %v", err)
	}
	// 不会产生同时写的冲突 如果已经有ep 则跳过wait.pollimmediate方法
	if Endpoint != "" {
		klog.Infof("Endpoint already set: %s", Endpoint)
		// 已经绑定成功
		return nil
	}

	// 使用新的 wait API
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// 尝试条件函数，直到返回true error timeout
	err = wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		// check if service endpoint exists
		get, err := client.CoreV1().Endpoints(svcNamespace).Get(context.TODO(), svcName, metav1.GetOptions{})
		if err != nil {
			if IsNotFound(err) {
				return false, nil
			}
			return false, err
		} else {
			if len(get.Subsets) != 0 {
				// 已经及时绑定，跳出方法
				if len(get.Subsets[0].Addresses) == 0 {
					glog.Errorf("endpoint no bind wait check")
					// 此时ip在notReadyAddresses字段中 等待下一次轮询
					return false, nil
				}
				Endpoint = get.Subsets[0].Addresses[0].IP
				return true, nil
			} else {
				glog.Errorf("endpoint no bind wait check")
				// 继续尝试
				return false, nil
			}
		}
	})
	if err != nil {
		glog.Errorf("wait for svc binding endpoint err : %v", err)
		return err
	}
	return nil
}
