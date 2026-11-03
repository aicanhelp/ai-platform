/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package webhook

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/utils"
)

// Endpoint 绑定的endpoint
var Endpoint string

// WaitForSvcRunning  等待服务绑定endpoint
func WaitForSvcRunning(interval, timeout time.Duration, svcNamespace, svcName string) error {
	var err error

	// create apiserver client
	client, err := utils.InitKubeClient()
	if err != nil {
		return fmt.Errorf("failed to init kubenetes client: %v", err)
	}
	// 不会产生同时写的冲突 如果已经有ep 则跳过wait.pollimmediate方法
	if Endpoint != "" {
		// 已经绑定成功
		return nil
	}
	// 尝试条件函数，直到返回true error timeout
	err = wait.PollImmediate(interval, timeout, func() (bool, error) {
		// check if service endpoint exists
		get, err := client.CoreV1().Endpoints(svcNamespace).Get(context.TODO(), svcName, v1.GetOptions{})
		if err != nil {
			if errors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		} else {
			if len(get.Subsets) != 0 {
				// 已经及时绑定，跳出方法
				if len(get.Subsets[0].Addresses) == 0 {
					klog.Errorf("endpoint no bind wait check")
					// 此时ip在notReadyAddresses字段中 等待下一次轮训
					return false, nil
				}
				Endpoint = get.Subsets[0].Addresses[0].IP
				return true, nil
			} else {
				klog.Errorf("endpoint no bind wait check")
				// 继续尝试
				return false, nil
			}
		}
	})
	if err != nil {
		klog.Errorf("wait for svc binding endpoint err : %v", err)
		return err
	}
	return nil
}
