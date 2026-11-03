/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package containerd

import (
	"context"
	"errors"

	"github.com/containerd/containerd/v2/client"
	"github.com/containerd/containerd/v2/core/images"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

type ContainerdClient interface {
	GetClient() *client.Client
}

type ImageRef struct {
	Image    string `json:"image"`
	Username string `json:"username"`
	Password string `json:"password"`
}

type Client struct {
	imageClient images.Store
	condClient  *client.Client
	ctx         context.Context
	cancel      context.CancelFunc
}

var (
	containerdSock      = "unix:///var/run/containerd/containerd.sock"
	containerdNamespace = "k8s.io"
	containerdSockLinux = "/var/run/containerd/containerd.sock"
)

func NewContainedClient() (ContainerdClient, error) {
	if !utils.Exists(containerdSockLinux) {
		return nil, errors.New("containerd service does not exist. ")
	}

	condClient, err := client.New(
		containerdSockLinux,
		client.WithDefaultNamespace(containerdNamespace), // 关键变更
	)

	if err != nil {
		return nil, err
	}

	// 获取 ImageService（接口位置变化）
	imageClient := condClient.ImageService()

	ctx, cancel := context.WithCancel(context.Background())
	return &Client{
		imageClient: imageClient,
		condClient:  condClient,
		ctx:         ctx,
		cancel:      cancel,
	}, nil
}

func (c *Client) GetClient() *client.Client {
	return c.condClient
}
