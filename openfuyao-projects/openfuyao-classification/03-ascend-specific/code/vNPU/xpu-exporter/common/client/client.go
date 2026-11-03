/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package client

import (
	"context"
	"net"
	"time"

	"google.golang.org/grpc"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/common/service"
)

const (
	pidsSockPath = "/var/lib/xpu/pids.sock"
	dialTimeout  = 5
	megabyte     = 1024 * 1024
)

func GetAllVxpuInfo() (string, error) {
	conn, err := grpc.Dial(pidsSockPath, grpc.WithInsecure(),
		grpc.WithBlock(), grpc.WithTimeout(dialTimeout*time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)
	if err != nil {
		log.Errorf("grpc dial error: %v", err)
		return "", err
	}
	defer conn.Close()

	client := service.NewPidsServiceClient(conn)
	getAllVgpuInfoResponse, err := client.GetAllVxpuInfo(context.Background(),
		&service.GetAllVxpuInfoRequest{Period: "60"})
	if err != nil {
		log.Errorf("client GetAllVgpuInfo error: %v", err)
		return "", err
	}
	return getAllVgpuInfoResponse.VxpuInfos, nil
}
