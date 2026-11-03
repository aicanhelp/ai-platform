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
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"time"

	"google.golang.org/grpc"

	"huawei.com/vxpu-device-plugin/pkg/api/runtime/service"
	"huawei.com/vxpu-device-plugin/pkg/log"
)

const (
	pidsSockPath = "/var/lib/xpu/pids.sock"
	dialTimeout  = 5
)

var dialGRPCFunc = func() (*grpc.ClientConn, error) {
	return grpc.Dial(pidsSockPath, grpc.WithInsecure(),
		grpc.WithBlock(), grpc.WithTimeout(dialTimeout*time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)
}

var getPidsFunc = func(client service.PidsServiceClient, cgroupPath string) error {
	_, err := client.GetPids(context.Background(), &service.GetPidsRequest{CgroupPath: cgroupPath})
	return err
}

func updatePidsConfig(cgroupPath string) error {
	conn, err := dialGRPCFunc()
	if err != nil {
		log.Errorf("grpc dial error: %v", err)
		return err
	}
	if conn == nil {
		return fmt.Errorf("client connection is nil")
	}
	defer func() {
		if conn != nil {
			conn.Close()
		}
	}()

	client := service.NewPidsServiceClient(conn)
	err = getPidsFunc(client, cgroupPath)
	if err != nil {
		log.Errorf("client GetPids error: %v", err)
		return err
	}

	return nil
}

func main() {
	var cgroupPath string
	flag.StringVar(&cgroupPath, "cgroup-path", "", "cgroup path")
	flag.Parse()

	err := updatePidsConfig(cgroupPath)
	if err != nil {
		log.Errorf("get pids failed, cgroupPath: %s", cgroupPath)
		os.Exit(1)
	}
}
