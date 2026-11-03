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

// Package rpc define rpc server
package rpc

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chart/loader"

	pb "application-management-service/pkg/rpc/helmchart"
	"application-management-service/pkg/utils/httputil"
	"application-management-service/pkg/zlog"
)

const (
	contextTimeOutSeconds = 30
)

// GetHelmChartBytes rpc client side function, get helm chart from marketplace service
func GetHelmChartBytes(repoName, chartName, chartVersion string) (io.Reader, error) {
	conn, err := getGrpcConnection()
	if err != nil {
		zlog.Errorf("failed to connect grpc: %w", err)
		return nil, fmt.Errorf("failed to connect grpc: %w", err)
	}
	defer func() {
		if conn != nil {
			if cerr := conn.Close(); cerr != nil {
				zlog.Errorf("grpc connection close error: %v", cerr)
			}
		}
	}()

	client := pb.NewChartManagerClient(conn)

	req := &pb.ChartRequest{
		TargetSystem: 1,
		RepoName:     repoName,
		ChartName:    chartName,
		ChartVersion: chartVersion,
	}
	ctx, cancel := context.WithTimeout(context.Background(), contextTimeOutSeconds*time.Second)
	defer cancel()
	stream, err := client.GetHelmChart(ctx, req)
	if err != nil {
		zlog.Errorf("could not get chart: %v", err)
		return nil, fmt.Errorf("failed to connect grpc: %w", err)
	}

	var buffer bytes.Buffer

	for {
		res, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			grpcErr, ok := status.FromError(err)
			if ok {
				zlog.Errorf("gRPC error %s", grpcErr.Code())
			}
			return nil, err
		}
		if !res.Success {
			zlog.Errorf("Failed to get chart")
			return nil, fmt.Errorf("failed to get chart")
		}
		buffer.Write(res.ChartBytes)
	}
	return &buffer, nil
}

func getGrpcConnection() (*grpc.ClientConn, error) {
	var enableTLS bool
	enableTLS, err := httputil.IsHttpsEnabled()
	if err != nil {
		enableTLS = false
	}
	tlsCfg, err := httputil.GetHttpConfig(enableTLS)
	if err != nil {
		zlog.Errorf("get tls transport failed: %v")
		enableTLS = false
	}

	var conn *grpc.ClientConn
	if !enableTLS {
		conn, err = grpc.NewClient("marketplace-service.openfuyao-system.svc.cluster.local:9038",
			grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		conn, err = grpc.NewClient("marketplace-service.openfuyao-system.svc.cluster.local:9038",
			grpc.WithTransportCredentials(credentials.NewTLS(tlsCfg)))
	}

	if err != nil {
		zlog.Errorf("did not connect: %v", err)
	}

	return conn, err
}

// GetHelmChart rpc client side function, get helm chart from marketplace service
func GetHelmChart(repoName, chartName, chartVersion string) (*chart.Chart, error) {
	chartBytes, err := GetHelmChartBytes(repoName, chartName, chartVersion)
	if err != nil {
		zlog.Errorf("Error geting helm chart bytes :%v", err)
		return nil, err
	}
	chartStruct, err := loader.LoadArchive(chartBytes)
	if err != nil {
		fmt.Println("Error loading chartStruct archive:", err)
		return nil, err
	}
	return chartStruct, nil
}
