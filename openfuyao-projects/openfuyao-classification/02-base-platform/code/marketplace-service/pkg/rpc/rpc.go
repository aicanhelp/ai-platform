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

package rpc

import (
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	helmv1 "marketplace-service/pkg/api/marketplace/v1beta1"
	pb "marketplace-service/pkg/rpc/helmchart"
	"marketplace-service/pkg/zlog"
)

// Server rpc server struct
type Server struct {
	pb.UnimplementedChartManagerServer
	Handler *helmv1.Handler
}

// GetHelmChart rpc server side function, return helm chart from registry
func (s *Server) GetHelmChart(req *pb.ChartRequest, stream pb.ChartManager_GetHelmChartServer) error {
	chartBytes, err := s.Handler.HelmHandler.GetChartBytesByVersion(req.GetRepoName(), req.GetChartName(),
		req.GetChartVersion())
	if err != nil {
		zlog.Errorf("Failed to get chart bytes: %v", err)
		return status.Error(codes.Internal, "internal server error")
	}
	chunkSize := 16384
	for chartBytes.Len() > 0 {
		chunk := make([]byte, chunkSize)
		n, _ := chartBytes.Read(chunk)
		if err := stream.Send(&pb.ChartResponse{
			Success:    true,
			ChartBytes: chunk[:n],
		}); err != nil {
			return err
		}
	}

	return nil
}
