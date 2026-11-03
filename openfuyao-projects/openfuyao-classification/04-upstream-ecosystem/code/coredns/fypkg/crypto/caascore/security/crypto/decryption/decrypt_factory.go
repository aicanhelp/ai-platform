//go:build !DisableGrpcDecryption && !windows
// +build !DisableGrpcDecryption,!windows

// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package decryption

import (
	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/grpc"
)

func init() {
	decryptServiceFactory = grpc.Holder.GetGRPCService
}
