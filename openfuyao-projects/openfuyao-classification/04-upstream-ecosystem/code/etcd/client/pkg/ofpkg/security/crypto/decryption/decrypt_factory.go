package decryption

import (
	"go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/grpc"
)

func init() {
	decryptKey = OfTryDecryptCert
	decryptServiceFactory = grpc.CreateGRPCService
}
