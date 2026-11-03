// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package api

// Service allows decrypting data using an external Decryption Service.
type Service interface {
	// DecryptCert a given bytearray to obtain the original data as bytes.
	DecryptCert(data []byte) ([]byte, error)

	// DecryptSecret a given bytearray to obtain the original data as bytes.
	DecryptSecret(data []byte) ([]byte, error)

	// Release service resource.
	Release()
}
