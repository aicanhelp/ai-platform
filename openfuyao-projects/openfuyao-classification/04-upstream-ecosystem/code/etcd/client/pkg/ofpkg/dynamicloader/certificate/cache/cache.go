// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package cache provide common struct, interface, func for certificates
package cache

// CertificateCacheKey 证书缓存键，用于唯一标识一组证书文件
type CertificateCacheKey struct {
	certFile    string
	keyFile     string
	trustCAFile string
}

// NewCacheKey 创建新的证书缓存键
func NewCacheKey(cert, key, ca string) CertificateCacheKey {
	return CertificateCacheKey{
		certFile:    cert,
		keyFile:     key,
		trustCAFile: ca,
	}
}
