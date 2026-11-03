package decryption

import (
	"crypto/sha512"
	"crypto/tls"
	"fmt"
	"sync"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/util/log"
)

var (
	isEnable bool

	cache *sync.Map

	infoOnce *sync.Once

	// for test decrypt api
	decryptKey = disableDecrypt
)

func init() {
	runInit()
}

func runInit() {
	isEnable = grpcEnable()
	if isEnable {
		cache = &sync.Map{}
	} else {
		infoOnce = &sync.Once{}
	}
}

// OFTryDecryptWithCache 解密私钥，并生成 tls.Certificate，提供缓存功能
func OFTryDecryptWithCache(certPEMBlock, keyPEMBlock []byte,
	parseFunc func([]byte, []byte) (tls.Certificate, error)) (*tls.Certificate, error, bool) {
	if !isEnable {
		infoOnce.Do(func() {
			log.GetLogger().Info("cert decrypt is not enabled")
		})
		return nil, nil, false
	}
	key := toKey(certPEMBlock, keyPEMBlock)
	value, ok := cache.Load(key)
	if ok {
		result, ok := value.(tls.Certificate)
		if !ok {
			return &tls.Certificate{}, fmt.Errorf("type assertion failed"), false
		}
		return &result, nil, true
	}
	decryptedKey, err := decryptKey(keyPEMBlock)
	if err != nil {
		return nil, err, true
	}
	keyPair, err := parseFunc(certPEMBlock, decryptedKey)
	if err == nil {
		cache.Store(key, keyPair) // 仅缓存值，不缓存指针
		return &keyPair, nil, true
	}
	return nil, err, true
}

// 强制使用 byte数组 作为key，避免额外的 string 转换
func toKey(certPEMBlock, keyPEMBlock []byte) (sum384 [sha512.Size384]byte) {
	hash := sha512.New384()
	hash.Write(certPEMBlock)
	hash.Write(keyPEMBlock)
	sum := hash.Sum(nil)

	copy(sum384[:], sum[0:sha512.Size384])
	return
}

func disableDecrypt(keyBlock []byte) ([]byte, error) {
	return keyBlock, nil
}
