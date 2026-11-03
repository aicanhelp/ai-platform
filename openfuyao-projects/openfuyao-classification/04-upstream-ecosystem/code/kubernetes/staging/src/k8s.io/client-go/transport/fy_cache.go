/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package transport

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
)

type fyTlsTransportCache struct {
	mu         sync.Mutex
	transports map[fyTlsCacheKey]*http.Transport
}

var fyTlsCache = &fyTlsTransportCache{transports: make(map[fyTlsCacheKey]*http.Transport)}

const enableReusingFlag = "FY_ENABLE_TRANSPORT_REUSING"

type fyTlsCacheKey struct {
	tlsCacheKey
	getCertPointer string
	proxyPointer   string
	dialPointer    string
}

// FY: declare as a function to make it test easier
var enableFYTransportReusing = func() bool {
	return os.Getenv(enableReusingFlag) == "true"
}

func (t fyTlsCacheKey) String() string {
	ketText := "<none>"
	if len(t.keyData) > 0 {
		ketText = "<redacted>"
	}

	return fmt.Sprintf("caData:%#v, disableCompression:%t, certData:%#v, insecure:%v, "+
		"getCert:%s, dial:%s, proxy:%s, keyData:%s, serverName:%s,",
		t.tlsCacheKey.caData, t.tlsCacheKey.disableCompression, t.tlsCacheKey.certData, t.tlsCacheKey.insecure,
		t.getCertPointer, t.dialPointer, t.proxyPointer, ketText, t.tlsCacheKey.serverName)
}

// getFyTlsKey returns tls.Config key
func getFyTlsKey(c *Config) (fyTlsCacheKey, bool, error) {
	var k fyTlsCacheKey
	cacheKey := tlsCacheKey{
		caData:             string(c.TLS.CAData),
		disableCompression: c.DisableCompression,
		insecure:           c.TLS.Insecure,
		nextProtos:         strings.Join(c.TLS.NextProtos, ","),
		serverName:         c.TLS.ServerName,
	}
	// if it needs reload ca from file, ca key filed should be file path
	tweakCacheKeyIfNeedReloadCAFile := func(key *tlsCacheKey) {
		if c.TLS.FYReloadCAFile {
			key.fyCaFile = c.TLS.CAFile
			key.caData = ""
		}
	}

	if !enableFYTransportReusing() {
		tlsKey, canCache, err := tlsConfigKey(c)
		tweakCacheKeyIfNeedReloadCAFile(&tlsKey)
		k.tlsCacheKey = tlsKey
		return k, canCache, err
	}

	if err := loadTLSFiles(c); err != nil {
		return fyTlsCacheKey{}, true, err
	}
	k = fyTlsCacheKey{
		tlsCacheKey:    cacheKey,
		getCertPointer: fmt.Sprintf("%p", c.TLS.GetCertHolder),
		dialPointer:    fmt.Sprintf("%p", c.DialHolder),
		proxyPointer:   fmt.Sprintf("%p", c.Proxy),
	}

	if !c.TLS.ReloadTLSFiles {
		k.certData = string(c.TLS.CertData)
		k.keyData = string(c.TLS.KeyData)
	} else {
		k.certFile = c.TLS.CertFile
		k.keyFile = c.TLS.KeyFile
	}

	tweakCacheKeyIfNeedReloadCAFile(&k.tlsCacheKey)

	return k, true, nil
}
