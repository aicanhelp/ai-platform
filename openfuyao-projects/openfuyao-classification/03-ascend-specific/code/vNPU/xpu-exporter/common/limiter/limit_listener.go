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
package limiter

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/common/cache"
)

const (
	maxConnection   = 1024
	maxIPConnection = 512
)

func LimitListener(l net.Listener, totalConnLimit, IPConnLimit, cacheSize int) (net.Listener, error) {
	if totalConnLimit < 0 || totalConnLimit > maxConnection {
		return nil, errors.New("the parameter totalConnLimit is illegal")
	}
	if IPConnLimit < 0 || IPConnLimit > maxIPConnection {
		return nil, errors.New("the parameter IPConnLimit is illegal")
	}
	bucket := make(chan struct{}, totalConnLimit)
	ll := &localLimitListener{
		Listener:    l,
		buckets:     bucket,
		ipConnLimit: int64(IPConnLimit),
	}
	if cacheSize > 0 {
		ll.ipCache = cache.New(cacheSize)
	}
	return ll, nil
}

type localLimitListener struct {
	net.Listener
	buckets     chan struct{}
	closeOnce   sync.Once
	ipCache     *cache.ConcurrencyLRUCache
	ipConnLimit int64
}

func (l *localLimitListener) acquire() bool {
	select {
	case l.buckets <- struct{}{}:
		return true
	default:
		return false
	}
}

func (l *localLimitListener) release() { <-l.buckets }

func (l *localLimitListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	ip, cacheKey := getIpAndKey(c)
	if ip != "" && l.ipCache != nil {
		if counts, err := l.ipCache.IncreaseOne(cacheKey, -1); err == nil && counts > l.ipConnLimit {
			log.Warningln("ip connections reach max limit, connection will to force closed")
			return closeImmediately(c, l.ipCache), nil
		}
	}

	if l.acquire() {
		return &limitListenerConn{Conn: c, release: l.release, ipCache: l.ipCache}, nil
	}
	log.Warningln("limit forbidden, connection will to force closed")
	return closeImmediately(c, l.ipCache), nil
}

func getIpAndKey(c net.Conn) (string, string) {
	ipWithPort := c.RemoteAddr().String()
	if ipWithPort != "" {
		s := strings.Split(ipWithPort, ":")
		return s[0], fmt.Sprintf("key-conn-%s", s[0])
	}
	return "", ""
}

func closeImmediately(c net.Conn, lruCache *cache.ConcurrencyLRUCache) net.Conn {
	tcpConn, ok := c.(*net.TCPConn)
	if ok {
		if err := tcpConn.SetLinger(0); err != nil {
			log.Warningf("Error when setting linger: %s", err)
		}
	}

	err := c.Close()
	if err != nil {
		log.Warningln(err)
	}
	return &limitListenerConn{Conn: c, ipCache: lruCache}
}

func (l *localLimitListener) Close() error {
	err := l.Listener.Close()
	l.closeOnce.Do(func() { close(l.buckets) })
	return err
}

type limitListenerConn struct {
	net.Conn
	releaseOnce sync.Once
	release     func()
	ipCache     *cache.ConcurrencyLRUCache
}

func (l *limitListenerConn) Close() error {
	err := l.Conn.Close()
	l.releaseOnce.Do(l.release)
	ip, cacheKey := getIpAndKey(l.Conn)
	if ip != "" && l.ipCache != nil {
		d, err := l.ipCache.DecreaseOne(cacheKey, time.Hour)
		if err != nil {
			log.Errorln(err)
		}
		log.Infof("decrement ip connections %d", d)
	}
	return err
}
