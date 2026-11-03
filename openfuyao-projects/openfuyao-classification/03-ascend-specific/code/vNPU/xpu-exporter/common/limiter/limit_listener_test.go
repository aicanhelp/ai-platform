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
	"net"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"huawei.com/xpu-exporter/common/cache"
)

const (
	ConnLimit = 2
)

type mockLicener struct {
}

func (l *mockLicener) Accept() (net.Conn, error) {
	return &net.TCPConn{}, nil
}

func (l *mockLicener) Addr() net.Addr {
	return &net.IPAddr{
		IP:   []byte("127.0.0.1"),
		Zone: "",
	}
}

func (l *mockLicener) Close() error {
	return nil
}

func TestLimitListener(t *testing.T) {
	l, err := LimitListener(&mockLicener{}, maxConnection, maxIPConnection, DefaultDataLimit)
	assert.NotEqual(t, nil, l)
	assert.Equal(t, nil, err)

	_, err = LimitListener(&mockLicener{}, maxConnection+1, maxIPConnection, DefaultDataLimit)
	assert.NotEqual(t, nil, err)

	_, err = LimitListener(&mockLicener{}, maxConnection, maxIPConnection+1, DefaultDataLimit)
	assert.NotEqual(t, nil, err)
}

func TestAccept(t *testing.T) {
	limitLor, err := LimitListener(&mockLicener{}, ConnLimit, ConnLimit, DefaultCacheSize)
	if err != nil {
		t.Error("LimitListener test failed")
	}
	l, ok := limitLor.(*localLimitListener)
	if !ok {
		t.Error("test failed")
	}
	patchGetIpAndKey := gomonkey.ApplyFunc(getIpAndKey, func(net.Conn) (string, string) {
		return "127.0.0.1", "key-127.0.0.1"
	})
	defer patchGetIpAndKey.Reset()
	_, err = l.Accept()
	if err != nil {
		t.Error("Accept test failed")
	}

	patchAcquire := gomonkey.ApplyPrivateMethod(l, "acquire", func(*localLimitListener) bool {
		return false
	})
	defer patchAcquire.Reset()
	con, err := l.Accept()
	assert.Equal(t, nil, err)
	conm, ok := con.(*limitListenerConn)
	if !ok {
		t.Error("Accept test failed")
	}
	if conm.release == nil && conm.ipCache == nil {
		t.Error("Accept test failed")
	} else {
		t.Log("Accept test success")
	}
}

func TestGetIpAndKey(t *testing.T) {
	c := net.TCPConn{}
	patch := gomonkey.ApplyMethodFunc(&c, "RemoteAddr", func() net.Addr {
		return &net.IPAddr{
			IP:   []byte("127.0.0.1"),
			Zone: "",
		}
	})
	defer patch.Reset()
	ip, _ := getIpAndKey(&c)
	assert.NotEqual(t, "", ip)
}

type mockErrorListener struct{}

func (l *mockErrorListener) Accept() (net.Conn, error) {
	return nil, errors.New("mock accept error")
}

func (l *mockErrorListener) Addr() net.Addr {
	return &net.IPAddr{IP: net.ParseIP("127.0.0.1")}
}

func (l *mockErrorListener) Close() error {
	return nil
}

type mockAddr struct {
	str string
}

func (a *mockAddr) Network() string { return "tcp" }
func (a *mockAddr) String() string  { return a.str }

type mockConn struct {
	addr net.Addr
	cerr error
}

func (c *mockConn) Read(b []byte) (int, error)         { return 0, nil }
func (c *mockConn) Write(b []byte) (int, error)        { return 0, nil }
func (c *mockConn) Close() error                       { return c.cerr }
func (c *mockConn) LocalAddr() net.Addr                { return nil }
func (c *mockConn) RemoteAddr() net.Addr               { return c.addr }
func (c *mockConn) SetDeadline(t time.Time) error      { return nil }
func (c *mockConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *mockConn) SetWriteDeadline(t time.Time) error { return nil }

type mockConnListener struct{}

func (l *mockConnListener) Accept() (net.Conn, error) {
	return &mockConn{addr: &mockAddr{str: "127.0.0.1:50000"}, cerr: nil}, nil
}

func (l *mockConnListener) Addr() net.Addr {
	return &net.IPAddr{IP: net.ParseIP("127.0.0.1")}
}

func (l *mockConnListener) Close() error {
	return nil
}

func TestLimitListenerNegativeTotalConnLimit(t *testing.T) {
	_, err := LimitListener(&mockLicener{}, -1, maxIPConnection, DefaultDataLimit)
	assert.NotNil(t, err)
	assert.Equal(t, "the parameter totalConnLimit is illegal", err.Error())
}

func TestLimitListenerNegativeIPConnLimit(t *testing.T) {
	_, err := LimitListener(&mockLicener{}, maxConnection, -1, DefaultDataLimit)
	assert.NotNil(t, err)
	assert.Equal(t, "the parameter IPConnLimit is illegal", err.Error())
}

func TestLimitListenerZeroCacheSize(t *testing.T) {
	l, err := LimitListener(&mockLicener{}, maxConnection, maxIPConnection, 0)
	assert.Nil(t, err)
	ll, ok := l.(*localLimitListener)
	if !ok {
		t.Fatal("type assertion failed")
		return
	}
	assert.Nil(t, ll.ipCache)
}

func TestAcquireAndRelease(t *testing.T) {
	l, err := LimitListener(&mockLicener{}, 2, 2, 0)
	assert.Nil(t, err)
	ll, ok := l.(*localLimitListener)
	if !ok {
		t.Fatal("type assertion failed")
		return
	}
	assert.True(t, ll.acquire())
	assert.True(t, ll.acquire())
	assert.False(t, ll.acquire())
	ll.release()
	assert.True(t, ll.acquire())
	ll.release()
	ll.release()
}

func TestAcceptListenerError(t *testing.T) {
	l, err := LimitListener(&mockErrorListener{}, ConnLimit, ConnLimit, 0)
	assert.Nil(t, err)
	_, err = l.Accept()
	assert.NotNil(t, err)
}

func TestAcceptIPLimitExceeded(t *testing.T) {
	l, err := LimitListener(&mockConnListener{}, ConnLimit, 1, DefaultCacheSize)
	assert.Nil(t, err)
	ll, ok := l.(*localLimitListener)
	if !ok {
		t.Fatal("type assertion failed")
		return
	}
	patchIncrease := gomonkey.ApplyMethodFunc(&cache.ConcurrencyLRUCache{}, "IncreaseOne", func(_ string, _ time.Duration) (int64, error) {
		return 10, nil
	})
	defer patchIncrease.Reset()
	c, err := ll.Accept()
	assert.Nil(t, err)
	_, ok = c.(*limitListenerConn)
	assert.True(t, ok)
}

func TestAcceptIncreaseOneError(t *testing.T) {
	l, err := LimitListener(&mockConnListener{}, ConnLimit, 1, DefaultCacheSize)
	assert.Nil(t, err)
	ll, ok := l.(*localLimitListener)
	if !ok {
		t.Fatal("type assertion failed")
		return
	}
	patchIncrease := gomonkey.ApplyMethodFunc(&cache.ConcurrencyLRUCache{}, "IncreaseOne", func(_ string, _ time.Duration) (int64, error) {
		return 0, errors.New("increase error")
	})
	defer patchIncrease.Reset()
	c, err := ll.Accept()
	assert.Nil(t, err)
	lc, ok2 := c.(*limitListenerConn)
	assert.True(t, ok2)
	assert.NotNil(t, lc.release)
}

func TestCloseImmediatelyNonTCPConn(t *testing.T) {
	mc := &mockConn{addr: &mockAddr{str: "127.0.0.1:50000"}, cerr: nil}
	c := closeImmediately(mc, nil)
	lc, ok := c.(*limitListenerConn)
	assert.True(t, ok)
	assert.Nil(t, lc.ipCache)
}

func TestCloseImmediatelyCloseError(t *testing.T) {
	mc := &mockConn{addr: &mockAddr{str: "127.0.0.1:50000"}, cerr: errors.New("close error")}
	c := closeImmediately(mc, nil)
	_, ok := c.(*limitListenerConn)
	assert.True(t, ok)
}

func TestCloseImmediatelyWithCache(t *testing.T) {
	lru := cache.New(DefaultCacheSize)
	mc := &mockConn{addr: &mockAddr{str: "127.0.0.1:50000"}, cerr: nil}
	c := closeImmediately(mc, lru)
	lc, ok := c.(*limitListenerConn)
	assert.True(t, ok)
	assert.NotNil(t, lc.ipCache)
}

func TestLocalLimitListenerClose(t *testing.T) {
	l, err := LimitListener(&mockLicener{}, ConnLimit, ConnLimit, 0)
	assert.Nil(t, err)
	ll, ok := l.(*localLimitListener)
	if !ok {
		t.Fatal("type assertion failed")
		return
	}
	err = ll.Close()
	assert.Nil(t, err)
	err = ll.Close()
	assert.Nil(t, err)
}

func TestLimitListenerConnClose(t *testing.T) {
	mc := &mockConn{addr: &mockAddr{str: "127.0.0.1:50000"}}
	released := false
	conn := &limitListenerConn{
		Conn:    mc,
		release: func() { released = true },
	}
	err := conn.Close()
	assert.Nil(t, err)
	assert.True(t, released)
	err = conn.Close()
	assert.Nil(t, err)
}

func newLimitListenerConn(ip string, ipCache *cache.ConcurrencyLRUCache) (*limitListenerConn, *bool) {
	mc := &mockConn{addr: &mockAddr{str: ip}}
	released := false
	conn := &limitListenerConn{
		Conn:    mc,
		release: func() { released = true },
		ipCache: ipCache,
	}
	return conn, &released
}

func TestLimitListenerConnCloseWithIPCache(t *testing.T) {
	lru := cache.New(DefaultCacheSize)
	_, _ = lru.IncreaseOne("key-conn-127", time.Hour)
	conn, released := newLimitListenerConn("127.0.0.1:50000", lru)
	err := conn.Close()
	assert.Nil(t, err)
	assert.True(t, *released)
}

func TestLimitListenerConnCloseDecreaseError(t *testing.T) {
	lru := cache.New(DefaultCacheSize)
	patchDecrease := gomonkey.ApplyMethodFunc(&cache.ConcurrencyLRUCache{}, "DecreaseOne", func(_ string, _ time.Duration) (int64, error) {
		return 0, errors.New("mock decrease error")
	})
	defer patchDecrease.Reset()
	conn, released := newLimitListenerConn("127.0.0.1:50000", lru)
	err := conn.Close()
	assert.Nil(t, err)
	assert.True(t, *released)
}

func TestLimitListenerConnCloseNoIP(t *testing.T) {
	conn, released := newLimitListenerConn("", nil)
	err := conn.Close()
	assert.Nil(t, err)
	assert.True(t, *released)
}

func TestGetIpAndKeyEmpty(t *testing.T) {
	mc := &mockConn{addr: &mockAddr{str: ""}}
	ip, key := getIpAndKey(mc)
	assert.Equal(t, "", ip)
	assert.Equal(t, "", key)
}
