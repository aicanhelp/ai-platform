/*
Copyright (c) 2026 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// check  package provides check functions for oschecktool
package checker

import (
	"errors"
	"net"
	"os"
	"sync"
	"testing"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

// mockIcmpPing 实现 IcmpPing 接口，用于测试，不依赖真实 socket 与网络。
type mockIcmpPing struct {
	mu         sync.Mutex
	processors map[int64]func(ICMPResp)
	sendErr    error
}

func newMockIcmpPing(sendErr error) *mockIcmpPing {
	return &mockIcmpPing{
		processors: make(map[int64]func(ICMPResp)),
		sendErr:    sendErr,
	}
}

func (m *mockIcmpPing) Start() error { return nil }

func (m *mockIcmpPing) AddRespProcessor(id int64, processor func(ICMPResp)) {
	m.mu.Lock()
	m.processors[id] = processor
	m.mu.Unlock()
}

func (m *mockIcmpPing) DelRespProcessor(id int64) {
	m.mu.Lock()
	delete(m.processors, id)
	m.mu.Unlock()
}

// SendICMPEchoRequest 模拟发送 ICMP 回显请求。
// 若 sendErr 不为空则直接返回错误；否则同步触发已注册的响应处理器，
// 模拟一次成功的回显应答（seq 与入参一致）。
func (m *mockIcmpPing) SendICMPEchoRequest(target net.IP, id int64, seq int) error {
	if m.sendErr != nil {
		return m.sendErr
	}
	m.mu.Lock()
	processor := m.processors[id]
	m.mu.Unlock()
	if processor != nil {
		processor(ICMPResp{Id: id, seq: seq})
	}
	return nil
}

func (m *mockIcmpPing) Stop() {}

// newPingCheckerWithMocks 构造一个直接注入 mock IcmpPing 的 PingChecker，
// 避免触发 Init 中的真实 ICMP socket 创建。
func newPingCheckerWithMocks(targets []PingTarget, ipv4Ping, ipv6Ping IcmpPing) *PingChecker {
	targetMap := make(map[string]PingTarget, len(targets))
	for _, t := range targets {
		targetMap[t.IP.String()] = t
	}
	return &PingChecker{
		config:   conf.CheckItem{Name: "ping-test", Doc: "ping check"},
		logger:   utils.GetLogger("PingChecker"),
		targets:  targetMap,
		ipv4Ping: ipv4Ping,
		ipv6Ping: ipv6Ping,
	}
}

// TestPingCheckerInit 测试 Init 方法及目标解析逻辑。
// 通过直接测试 getTarget 与不触发 socket 创建的 Init 路径，
// 避免依赖管理员权限与真实 ICMP socket。
func TestPingCheckerInit(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		_ = os.Remove(logPath)
	}()

	ctx := Context{}

	t.Run("合法IPv4目标解析", func(t *testing.T) {
		pc := &PingChecker{targets: make(map[string]PingTarget)}
		spec := &PingTargets{Targets: []string{"192.168.1.1"}}
		hasIpv4, hasIpv6, failed := pc.getTarget(spec)
		if !hasIpv4 || hasIpv6 || len(failed) != 0 {
			t.Errorf("getTarget() = hasIpv4=%v, hasIpv6=%v, failed=%v, want hasIpv4=true", hasIpv4, hasIpv6, failed)
		}
		if len(pc.targets) != 1 {
			t.Errorf("expected 1 target, got %d", len(pc.targets))
		}
	})

	t.Run("合法IPv6目标解析", func(t *testing.T) {
		pc := &PingChecker{targets: make(map[string]PingTarget)}
		spec := &PingTargets{Targets: []string{"::1"}}
		hasIpv4, hasIpv6, failed := pc.getTarget(spec)
		if hasIpv4 || !hasIpv6 || len(failed) != 0 {
			t.Errorf("getTarget() = hasIpv4=%v, hasIpv6=%v, failed=%v, want hasIpv6=true", hasIpv4, hasIpv6, failed)
		}
		if len(pc.targets) != 1 {
			t.Errorf("expected 1 target, got %d", len(pc.targets))
		}
	})

	t.Run("带名称目标解析", func(t *testing.T) {
		pc := &PingChecker{targets: make(map[string]PingTarget)}
		spec := &PingTargets{Targets: []string{"192.168.1.1@gateway"}}
		hasIpv4, _, failed := pc.getTarget(spec)
		if !hasIpv4 || len(failed) != 0 {
			t.Errorf("getTarget() = hasIpv4=%v, failed=%v, want hasIpv4=true", hasIpv4, failed)
		}
		target, ok := pc.targets["192.168.1.1"]
		if !ok {
			t.Fatalf("expected target 192.168.1.1, not found")
		}
		if target.Name != "gateway" {
			t.Errorf("expected name gateway, got %s", target.Name)
		}
	})

	t.Run("非法IP地址", func(t *testing.T) {
		pc := &PingChecker{targets: make(map[string]PingTarget)}
		spec := &PingTargets{Targets: []string{"invalid-ip"}}
		hasIpv4, hasIpv6, failed := pc.getTarget(spec)
		if hasIpv4 || hasIpv6 || len(failed) != 1 || failed[0] != "invalid-ip" {
			t.Errorf("getTarget() = hasIpv4=%v, hasIpv6=%v, failed=%v, want failed=[invalid-ip]", hasIpv4, hasIpv6, failed)
		}
		if len(pc.targets) != 0 {
			t.Errorf("expected 0 target, got %d", len(pc.targets))
		}
	})

	t.Run("空目标Init成功", func(t *testing.T) {
		item := conf.CheckItem{
			Name: "ping-empty",
			Spec: map[string]interface{}{
				"targets": []string{},
			},
		}
		pc := &PingChecker{}
		if err := pc.Init(ctx, item); err != nil {
			t.Errorf("Init() with empty targets error = %v, want nil", err)
		}
		if len(pc.targets) != 0 {
			t.Errorf("expected 0 target, got %d", len(pc.targets))
		}
	})

	t.Run("非法IP Init失败", func(t *testing.T) {
		item := conf.CheckItem{
			Name: "ping-invalid",
			Spec: map[string]interface{}{
				"targets": []string{"invalid-ip"},
			},
		}
		pc := &PingChecker{}
		if err := pc.Init(ctx, item); err == nil {
			t.Errorf("Init() with invalid ip error = nil, want error")
		}
	})

	t.Run("spec格式错误Init失败", func(t *testing.T) {
		item := conf.CheckItem{
			Name: "ping-bad-spec",
			Spec: map[string]interface{}{
				"targets": "not-a-list",
			},
		}
		pc := &PingChecker{}
		if err := pc.Init(ctx, item); err == nil {
			t.Errorf("Init() with bad spec error = nil, want error")
		}
	})
}

// TestPingCheckerCheck 测试 Check 方法。
// 通过注入 mockIcmpPing 控制每次 ping 的结果，确保测试确定且不依赖真实网络。
func TestPingCheckerCheck(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		_ = os.Remove(logPath)
	}()

	t.Run("ipv4全部成功", func(t *testing.T) {
		target := PingTarget{IP: net.ParseIP("192.168.1.1"), Name: "192.168.1.1"}
		pc := newPingCheckerWithMocks([]PingTarget{target}, newMockIcmpPing(nil), nil)
		result := pc.Check()
		if result.Result != ResultValid {
			t.Errorf("Check() result = %v, want ResultValid", result.Result)
		}
		if len(result.SubItems) != 1 {
			t.Fatalf("expected 1 subItem, got %d", len(result.SubItems))
		}
		if result.SubItems[0].Result != ResultValid {
			t.Errorf("subItem result = %v, want ResultValid", result.SubItems[0].Result)
		}
	})

	t.Run("ipv6全部成功", func(t *testing.T) {
		target := PingTarget{IP: net.ParseIP("::1"), Name: "::1"}
		pc := newPingCheckerWithMocks([]PingTarget{target}, nil, newMockIcmpPing(nil))
		result := pc.Check()
		if result.Result != ResultValid {
			t.Errorf("Check() result = %v, want ResultValid", result.Result)
		}
		if len(result.SubItems) != 1 {
			t.Fatalf("expected 1 subItem, got %d", len(result.SubItems))
		}
		if result.SubItems[0].Result != ResultValid {
			t.Errorf("subItem result = %v, want ResultValid", result.SubItems[0].Result)
		}
	})

	t.Run("发送失败结果无效", func(t *testing.T) {
		target := PingTarget{IP: net.ParseIP("192.168.1.1"), Name: "192.168.1.1"}
		pc := newPingCheckerWithMocks(
			[]PingTarget{target},
			newMockIcmpPing(errors.New("send failed")),
			nil,
		)
		result := pc.Check()
		if result.Result != ResultInvalid {
			t.Errorf("Check() result = %v, want ResultInvalid", result.Result)
		}
		if len(result.SubItems) != 1 {
			t.Fatalf("expected 1 subItem, got %d", len(result.SubItems))
		}
		if result.SubItems[0].Result != ResultInvalid {
			t.Errorf("subItem result = %v, want ResultInvalid", result.SubItems[0].Result)
		}
	})

	t.Run("多目标混合结果", func(t *testing.T) {
		okTarget := PingTarget{IP: net.ParseIP("192.168.1.1"), Name: "ok"}
		failTarget := PingTarget{IP: net.ParseIP("::1"), Name: "fail"}
		pc := newPingCheckerWithMocks(
			[]PingTarget{okTarget, failTarget},
			newMockIcmpPing(nil),
			newMockIcmpPing(errors.New("send failed")),
		)
		result := pc.Check()
		if result.Result != ResultInvalid {
			t.Errorf("Check() result = %v, want ResultInvalid", result.Result)
		}
		if len(result.SubItems) != 2 {
			t.Fatalf("expected 2 subItems, got %d", len(result.SubItems))
		}
	})
}
