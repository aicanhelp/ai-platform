/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

const (
	icmpPayloadMagic = "OS_CHECK_ICMP_ECHO"
	protocolICMP     = 1
	protocolICMPv6   = 58
	pingTimeout      = 5 * time.Second
	pingTimes        = 5
	pingReadInterval = 100 * time.Microsecond
	pidMask          = 0xffff
	successRate      = 0.5
)

const (
	icmpPingStatusToStart = iota
	icmpPingStatusRunning
	icmpPingStatusToStop
	icmpPingStoping
	icmpPingStatusStopped
)

var (
	idGenerator = NewIdGenerator()
)

// PingChecker 实现了Checker接口，用于ping检查
type PingChecker struct {
	ctx      Context
	config   conf.CheckItem
	logger   *log.Logger
	targets  map[string]PingTarget
	ipv4Ping IcmpPing
	ipv6Ping IcmpPing
}

// PingTarget 定义了ping检查的目标，為了解析yaml定義
type PingTarget struct {
	IP   net.IP
	Name string
}

// PingTargets 定义了ping检查的目标，為了解析yaml定義
type PingTargets struct {
	Targets []string `yaml:"targets" json:"targets"`
}

// Init 初始化PingChecker
// 初始化PingChecker，返回初始化错误
func (c *PingChecker) Init(ctx Context, item conf.CheckItem) *ParamCheckError {
	c.logger = utils.GetLogger("PingChecker")
	c.ctx = ctx
	c.config = item

	spec := &PingTargets{}
	err := convertSpec2Special(item, &spec)
	if err != nil {
		c.logger.Printf("Failed to init PingChecker from %s, because: %s", item.FilePath, err.Error())
		return &ParamCheckError{
			FormatErr: true,
			Msg:       "Format Error",
		}
	}

	c.targets = make(map[string]PingTarget, len(spec.Targets))
	hasIpv4Target, hasIpv6Target, failedIP := c.getTarget(spec)
	if len(failedIP) > 0 {
		c.logger.Printf("Failed to init PingChecker from %s, because some ip format is error, ips: %s",
			item.FilePath, strings.Join(failedIP, ","))
		return &ParamCheckError{
			ErrField:  []string{fmt.Sprintf("spec.targets:%s", strings.Join(failedIP, ","))},
			FormatErr: true,
			Msg:       "Format Error",
		}
	}

	if hasIpv4Target {
		c.ipv4Ping, err = NewIcmpPing(c.logger, true)
		if err != nil {
			c.logger.Printf("Failed to init PingChecker from %s, because: %s", item.FilePath, err.Error())
			return &ParamCheckError{
				FormatErr: true,
				Msg:       "Format Error",
			}
		}
	}
	if hasIpv6Target {
		c.ipv6Ping, err = NewIcmpPing(c.logger, false)
		if err != nil {
			c.logger.Printf("Failed to init PingChecker from %s, because: %s", item.FilePath, err.Error())
			return &ParamCheckError{
				FormatErr: true,
				Msg:       "Format Error",
			}
		}
	}
	return nil
}

func (c *PingChecker) getTarget(spec *PingTargets) (bool, bool, []string) {
	failedIps := make([]string, 0)
	hasIpv4Target := false
	hasIpv6Target := false
	for _, target := range spec.Targets {
		target = strings.TrimSpace(target)
		if len(target) == 0 {
			continue
		}
		splitedTarget := strings.Split(target, "@")
		target := PingTarget{}

		targetIp := splitedTarget[0]
		target.Name = targetIp
		if len(splitedTarget) > 1 {
			target.Name = splitedTarget[1]
		}

		// 解析target是否是IPv4或IPv6地址
		target.IP = net.ParseIP(targetIp)
		if target.IP == nil {
			failedIps = append(failedIps, targetIp)
			continue
		}
		c.targets[targetIp] = target
		if target.IP.To4() != nil {
			hasIpv4Target = true
		} else {
			hasIpv6Target = true
		}
	}
	return hasIpv4Target, hasIpv6Target, failedIps
}

// Check 检查目标是否可达
func (c *PingChecker) Check() ItemRslt {

	rslt := ItemRslt{
		Key:      c.config.Name,
		Result:   ResultValid,
		Doc:      c.config.Doc,
		SubItems: make([]SubItemRslt, 0, len(c.targets)),
	}
	c.logger.Printf("Start to check ping targets: %v", c.targets)
	// 针对每个target，创建goroutine进行进行icmp可达性检查
	var wg sync.WaitGroup
	resultChan := make(chan SubItemRslt, len(c.targets))

	for _, target := range c.targets {
		wg.Add(1)
		go func(target PingTarget) {
			defer wg.Done()
			c.execPing(target, resultChan)
		}(target)
	}
	wg.Wait()
	close(resultChan)

	for subRslt := range resultChan {
		rslt.SubItems = append(rslt.SubItems, subRslt)
		if subRslt.Result == ResultError {
			rslt.Result = ResultError
		} else if subRslt.Result == ResultInvalid && rslt.Result != ResultError {
			rslt.Result = ResultInvalid
		}
	}

	if c.ipv4Ping != nil {
		c.ipv4Ping.Stop()
	}
	if c.ipv6Ping != nil {
		c.ipv6Ping.Stop()
	}
	c.logger.Printf("Ping check result: %v", rslt)
	return rslt
}

func (c *PingChecker) execPing(target PingTarget, resultChan chan SubItemRslt) {
	subRslt := SubItemRslt{
		Key:    target.Name,
		Expect: fmt.Sprintf("%s Reachable", target.IP),
		Result: ResultValid,
	}
	pingInstance := c.ipv4Ping
	if target.IP.To4() == nil {
		pingInstance = c.ipv6Ping
	}
	c.logger.Printf("Start to ping target %s", target.IP)
	successCount, err := pingAndGetSuccessCount(
		pingInstance, target.IP, pingTimes, pingTimeout, c.logger)
	if err != nil {
		c.logger.Printf("ICMP Ping task %d ping failed, error: %s", target.IP, err.Error())
		subRslt.Result = ResultError
	}
	if float64(successCount) < float64(pingTimes)*successRate {
		subRslt.Result = ResultInvalid
	}
	subRslt.Real = fmt.Sprintf("%d successful, %d attempts", successCount, pingTimes)
	c.logger.Printf("Ping target %s result: %v, %d/%d", target.IP, subRslt, successCount, pingTimes)
	resultChan <- subRslt
}

// IdGenerator 用于生成唯一标识符
type IdGenerator struct {
	mu      sync.Mutex
	counter int64
}

// NewIdGenerator 创建一个新的IdGenerator
func NewIdGenerator() *IdGenerator {
	return &IdGenerator{}
}

// NextId 生成下一个唯一标识符
func (g *IdGenerator) NextId() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.counter++
	return g.counter
}

// ICMPResp 定义了ICMP响应
type ICMPResp struct {
	Addr      net.Addr
	Id        int64
	Timestamp int64
	seq       int
}

func pingAndGetSuccessCount(
	icmpPing IcmpPing, target net.IP, times int,
	timeout time.Duration, logger *log.Logger) (int, error) {

	msgChan := make(chan ICMPResp, times)
	id := idGenerator.NextId()
	icmpPing.AddRespProcessor(id, func(resp ICMPResp) {
		msgChan <- resp
	})

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	successCount := 0
	for i := 1; i <= times; i++ {
		err := icmpPing.SendICMPEchoRequest(target, id, i)
		if err != nil {
			logger.Printf("ICMP Ping task %d send request failed, error: %s", id, err.Error())
			return successCount, err
		}
		select {
		case resp := <-msgChan:
			logger.Printf("ICMP Ping task %d received response: %v, expect seq: %d", id, resp, i)
			if resp.seq == i {
				successCount++
			}
			if !timer.Stop() {
				<-timer.C
			}
			timer.Reset(timeout)
		case <-timer.C:
			logger.Printf("ICMP Ping task %d timed out", id)
			timer.Reset(timeout)
		}
	}
	return successCount, nil

}

// IcmpPing 定义了 ICMP ping 操作的接口
type IcmpPing interface {
	// Start 启动 ICMP ping 服务，初始化必要的资源
	Start() error

	// AddRespProcessor 添加一个响应处理器，用于处理接收到的 ICMP 响应
	// id: 处理逻辑的唯一标识符
	// processor: 处理 ICMP 响应的函数
	AddRespProcessor(id int64, respProcessor func(ICMPResp))

	// DelRespProcessor 删除指定 ID 的响应处理器
	// id: 要删除的处理逻辑的唯一标识符
	DelRespProcessor(id int64)

	// SendICMPEchoRequest 发送 ICMP Echo 请求到指定目标
	// target: 目标 IP 地址
	// id: 请求的唯一标识符，实际与Processor相对应的
	// seq: 请求序列号
	SendICMPEchoRequest(target net.IP, id int64, seq int) error

	// Stop 停止 ICMP ping 服务，释放资源
	Stop()
}

// IcmpPingImpl 实现了IcmpPing接口，用于icmp协议的ping检查
// 可以并发检测
type IcmpPingImpl struct {
	// 使用互斥锁保护连接
	mutex *sync.RWMutex
	// 全局链接
	conn *icmp.PacketConn
	// 任务消息接收者
	taskMsgRecevers map[int64]func(ICMPResp)
	// 状态
	status int
	isIpv4 bool
	logger *log.Logger
}

// DelRespProcessor 删除指定 ID 的响应处理器
// id: 要删除的处理逻辑的唯一标识符
func (p *IcmpPingImpl) DelRespProcessor(id int64) {
	p.mutex.Lock()
	delete(p.taskMsgRecevers, id)
	p.mutex.Unlock()
}

// AddRespProcessor 添加一个响应处理器，用于处理接收到的 ICMP 响应
// id: 处理逻辑的唯一标识符
// processor: 处理 ICMP 响应的函数
func (p *IcmpPingImpl) AddRespProcessor(id int64, processor func(ICMPResp)) {
	p.mutex.Lock()
	p.taskMsgRecevers[id] = processor
	p.mutex.Unlock()
}

// getConn 获取或创建全局ICMP连接
func (p *IcmpPingImpl) newConn(isIPv4 bool) (*icmp.PacketConn, error) {

	// 根据地址类型选择连接
	var network string
	var localAddr string
	if isIPv4 {
		network = "ip4:icmp"
		localAddr = "0.0.0.0"
	} else {
		network = "ip6:ipv6-icmp"
		localAddr = "::"
	}
	c, err := icmp.ListenPacket(network, localAddr)
	if err != nil {
		return nil, fmt.Errorf("failed to create ICMP connection: %w", err)
	}
	return c, nil
}

// Start 启动ICMP Ping检查，实际就是开始启动监听
func (p *IcmpPingImpl) Start() error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// 如果能够抢到锁，那么只有可能是ToStart, ToStop, Running，Stoped, 不可能是Stoping和Starting
	// 如果已在运行，直接返回
	if p.status == icmpPingStatusRunning {
		return nil
	}
	// 只支持从生到死，不支持复活，那么如果不是ToStart，直接返回异常
	if p.status != icmpPingStatusToStart {
		return fmt.Errorf("failed to start icmp ping, status is not to start nor running, but %d", p.status)
	}
	// ToStart和Stopped状态，则创建链接并启动监听
	conn, err := p.newConn(p.isIpv4)
	if err != nil {
		p.logger.Printf("Failed to create ICMP connection: %s", err.Error())
		return fmt.Errorf("failed to create ICMP connection: %w", err)
	}
	p.conn = conn

	// 开始监听消息
	if p.status == icmpPingStatusToStart || p.status == icmpPingStatusStopped {
		go p.readIcmpMsgLoop()
	}
	p.status = icmpPingStatusRunning
	return nil
}

func (p *IcmpPingImpl) readIcmpMsgLoop() {
	conn := p.conn
	pid := os.Getpid() & pidMask
	var proto = protocolICMP
	if !p.isIpv4 {
		proto = protocolICMPv6
	}
	for {
		p.mutex.RLock()
		status := p.status
		p.mutex.RUnlock()
		if status == icmpPingStatusToStop {
			p.logger.Printf("ICMP ping read loop stoped")
			break
		}
		// 读取ICMP响应消息
		buffer := make([]byte, 1024)
		byteSize, addr, err := conn.ReadFrom(buffer)
		// 如果等待停止，则退出循环并关闭链接
		if p.status == icmpPingStatusToStop {
			p.logger.Printf("ICMP ping read loop stoped")
			return
		}
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				// 超时，继续等待
				continue
			}
			p.logger.Printf("ICMP conn error while read, error: %s", err.Error())
			break
		}
		icmpMsg, err := icmp.ParseMessage(proto, buffer[:byteSize])

		if err != nil {
			p.logger.Printf("Failed to parse ICMP message: %s", err.Error())
			continue
		}
		p.parseEchoReply(icmpMsg, addr, pid)
	}
	p.mutex.Lock()
	defer p.mutex.Unlock()
	err := conn.Close()
	if err != nil {
		p.logger.Printf("ICMP conn error while close, error: %s", err.Error())
	}
	p.status = icmpPingStatusStopped
	p.conn = nil
}

func (p *IcmpPingImpl) parseEchoReply(icmpMsg *icmp.Message, addr net.Addr, pid int) bool {
	switch icmpMsg.Type {
	case ipv4.ICMPTypeEchoReply, ipv6.ICMPTypeEchoReply:
		echo, ok := icmpMsg.Body.(*icmp.Echo)
		if !ok {
			p.logger.Printf("Failed to parse ICMP message, not Echo msg")
			return false
		}
		if echo.ID != pid {
			p.logger.Printf("Failed to parse ICMP message, other pid: %d", echo.ID)
			return false

		}
		payload := &ICMPPayload{Header: icmpPayloadMagic}
		payload.Unmarshal(echo.Data)
		// payload的Head进行byte对比
		if payload.Header != icmpPayloadMagic {
			p.logger.Printf("Failed to parse ICMP message, payload magic does not match: %v, expect: %v",
				payload.Header, icmpPayloadMagic)
			return false
		}
		resp := ICMPResp{
			Addr:      addr,
			Id:        payload.Id,
			Timestamp: payload.Timestamp,
			seq:       echo.Seq,
		}
		receiver, ok := p.taskMsgRecevers[resp.Id]
		if ok {
			receiver(resp)
			return true
		}
		return false
	default:
		return false
	}
}

// Stop 停止 ICMP ping 服务，释放资源
func (p *IcmpPingImpl) Stop() {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if p.status == icmpPingStatusToStop {
		return
	}
	p.status = icmpPingStatusToStop
}

// SendICMPEchoRequest 发送带target和ID的ICMP Echo请求
func (p *IcmpPingImpl) SendICMPEchoRequest(target net.IP, id int64, seq int) error {
	payload := &ICMPPayload{
		Id:        id,
		Timestamp: time.Now().UnixNano(),
	}
	payloadBytes, err := payload.Marshal()
	if err != nil {
		return fmt.Errorf("failed to marshal ICMP payload: %w", err)
	}

	// 根据目标地址类型设置ICMP消息类型
	var msgType icmp.Type
	if target.To4() != nil {
		msgType = ipv4.ICMPTypeEcho
	} else {
		msgType = ipv6.ICMPTypeEchoRequest
	}

	msg := icmp.Message{
		Type: msgType,
		Code: 0,
		Body: &icmp.Echo{
			ID:   os.Getpid() & pidMask,
			Seq:  seq,
			Data: payloadBytes,
		},
	}
	// 序列化消息
	msgBytes, err := msg.Marshal(nil)
	if err != nil {
		return fmt.Errorf("failed to marshal ICMP message: %w", err)
	}
	_, err = p.conn.WriteTo(msgBytes, &net.IPAddr{IP: target})
	if err != nil {
		return fmt.Errorf("failed to send ICMP echo request: %w", err)
	}
	return nil
}

// NewIcmpPing 创建一个新的IcmpPing实例
// isIpv4: 是否为IPv4协议
func NewIcmpPing(logger *log.Logger, isIpv4 bool) (IcmpPing, error) {
	ping := IcmpPingImpl{
		logger:          logger,
		isIpv4:          isIpv4,
		taskMsgRecevers: make(map[int64]func(ICMPResp)),
		status:          icmpPingStatusToStart,
		mutex:           &sync.RWMutex{},
		conn:            nil,
	}
	err := ping.Start()
	if err != nil {
		return nil, err
	}
	return &ping, nil
}

// ICMPPayload 定义了ICMP负载
// 负载格式：16字节头 + 8字节ID + 8字节时间戳
type ICMPPayload struct {
	Header    string
	Id        int64
	Timestamp int64
}

// Marshal 序列化ICMP负载
func (p *ICMPPayload) Marshal() ([]byte, error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(p); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Unmarshal 反序列化ICMP负载
func (p *ICMPPayload) Unmarshal(data []byte) error {
	decoder := gob.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(p); err != nil {
		return err
	}
	return nil
}

func init() {
	factory := func() Checker {
		return &PingChecker{}
	}
	RegisterCheckerFactory("ping", factory)
	RegisterCheckerFactory("ping-checker", factory)
}
