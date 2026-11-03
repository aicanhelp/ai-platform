// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package grpc

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	mockService "github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/api/testing"
	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/testing/v1"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

func Test_should_not_cache_while_create_grpc_service_failed(t *testing.T) {
	fakeEndpoint := "invalidate_endpoint"
	timeout := time.Second

	service, err := Holder.GetGRPCService(fakeEndpoint, timeout)

	assert.Error(t, err, "should return an error given invalidate endpoint")
	assert.Nil(t, service, "should return nil service given invalidate endpoint")
	assert.NotContains(t, Holder.services, key(fakeEndpoint, timeout), "should not cache service given invalidate endpoint")
}

func Test_GetGRPCService_clean_and_timeout(t *testing.T) {
	t.Parallel()
	// mock
	cleanFunc, endpoint, failed := mockGrpcServer(t)
	if failed {
		return
	}
	t.Cleanup(cleanFunc)

	type args struct {
		endpoint          string
		firstWasteTime    time.Duration
		decryptApiTimeout time.Duration
		secondWasteTime   time.Duration
	}
	tests := []struct {
		name           string
		args           args
		wantEqual      assert.BoolAssertionFunc
		wantDecryptErr bool
		decryptSuccess bool
	}{
		{
			name: "should get last one from cache before be clean",
			args: args{
				endpoint:          endpoint,
				decryptApiTimeout: time.Second, // 接口超时一秒 则超时clean时间为default时间: 3秒
			},
			wantEqual:      assert.True,
			wantDecryptErr: false,
			decryptSuccess: true,
		},
		{
			name: "should not be clean while api in use",
			args: args{
				endpoint:          endpoint,
				decryptApiTimeout: time.Second * 3,                // 接口超时3秒 则超时clean时间以API使劲按为准: 3+1=4秒
				firstWasteTime:    time.Second*3 + time.Second/10, // 模拟第一次api请求处理耗时3.1秒, 保证第二次从缓存取通道仍可用
			},
			wantEqual:      assert.True,
			wantDecryptErr: false,
			decryptSuccess: true,
		},
		{
			name: "should create a new one while the last one be clean",
			args: args{
				endpoint:          endpoint,
				decryptApiTimeout: time.Second * 2, // 接口超时2秒 则超时clean时间为: 3秒
				firstWasteTime:    time.Second * 4, // 模拟第一次api请求后间隔4秒, 通道被清理后, 可以获取到新对象
			},
			wantEqual:      assert.False,
			wantDecryptErr: false,
			decryptSuccess: true,
		},
		{
			name: "should be clean while timeout",
			args: args{
				endpoint:          endpoint,
				decryptApiTimeout: time.Second * 2, // 接口超时2秒 则超时clean时间为: 3秒
				firstWasteTime:    time.Second * 4, // 第一次取到通道后空闲间隔时间为4秒, 确保通道被释放清理, 获取到新对象
				secondWasteTime:   time.Second * 4, // 第二次拿到通道后空闲间隔时间为4秒, 确保通道被释放清理, 从而请求失败
			},
			wantEqual:      assert.False,
			wantDecryptErr: true,
			decryptSuccess: false,
		},
	}

	for _, tt := range tests {
		testCase := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			holder := &cachedService{services: make(map[string]*serviceWrap, 1)}

			gotFirst, err := holder.GetGRPCService(testCase.args.endpoint, testCase.args.decryptApiTimeout)
			if err != nil {
				t.Errorf("GetGRPCService() error = %v", err)
				return
			}
			if testCase.args.firstWasteTime > 0 {
				time.Sleep(testCase.args.firstWasteTime) // 模拟服务调用耗时
			}

			gotSecond, err := holder.GetGRPCService(testCase.args.endpoint, testCase.args.decryptApiTimeout)
			if err != nil {
				t.Errorf("GetGRPCService() error = %v", err)
				return
			}
			firstObject := fmt.Sprintf("%p", gotFirst)
			secondObject := fmt.Sprintf("%p", gotSecond)
			testCase.wantEqual(t, firstObject == secondObject, "firstObject: %s, secondObject: %s", firstObject, secondObject)

			if testCase.args.secondWasteTime > 0 {
				time.Sleep(testCase.args.secondWasteTime)
			}
			// 验证链接通道是否可用, 是否超时被关闭
			decryptResult, err := gotSecond.DecryptSecret([]byte(v1.DefaultKey))
			isDecryptErr := err != nil
			if isDecryptErr != testCase.wantDecryptErr {
				// rpc error: code = Canceled desc = grpc: the client connection is closing
				t.Errorf("decrypt error: %v, expect error %t", err, testCase.wantDecryptErr)
			}
			if isDecryptErr && !strings.Contains(err.Error(), "connection is closing") {
				t.Errorf("connection not be closed for error: %v", err)
			}
			if (string(decryptResult) == v1.DefaultValue) != testCase.decryptSuccess {
				t.Errorf("decrypt value: %s, expect: %s, expect equals: %t", string(decryptResult), v1.DefaultValue, testCase.decryptSuccess)
			}
		})
	}
}

func Test_should_ensure_timeToClean_to_stop(t *testing.T) {
	t.Parallel()
	var cache = &cachedService{services: make(map[string]*serviceWrap, 1)}
	serviceMock := &mockService.Service{}
	wrap := &serviceWrap{
		cacheKey:  "cacheKey",
		service:   serviceMock,
		period:    time.Second,
		timestamp: time.Now(),
	}
	var end = false
	go func() {
		wrap.timeToClean(cache)
		end = true
	}()
	time.Sleep(2 * time.Second)

	if !end {
		t.Errorf("timeToClean not complete on time")
	}
	if !serviceMock.IsRelease {
		t.Errorf("the service should be closed")
	}
}

func mockGrpcServer(t *testing.T) (func(), string, bool) {
	originCached := Holder.services
	originPeriod := defaultCleanPeriod
	defaultCleanPeriod = time.Second * 3 // 为缩短UT耗时, 并发验证, 全局cache缓存时间设置为固定值: 3秒
	Holder.services = make(map[string]*serviceWrap)
	endpointPath := fmt.Sprintf("@%s.sock", uuid.New().String())
	endpoint := fmt.Sprintf("unix:///%s", endpointPath)

	tmpDir, cleanTmpDirFunc, err := createTmpDirWithCleanup("decrypt-service-holder")
	if err != nil {
		t.Fatalf("failed to create temp dir, error: %v", err)
		return func() {}, "", true
	}
	mockGrpcService, err := v1.NewFileMapPlugin(endpointPath, tmpDir)
	err = mockGrpcService.Start(true)
	if err != nil {
		t.Fatalf("start mock grpc server failed, error: %v", err)
		return func() { cleanTmpDirFunc() }, "", true
	}

	return func() {
		Holder.services = originCached
		defaultCleanPeriod = originPeriod
		mockGrpcService.CleanUp()
		cleanTmpDirFunc()
	}, endpoint, false
}

func createTmpDirWithCleanup(key string) (string, func(), error) {
	tempDir, err := os.MkdirTemp("", key+"-*-decrypt-holder-test")
	if err != nil {
		return "", nil, err
	}
	return tempDir, func() {
		_ = os.RemoveAll(tempDir)
	}, nil
}
