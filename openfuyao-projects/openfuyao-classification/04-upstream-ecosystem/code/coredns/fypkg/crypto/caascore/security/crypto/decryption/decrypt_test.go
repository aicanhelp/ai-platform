// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package decryption

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/api"
	mock "github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/api/testing"
)

func TestDecryptParameterCustom(t *testing.T) {
	type args struct {
		key         []byte
		endpointEnv string
		timeoutEnv  string
	}
	test := []struct {
		name    string
		args    args
		want    []byte
		wantErr bool
	}{
		{
			name: "unable decrypt plugin to return origin date",
			args: args{
				key:         []byte("abc"),
				endpointEnv: "",
				timeoutEnv:  "",
			},
			want: []byte("abc"),
		},
		{
			name: "failed while timeout is not a number",
			args: args{
				key:         []byte("abc"),
				endpointEnv: "unix://enable.sock",
				timeoutEnv:  "not number",
			},
			want:    []byte("abc"),
			wantErr: true,
		},
		{
			name: "failed while timeout is not a negative",
			args: args{
				key:         []byte("abc"),
				endpointEnv: "unix://enable.sock",
				timeoutEnv:  "-100",
			},
			want:    []byte("abc"),
			wantErr: true,
		},
	}
	for _, tt := range test {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(udsKey, tt.args.endpointEnv)
			t.Setenv(timeoutKey, tt.args.timeoutEnv)
			got, err := FYTryDecryptSecret(tt.args.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("FYTryDecryptSecret() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("FYTryDecryptSecret() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFYTryDecrypt(t *testing.T) {
	oldFactory := decryptServiceFactory
	t.Cleanup(func() {
		decryptServiceFactory = oldFactory
	})
	factoryHolder := mockFactoryHolder{}
	decryptServiceFactory = factoryHolder.newMockService
	t.Setenv(udsKey, "unix://enable.sock")

	tests := []struct {
		name         string
		certOrSecret bool
		keys         []byte
		want         []byte
		wantErr      bool
	}{
		{
			name:         "decrypt cert key",
			certOrSecret: true,
			keys:         []byte("abc"),
			want:         []byte("[DecryptCert]abc"),
		},
		{
			name:         "decrypt secret key",
			certOrSecret: false,
			keys:         []byte("abc"),
			want:         []byte("[DecryptSecret]abc"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotKey []byte
			var err error
			if tt.certOrSecret {
				gotKey, err = FYTryDecryptCert(tt.keys)
			} else {
				gotKey, err = FYTryDecryptSecret(tt.keys)
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("FYTryDecrypt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotKey, tt.want) {
				t.Errorf("FYTryDecrypt() got = %v, want %v", gotKey, tt.want)
			}
		})
	}
}

func TestShouldErrorWithOriginValueWhileServiceCreateFailed(t *testing.T) {
	oldFactory := decryptServiceFactory
	t.Cleanup(func() {
		decryptServiceFactory = oldFactory
	})
	factoryHolder := mockFactoryHolder{
		failed: true,
	}
	decryptServiceFactory = factoryHolder.newMockService
	t.Setenv(udsKey, "unix://enable.sock")
	gotKey, err := FYTryDecryptSecret([]byte("abc"))
	if err == nil {
		t.Errorf("FYTryDecryptSecret() should be error")
		return
	}
	if !reflect.DeepEqual(gotKey, []byte("abc")) {
		t.Errorf("FYTryDecryptSecret() got = %v, want %v", gotKey, []byte("abc"))
	}
	if factoryHolder.service != nil {
		t.Errorf("should not create service while error")
	}
}

type mockFactoryHolder struct {
	failed  bool
	service *mock.Service
}

func (h *mockFactoryHolder) newMockService(endpoint string, callTimeout time.Duration) (api.Service, error) {
	if h.failed {
		return nil, fmt.Errorf("create service failed")
	}
	mockService := &mock.Service{}
	h.service = mockService
	return mockService, nil
}
