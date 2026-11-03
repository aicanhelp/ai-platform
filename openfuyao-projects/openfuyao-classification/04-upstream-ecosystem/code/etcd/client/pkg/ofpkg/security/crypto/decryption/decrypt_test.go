package decryption

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/ofpkg/security/crypto/decryption/api"
)

func TestDecryptParameterCustom(t *testing.T) {
	type args struct {
		key         []byte
		endpointEnv string
		timeoutEnv  string
	}
	tests := []struct {
		name    string
		args    args
		want    []byte
		wantErr bool
	}{
		{
			name: "unable decrypt plugin to return origin data",
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
				endpointEnv: "unix:///enable.sock",
				timeoutEnv:  "not number",
			},
			want:    []byte("abc"),
			wantErr: true,
		},
		{
			name: "failed while timeout is negative",
			args: args{
				key:         []byte("abc"),
				endpointEnv: "unix:///enable.sock",
				timeoutEnv:  "-100",
			},
			want:    []byte("abc"),
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(udsKey, tt.args.timeoutEnv)
			t.Setenv(timeoutKey, tt.args.timeoutEnv)
			got, err := OfTryDecryptSecret(tt.args.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("OfTryDecryptSecret() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("OfTryDecryptSecret() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOfTryDecrypt(t *testing.T) {
	oldFactory := decryptServiceFactory
	t.Cleanup(func() {
		decryptServiceFactory = oldFactory
	})
	factoryHolder := mockFactoryHolder{}
	decryptServiceFactory = factoryHolder.newMockService
	t.Setenv(udsKey, "unix:///enable.sock")

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
				gotKey, err = OfTryDecryptCert(tt.keys)
			} else {
				gotKey, err = OfTryDecryptSecret(tt.keys)
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("OfTryDecryptCert() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(gotKey, tt.want) {
				t.Errorf("OfTryDecryptCert() got = %v, want %v", gotKey, tt.want)
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
	t.Setenv(udsKey, "unix:///enable.sock")
	gotKey, err := OfTryDecryptSecret([]byte("abc"))
	if err == nil {
		t.Errorf("OfTryDecryptSecret() should return an error")
		return
	}
	if !reflect.DeepEqual(gotKey, []byte("abc")) {
		t.Errorf("OfTryDecryptSecret() got = %v, want %v", gotKey, []byte("abc"))
	}
	if factoryHolder.service != nil {
		t.Errorf("OfTryDecryptSecret() should not create service while catch a error")
	}
}

func (h mockFactoryHolder) newMockService(endpoint string, callTimeout time.Duration) (api.Service, error) {
	if h.failed {
		return nil, fmt.Errorf("failed to creat service")
	}
	mockService := &mockService{
		endpoint:    endpoint,
		callTimeout: callTimeout,
	}
	h.service = mockService
	return mockService, nil
}

type mockFactoryHolder struct {
	failed  bool
	service *mockService
}

type mockService struct {
	endpoint    string
	callTimeout time.Duration
	isRelease   bool
}

func (*mockService) DecryptSecret(data []byte) ([]byte, error) {
	result := "[DecryptSecret]" + string(data)
	return []byte(result), nil
}

func (*mockService) DecryptCert(data []byte) ([]byte, error) {
	result := "[DecryptCert]" + string(data)
	return []byte(result), nil
}
func (s *mockService) Release() {
	s.isRelease = true
}
