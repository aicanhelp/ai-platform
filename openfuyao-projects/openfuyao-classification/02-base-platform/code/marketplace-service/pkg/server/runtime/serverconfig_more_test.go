package runtime

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"marketplace-service/pkg/constant"
)

func TestNewServerConfigBranches(t *testing.T) {
	t.Setenv("SERVICE_PORT", "9443")
	t.Setenv("ENABLE_TLS", "false")
	cfg := NewServerConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, 9443, cfg.InsecurePort)
	assert.Equal(t, 0, cfg.SecurePort)

	t.Setenv("ENABLE_TLS", "true")
	patchNotExist := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	cfg = NewServerConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, 9443, cfg.InsecurePort)
	assert.Equal(t, 0, cfg.SecurePort)
	patchNotExist.Reset()

	patchError := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, assert.AnError
	})
	cfg = NewServerConfig()
	assert.Nil(t, cfg)
	patchError.Reset()

	patchOK := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, nil
	})
	cfg = NewServerConfig()
	assert.NotNil(t, cfg)
	assert.Equal(t, 9443, cfg.SecurePort)
	assert.Equal(t, constant.TLSCertPath, cfg.CertFile)
	patchOK.Reset()
}

func TestServerConfigValidateBranches(t *testing.T) {
	cfg := &ServerConfig{}
	errs := cfg.Validate()
	assert.NotEmpty(t, errs)

	cfg = &ServerConfig{
		SecurePort: 1000,
	}
	errs = cfg.Validate()
	assert.GreaterOrEqual(t, len(errs), 2)

	cfg = &ServerConfig{
		SecurePort: 1000,
		CertFile:   "x",
		PrivateKey: "y",
	}
	patch := gomonkey.ApplyFunc(os.Stat, func(_ string) (os.FileInfo, error) {
		return nil, assert.AnError
	})
	defer patch.Reset()
	errs = cfg.Validate()
	assert.Len(t, errs, 2)
}

