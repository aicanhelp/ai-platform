package main

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"marketplace-service/cmd/config"
	"marketplace-service/pkg/server"
	"marketplace-service/pkg/zlog"
)

func TestMain_SuccessPath(t *testing.T) {
	runCfg := &config.RunConfig{}
	fatalCalled := 0

	patches := gomonkey.ApplyFunc(config.NewRunConfig, func() *config.RunConfig {
		return runCfg
	}).ApplyMethod(reflect.TypeOf(runCfg), "Validate", func(_ *config.RunConfig) []error {
		return nil
	}).ApplyFunc(server.NewServer, func(_ *config.RunConfig, _ context.Context) (*server.CServer, error) {
		return &server.CServer{}, nil
	}).ApplyMethod(reflect.TypeOf(&server.CServer{}), "Run", func(_ *server.CServer, _ context.Context) error {
		return nil
	}).ApplyFunc(zlog.Fatalf, func(_ string, _ ...interface{}) {
		fatalCalled++
	}).ApplyFunc(zlog.Sync, func() error {
		return nil
	})
	defer patches.Reset()

	main()
	assert.Equal(t, 0, fatalCalled)
}

func TestMain_FailurePath(t *testing.T) {
	runCfg := &config.RunConfig{}
	fatalCalled := 0

	patches := gomonkey.ApplyFunc(config.NewRunConfig, func() *config.RunConfig {
		return runCfg
	}).ApplyMethod(reflect.TypeOf(runCfg), "Validate", func(_ *config.RunConfig) []error {
		return []error{errors.New("invalid")}
	}).ApplyFunc(server.NewServer, func(_ *config.RunConfig, _ context.Context) (*server.CServer, error) {
		return &server.CServer{}, nil
	}).ApplyMethod(reflect.TypeOf(&server.CServer{}), "Run", func(_ *server.CServer, _ context.Context) error {
		return errors.New("run failed")
	}).ApplyFunc(zlog.Fatalf, func(_ string, _ ...interface{}) {
		fatalCalled++
	}).ApplyFunc(zlog.Sync, func() error {
		return nil
	})
	defer patches.Reset()

	main()
	assert.GreaterOrEqual(t, fatalCalled, 1)
}
