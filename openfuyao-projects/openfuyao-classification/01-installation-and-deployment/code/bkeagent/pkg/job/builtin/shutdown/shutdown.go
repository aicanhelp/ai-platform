package shutdown

import (
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"os"
)

const Name = "Shutdown"

type ShutDown struct{}

func New() plugin.Plugin {
	return &ShutDown{}
}

func (s ShutDown) Name() string {
	return Name
}

func (s ShutDown) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{}
}

func (s ShutDown) Execute(commands []string) ([]string, error) {
	log.Info("shutting down BKEAgent")
	os.Exit(0)
	return nil, nil
}
