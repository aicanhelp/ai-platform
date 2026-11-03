package ping

import (
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
)

const name = "Ping"

type Ping struct{}

func New() plugin.Plugin {
	return &Ping{}
}

func (Ping) Name() string {
	return name
}

func (Ping) Param() map[string]plugin.PluginParam {
	return nil
}

func (Ping) Execute(commands []string) ([]string, error) {
	return []string{"pong", utils.HostName()}, nil
}
