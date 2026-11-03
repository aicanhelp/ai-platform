package selfupdate

import (
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"testing"
)

func TestUpdatePlugin_Execute(t *testing.T) {

	updater := New(&exec.CommandExecutor{})

	commands := []string{
		Name,
	}

	updater.Execute(commands)

}
