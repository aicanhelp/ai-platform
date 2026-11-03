package job

import (
	"sync"

	"gopkg.openfuyao.cn/bkeagent/api/v1beta1"
	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/k8s"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/shell"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Task Global tasks
type Task struct {
	StopChan                chan struct{}        `json:"stopChan"`
	Phase                   v1beta1.CommandPhase `json:"phase"`
	ResourceVersion         string               `json:"resourceVersion"`
	Generation              int64                `json:"generation"`
	TTLSecondsAfterFinished int                  `json:"ttlSecondsAfterFinished"`
	HasAddTimer             bool                 `json:"hasAddTimer"`
	Once                    *sync.Once           `json:"once"`
}

type Job struct {
	BuiltIn builtin.BuiltIn
	K8s     k8s.K8s
	Shell   shell.Shell
	Task    map[string]*Task
}

func NewJob(client client.Client) (Job, error) {

	var j Job
	commandExec := &exec.CommandExecutor{}
	//do, err := docker.NewDockerClient()
	//if err != nil {
	//	log.Warn(err)
	//}
	//containerdClient, err := containerd.NewContainedClient()
	//if err != nil {
	//	log.Warn(err)
	//}

	j.BuiltIn = builtin.New(commandExec, client)
	j.K8s = &k8s.Task{
		K8sClient: client,
		Exec:      commandExec,
	}
	j.Shell = &shell.Task{
		Exec: commandExec,
	}
	j.Task = map[string]*Task{}

	return j, nil
}

func (t *Task) SafeClose() {
	t.Once.Do(func() {
		close(t.StopChan)
	})
}
