package kubelet

import (
	"testing"
)

func TestRunKubeletCommand(t *testing.T) {
	c := NewRunKubeletCommand().
		SetContainerRuntime("containerd").
		SetKubeletImage("k8s.gcr.io/hyperkube:v1.21.1").
		SetPauseImage("k8s.gcr.io/pause:3.9").
		SetKubeConfigPath("/etc/kubernetes/kubelet.conf").
		SetExtraVolumes([]string{"/var/lib/kubelet/pki:/var/lib/kubelet/pki"}).
		SetExtraKubeletArgs([]string{"--root-dir=/data/root"}).
		SetK8sVersion("v1.23.17").
		SetRootDirPath("/data/root").
		SetHostIP("1")

	err := c.ExportKubeletScript(true)
	if err != nil {
		return
	}

	//cmd, args, err := c.Command()
	//if err != nil {
	//	t.Fatal(err)
	//}
	//fmt.Println(cmd)
	//fmt.Println(args)
}
