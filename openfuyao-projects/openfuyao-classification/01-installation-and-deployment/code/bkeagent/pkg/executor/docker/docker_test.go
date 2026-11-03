package docker

import (
	"testing"
)

func TestAddInsecureRegistries(t *testing.T) {
	name := "deploy.bocloud.k8s:40443"
	err := ConfigInsecureRegistries([]string{name})
	if err != nil {
		t.Fatal(err)
	}
	//err = ConfigCgroupDriver("systemd")
	//if err != nil {
	//	t.Fatal(err)
	//}
	//err = ConfigRuntime("richrunc")
	//if err != nil {
	//	t.Fatal(err)
	//}
	//err = ConfigDockerTls("172.28.100.206")
	//if err != nil {
	//	t.Fatal(err)
	//}
	//err = OverrideDockerService()
	//if err != nil {
	//	t.Fatal(err)
	//}

}
