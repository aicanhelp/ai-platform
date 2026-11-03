package plugin

import (
	"fmt"
	"testing"
)

func TestGetBKECluster(t *testing.T) {
	ns := "zzz"
	name := "zzz"
	cluster, err := GetBKECluster(ns + ":" + name)
	if err != nil {
		return
	}
	fmt.Println(cluster.GetName())
}
