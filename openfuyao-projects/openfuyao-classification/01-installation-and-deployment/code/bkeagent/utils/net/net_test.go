package net

import (
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRemoveRepIP(t *testing.T) {
	ips := []net.IP{
		net.ParseIP("192.168.1.1"),
		net.ParseIP("192.168.1.1"),
		net.ParseIP("192.168.1.2"),
		net.ParseIP("192.168.1.2"),
		net.ParseIP("192.168.1.3"),
		net.ParseIP("192.168.1.3"),
	}

	res := []net.IP{
		net.ParseIP("192.168.1.1"),
		net.ParseIP("192.168.1.2"),
		net.ParseIP("192.168.1.3"),
	}

	assert.New(t).Equal(res, RemoveRepIP(ips))

}

func TestInterfaceIpExit(t *testing.T) {
	ip := "192.168.0.1"
	exit, err := InterfaceIpExit("eth0", ip)
	if err != nil {
		t.Error(err)
	}
	fmt.Println(exit)
}
