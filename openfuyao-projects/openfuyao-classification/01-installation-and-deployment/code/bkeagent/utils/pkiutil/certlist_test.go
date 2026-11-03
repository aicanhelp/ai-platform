package pkiutil

import (
	"fmt"
	"testing"
)

func TestCertificates_Export(t *testing.T) {
	certs := GetClusterAPICertList()

	a := certs.Export("/tmp")

	for i, s := range a {
		fmt.Println(i, s)

	}
}
