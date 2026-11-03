package featuregate

import (
	"testing"
)

func TestCertHotLoadingDefault(t *testing.T) {
	// Reset to default to avoid coupling with other tests that may flip it.
	CertHotLoading = false
	if CertHotLoading {
		t.Fatalf("CertHotLoading default = true, want false")
	}
}

func TestCertHotLoadingToggle(t *testing.T) {
	CertHotLoading = true
	if !CertHotLoading {
		t.Fatalf("after setting true, CertHotLoading = false, want true")
	}
	CertHotLoading = false
	if CertHotLoading {
		t.Fatalf("after setting false, CertHotLoading = true, want false")
	}
}
