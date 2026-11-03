package cache

import (
	"reflect"
	"testing"
)

func TestNewCacheKey(t *testing.T) {
	tests := []struct {
		name string
		cert string
		key  string
		ca   string
		want CertificateCacheKey
	}{
		{
			name: "all fields populated",
			cert: "/etc/cert.pem",
			key:  "/etc/key.pem",
			ca:   "/etc/ca.pem",
			want: CertificateCacheKey{
				certFile:    "/etc/cert.pem",
				keyFile:     "/etc/key.pem",
				trustCAFile: "/etc/ca.pem",
			},
		},
		{
			name: "empty strings allowed",
			cert: "",
			key:  "",
			ca:   "",
			want: CertificateCacheKey{
				certFile:    "",
				keyFile:     "",
				trustCAFile: "",
			},
		},
		{
			name: "only cert and key set",
			cert: "cert",
			key:  "key",
			ca:   "",
			want: CertificateCacheKey{
				certFile:    "cert",
				keyFile:     "key",
				trustCAFile: "",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewCacheKey(tt.cert, tt.key, tt.ca)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("NewCacheKey(%q, %q, %q) = %+v, want %+v",
					tt.cert, tt.key, tt.ca, got, tt.want)
			}
		})
	}
}

func TestNewCacheKeyDistinctInputs(t *testing.T) {
	a := NewCacheKey("cert", "key", "ca")
	b := NewCacheKey("cert2", "key", "ca")
	// different inputs produce different keys
	if reflect.DeepEqual(a, b) {
		t.Fatalf("expected distinct keys for distinct inputs, got equal: %+v", a)
	}
}
