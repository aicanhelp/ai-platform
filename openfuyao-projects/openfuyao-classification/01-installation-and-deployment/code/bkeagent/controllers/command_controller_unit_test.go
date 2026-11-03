package controllers

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func TestNodeMatchNodeSelector(t *testing.T) {
	r := CommandReconciler{
		NodeName: "worker-1",
	}

	nodeSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{
			"worker-1": "worker-1",
		},
	}

	tests := []struct {
		name   string
		want   bool
		labels map[string]string
	}{
		{
			name: "nodeName match",
			want: true,
			labels: map[string]string{
				"worker-1": "worker-1",
			},
		},
		{
			name: "nodeName not match",
			want: false,
			labels: map[string]string{
				"worker-2": "worker-2",
			},
		},
		{
			name: "nodeName not valid",
			want: false,
			labels: map[string]string{
				"worker-1": "worker-2",
			},
		},
		{
			name: "ip match",
			want: false,
			labels: map[string]string{
				"172.28.8.186": "172.28.8.186",
			},
		},
		{
			name: "ip not match",
			want: false,
			labels: map[string]string{
				"0.0.0.0": "0.0.0.0",
			},
		},
		{
			name: "ip not valid",
			want: false,
			labels: map[string]string{
				"172.28.8.186": "0.0.0.0",
			},
		},
		{
			name: "ip is localhost ipv4",
			want: false,
			labels: map[string]string{
				"127.0.0.1": "127.0.0.1",
			},
		},
		{
			name: "ip is localhost ipv6",
			want: false,
			labels: map[string]string{
				"::1": "::1",
			},
		},
		{
			name: "ip and nodeName match",
			want: true,
			labels: map[string]string{
				"worker-1":     "worker-1",
				"172.28.8.186": "172.28.8.186",
			},
		},
		{
			name: "ip and nodeName nodename match",
			want: true,
			labels: map[string]string{
				"worker-1": "worker-1",
				"0.0.0.0":  "0.0.0.0",
			},
		},
		{
			name: "ip and nodeName ip match",
			want: false,
			labels: map[string]string{
				"worker-2":     "worker-2",
				"172.28.8.186": "172.28.8.186",
			},
		},
		{
			name: "ip and nodeName not match",
			want: false,
			labels: map[string]string{
				"worker-2": "worker-2",
				"0.0.0.0":  "1.1.1.1",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeSelector.MatchLabels = tt.labels
			if got := r.nodeMatchNodeSelector(nodeSelector); got != tt.want {
				t.Errorf("test case: %s \n nodeMatchNodeSelector() = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
