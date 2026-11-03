package etcd

import (
	"k8s.io/apimachinery/pkg/util/wait"
	"time"
)

const (
	// EtcdListenPeerPort defines the port etcd listen on for peer traffic
	EtcdListenPeerPort = 2380

	// EtcdListenClientPort defines the port etcd listen on for client traffic
	EtcdListenClientPort = 2379

	EtcdAdvertiseClientUrlsAnnotationKey = "bkeagent.bocloud.com/etcd.advertise-client-urls"

	// Etcd defines variable used internally when referring to etcd component
	Etcd = "etcd"

	// ControlPlaneTier is the value used in the tier label to identify control plane components
	ControlPlaneTier = "control-plane"
)

var (
	StaticPodMirroringDefaultRetry = wait.Backoff{
		Steps:    30,
		Duration: 1 * time.Second,
		Factor:   1.0,
		Jitter:   0.1,
	}
)
