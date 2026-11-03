package utils

import (
	"os"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	GroupName  = "serverlessdb.openfuyao.cn"
	GroupVer   = "v1"
	KindPool   = "DBResourcePool"
	KindDBInst = "DBInstance"
	ResPools   = "dbresourcepools"
	ResDBInsts = "dbinstances"

	LabelAppRef    = "app-ref"
	LabelDBPool    = "dbresourcepool"
	LabelManaged   = GroupName + "/managed"
	AnnoPodAbsent  = GroupName + "/pod-absent-since"
	AnnoResizeSpec = GroupName + "/resize-spec"
	AnnoLastResize = GroupName + "/last-resize-spec"

	ContainerCompute = "compute"
	PortNamePg       = "pg"
	DefaultPgPort    = 5432

	OperatorPodLabel = "app=serverlessdb-operator"
	DefaultOpNs      = "default"
	TestNamespace    = "serverlessdb-test"

	DefaultComputeImage = "postgres:17"
	ComputeCpuReq       = "500m"
	ComputeMemReq       = "256Mi"
	ComputeCpuLimit     = "1"
	ComputeMemLimit     = "512Mi"

	OperatorGinPort = 8080
)

var (
	PoolGVR = schema.GroupVersionResource{Group: GroupName, Version: GroupVer, Resource: ResPools}
	DBIGVR  = schema.GroupVersionResource{Group: GroupName, Version: GroupVer, Resource: ResDBInsts}
)

// EnvOr returns the value of env var key, or fallback if unset/empty.
func EnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// LabelEq builds a "key=value" label selector string.
func LabelEq(key, val string) string {
	return key + "=" + val
}
