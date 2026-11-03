package utils

import "k8s.io/apimachinery/pkg/api/resource"

// mustParseQuantity 解析资源数量，非法字符串会 panic（与 resource.MustParse 一致）。
func mustParseQuantity(q string) resource.Quantity {
	return resource.MustParse(q)
}

// bePodManifest BE Pod YAML 模板
const bePodManifest = `
apiVersion: v1
kind: Pod
metadata:
  name: be-pod
  labels:
    app: test-nginx
  annotations:
    openfuyao.com/qos-level: BE
spec:
  containers:
  - name: test-nginx
    image: nginx
    imagePullPolicy: IfNotPresent
    resources:
      limits:
        kubernetes.io/be-cpu: "500"
        kubernetes.io/be-memory: 200Mi
      requests:
        kubernetes.io/be-cpu: "250"
        kubernetes.io/be-memory: 100Mi
`

// lsPodManifest LS Pod YAML 模板
const lsPodManifest = `
apiVersion: v1
kind: Pod
metadata:
  name: ls-pod
  labels:
    app: test-nginx
  annotations:
    openfuyao.com/qos-level: LS
spec:
  containers:
  - name: test-nginx
    image: nginx
    imagePullPolicy: IfNotPresent
    resources:
      limits:
        cpu: "0.5"
        memory: 200Mi
      requests:
        cpu: "0.25"
        memory: 100Mi
`

// hlsPodManifest HLS Pod YAML 模板
const hlsPodManifest = `
apiVersion: v1
kind: Pod
metadata:
  name: hls-pod
  labels:
    app: test-nginx
  annotations:
    openfuyao.com/qos-level: HLS
spec:
  containers:
  - name: test-nginx
    image: nginx
    imagePullPolicy: IfNotPresent
    resources:
      limits:
        cpu: "1"
        memory: 100Mi
      requests:
        cpu: "1"
        memory: 100Mi
`

// stressPodManifest stress-ng Pod YAML 模板
const stressPodManifest = `
apiVersion: v1
kind: Pod
metadata:
  name: stress-ng-1
  namespace: default
  labels:
    app: stress-ng
    volcano.sh/preemptable: "true"
  annotations:
    volcano.sh/preemptable: "true"
spec:
  restartPolicy: Always
  containers:
    - name: stress-ng
      image: colinianking/stress-ng
      command: ["/bin/sh","-c"]
      args: ["tail -f /dev/null"]
      resources:
        requests:
          kubernetes.io/be-cpu: "500"
          kubernetes.io/be-memory: "512Mi"
        limits:
          kubernetes.io/be-cpu: "4000"
          kubernetes.io/be-memory: "64Gi"
`
