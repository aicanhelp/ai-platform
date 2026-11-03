// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	v1 "openfuyao.com/npu-operator/api/v1"
)

func clusterdProbeTestTemplate(t *testing.T) *appsv1.Deployment {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "assets", "clusterd", "0500_deployment.yaml"))
	require.NoError(t, err)
	dep := &appsv1.Deployment{}
	require.NoError(t, yaml.Unmarshal(data, dep))
	return dep
}

type clusterdProbeCommandCase struct {
	name    string
	command []string // nil retains the real asset's bash -c -- command
	args    []string
	port    int32
	https   bool
	err     string
}

var clusterdProbeCommandCases = []clusterdProbeCommandCase{
	{name: "old image", args: []string{"/usr/local/bin/clusterd -logLevel=0"}},
	{name: "shell enabled", args: []string{"/usr/local/bin/clusterd --enable-healthz=true --healthz-address=11253"}, port: 11253},
	{name: "current initialization", args: []string{"useradd hwMindX; groupadd hwMindX; usermod -aG hwMindX hwMindX; chown hwMindX:hwMindX /var/log/mindx-dl/clusterd; chmod 750 /var/log/mindx-dl/clusterd; /usr/local/bin/clusterd -logFile=/var/log/mindx-dl/clusterd/clusterd.log -logLevel=0 --enable-healthz=true --healthz-address=11253"}, port: 11253},
	{name: "quoted exec and comments", args: []string{"# comment\n umask 027; exec '/usr/local/bin/clusterd' -logFile='/tmp/a b.log' --enable-healthz --healthz-address \"12000\"; # done"}, port: 12000},
	{name: "escaped space", args: []string{`clusterd -logFile=/tmp/a\ b.log --enable-healthz --healthz-address=11253`}, port: 11253},
	{name: "literal quoted metacharacters", args: []string{"clusterd -logFile='/tmp/$name;#literal' --enable-healthz --healthz-address=11253"}, port: 11253},
	{name: "line continuation", args: []string{"clusterd --enable-healthz=true \\\n --healthz-address=12000"}, port: 12000},
	{name: "sh command", command: []string{"/bin/sh", "-c"}, args: []string{"clusterd --enable-healthz --healthz-address=11253"}, port: 11253},
	{name: "script in command", command: []string{"/bin/bash", "-c", "clusterd --enable-healthz --healthz-address=12001"}, port: 12001},
	{name: "direct argv", command: []string{"/usr/local/bin/clusterd"}, args: []string{"-logFile", "/tmp/a b.log", "-enable-healthz", "-healthz-address", "12000"}, port: 12000},
	{name: "flags in command", command: []string{"clusterd", "--enable-healthz=true"}, args: []string{"--healthz-address=11253"}, port: 11253},
	{name: "disabled", args: []string{"clusterd --enable-healthz=false --healthz-address=11253"}},
	{name: "disabled ignores unused port", args: []string{"clusterd --enable-healthz=false --healthz-address=unused"}},
	{name: "port alone", args: []string{"clusterd --healthz-address=11253"}},
	{name: "last boolean wins", args: []string{"clusterd --enable-healthz=true --healthz-address=11253 --enable-healthz=false"}},
	{name: "last port wins", args: []string{"clusterd --enable-healthz=false --enable-healthz=true --healthz-address=11253 --healthz-address=12000"}, port: 12000},
	{name: "minimum supported port", args: []string{"clusterd --enable-healthz --healthz-address=1025"}, port: 1025},
	{name: "maximum port", args: []string{"clusterd --enable-healthz --healthz-address=65535"}, port: 65535},
	{name: "missing port", args: []string{"clusterd --enable-healthz"}, err: "healthz-address"},
	{name: "missing port value", args: []string{"clusterd --enable-healthz --healthz-address"}, err: "healthz-address"},
	{name: "empty port", args: []string{"clusterd --enable-healthz --healthz-address=''"}, err: "healthz-address"},
	{name: "non numeric port", args: []string{"clusterd --enable-healthz --healthz-address=abc"}, err: "healthz-address"},
	{name: "listen address is not port", args: []string{"clusterd --enable-healthz --healthz-address=0.0.0.0:11253"}, err: "healthz-address"},
	{name: "reserved port", args: []string{"clusterd --enable-healthz --healthz-address=1024"}, err: "healthz-address"},
	{name: "overflow port", args: []string{"clusterd --enable-healthz --healthz-address=65536"}, err: "healthz-address"},
	{name: "invalid boolean", args: []string{"clusterd --enable-healthz=maybe --healthz-address=11253"}, err: "enable-healthz"},
	{name: "invalid boolean before unknown flag", args: []string{"clusterd --enable-healthz=maybe --healthz-address=11253 --future-flag"}, err: "enable-healthz"},
	{name: "tls", args: []string{"clusterd --enable-healthz --healthz-address=11253 --tls-cert-file=/cert.crt --tls-private-key-file=/cert.key"}, port: 11253, https: true},
	{name: "incomplete tls", args: []string{"clusterd --enable-healthz --healthz-address=11253 --tls-cert-file=/cert.crt"}, err: "tls"},
	{name: "tls cleared", args: []string{"clusterd --enable-healthz --healthz-address=11253 --tls-cert-file=/cert.crt --tls-cert-file=''"}, port: 11253},
	{name: "version exits", args: []string{"clusterd --version --enable-healthz --healthz-address=11253"}},
	{name: "end of flags", args: []string{"clusterd -- --enable-healthz --healthz-address=11253"}},
	{name: "positional stops flags", args: []string{"clusterd input --enable-healthz --healthz-address=11253"}},
	{name: "bare boolean does not consume next word", args: []string{"clusterd --enable-healthz false --healthz-address=11253"}, err: "healthz-address"},
	{name: "flag text inside log path", args: []string{"clusterd -logFile='--enable-healthz=true --healthz-address=11253'"}},
	{name: "echo is not execution", args: []string{"echo 'clusterd --enable-healthz --healthz-address=11253'"}},
	{name: "comment is not execution", args: []string{"clusterd # --enable-healthz --healthz-address=11253"}},
	{name: "external script", args: []string{"/start.sh --enable-healthz --healthz-address=11253"}},
	{name: "conditional", args: []string{"if false; then clusterd --enable-healthz --healthz-address=11253; fi"}},
	{name: "and list", args: []string{"true && clusterd --enable-healthz --healthz-address=11253"}},
	{name: "pipeline", args: []string{"clusterd --enable-healthz --healthz-address=11253 | cat"}},
	{name: "background", args: []string{"clusterd --enable-healthz --healthz-address=11253 & wait"}},
	{name: "dynamic variable", args: []string{"clusterd --enable-healthz --healthz-address=$PORT"}},
	{name: "quoted dynamic variable", args: []string{`clusterd --enable-healthz --healthz-address="${PORT}"`}},
	{name: "command substitution", args: []string{"clusterd --enable-healthz --healthz-address=$(cat /port)"}},
	{name: "kubelet substitution", command: []string{"clusterd"}, args: []string{"--enable-healthz", "--healthz-address=$(PORT)"}},
	{name: "backticks", args: []string{"clusterd --enable-healthz --healthz-address=`cat /port`"}},
	{name: "unclosed quote", args: []string{"clusterd --enable-healthz --healthz-address='11253"}},
	{name: "trailing escape", args: []string{"clusterd --enable-healthz --healthz-address=11253\\"}},
	{name: "wildcard", args: []string{"clusterd --enable-healthz --healthz-address=1125?"}},
	{name: "exit before command", args: []string{"exit 0; clusterd --enable-healthz --healthz-address=11253"}},
	{name: "command after clusterd", args: []string{"clusterd --enable-healthz --healthz-address=11253; sleep 100"}},
	{name: "unknown flag", args: []string{"clusterd --future-flag --enable-healthz --healthz-address=11253"}},
	{name: "extra shell arguments", command: []string{"bash", "-c"}, args: []string{"clusterd --enable-healthz --healthz-address=11253", "arg0"}},
	{name: "shell without script", command: []string{"bash", "-c", "--"}},
	{name: "empty script", args: []string{""}},
	{name: "exec without executable", args: []string{"exec"}},
	{name: "consecutive semicolons", args: []string{"umask 027;; clusterd --enable-healthz --healthz-address=11253"}},
	{name: "empty command", command: []string{}},
}

// Exercise the registered transformer after CR command/args substitution, rather
// than testing only a parser that could be disconnected from reconciliation.
func TestClusterdProbeCommands(t *testing.T) {
	for _, tt := range clusterdProbeCommandCases {
		t.Run(tt.name, func(t *testing.T) {
			p := getInstance()
			p.Spec.ClusterD.Image = v1.ImageSpec{Repository: "clusterd", Tag: "v26.1.1"}
			p.Spec.ClusterD.Command = v1.CommandSpec{Command: tt.command, Args: tt.args}
			_, r := getManagerAndReconciler(nil, p)
			dep := clusterdProbeTestTemplate(t)
			if tt.command != nil {
				dep.Spec.Template.Spec.Containers[0].Command = []string{placeholderValue}
			}
			// Start with a stale probe to also check explicit removal on every opt-out.
			dep.Spec.Template.Spec.Containers[0].LivenessProbe = &corev1.Probe{InitialDelaySeconds: 99}
			err := newDeploymentReconcileHooks(dep).transform(context.Background(), &r, dep)
			if tt.err != "" {
				require.ErrorContains(t, err, tt.err)
				return
			}
			require.NoError(t, err)
			probe := dep.Spec.Template.Spec.Containers[0].LivenessProbe
			if tt.port == 0 {
				require.Nil(t, probe)
				return
			}
			require.NotNil(t, probe)
			scheme := corev1.URISchemeHTTP
			if tt.https {
				scheme = corev1.URISchemeHTTPS
			}
			require.Equal(t, &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
					Path: "/", Port: intstr.FromInt32(tt.port), Scheme: scheme,
				}},
				InitialDelaySeconds: 10, PeriodSeconds: 10, TimeoutSeconds: 3, FailureThreshold: 3,
			}, probe)
		})
	}
}

func TestClusterdProbeInvalidConfigurationPreservesDeployment(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	p.Spec.ClusterD.Image = v1.ImageSpec{Repository: "clusterd", Tag: "v26.1.1"}
	p.Spec.ClusterD.Command.Args = []string{"clusterd --enable-healthz --healthz-address=11253"}
	_, r := getManagerAndReconciler(nil, p)
	base := clusterdProbeTestTemplate(t)
	res := resourceCreators[gvkDeployment](base)
	_, err := res.clone().reconcile(ctx, &r, true)
	require.NoError(t, err)
	before := &appsv1.Deployment{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), before))

	after := &appsv1.Deployment{}
	for _, tt := range []struct{ script, errorText string }{
		{"clusterd --enable-healthz --healthz-address=invalid", "healthz-address"},
		{"clusterd --enable-healthz=maybe --healthz-address=11253 --future-flag", "enable-healthz"},
	} {
		r.instance.Spec.ClusterD.Command.Args = []string{tt.script}
		_, err = res.clone().reconcile(ctx, &r, true)
		require.ErrorContains(t, err, tt.errorText)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), after))
		require.Equal(t, before.Spec, after.Spec)
	}

	r.instance.Spec.ClusterD.Command.Args = []string{"clusterd --enable-healthz --healthz-address=12000"}
	_, err = res.clone().reconcile(ctx, &r, true)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), after))
	require.EqualValues(t, 12000, after.Spec.Template.Spec.Containers[0].LivenessProbe.HTTPGet.Port.IntVal)
}

func TestClusterdProbeMissingContainer(t *testing.T) {
	dep := clusterdProbeTestTemplate(t)
	dep.Spec.Template.Spec.Containers[0].Name = "other"
	err := workloadHealthProbeTransformer[appsv1.Deployment](context.Background(), nil, dep)
	require.ErrorContains(t, err, "clusterd container not found")
}

func TestClusterdProbeReconcileTransitions(t *testing.T) {
	ctx := context.Background()
	p := getInstance()
	p.Spec.ClusterD.Image = v1.ImageSpec{Repository: "clusterd", Tag: "v26.1.1"}
	_, r := getManagerAndReconciler(nil, p)
	base := clusterdProbeTestTemplate(t)
	sidecarProbe := &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}}}
	base.Spec.Template.Spec.Containers = append(base.Spec.Template.Spec.Containers, corev1.Container{
		Name: "sidecar", Image: "sidecar", LivenessProbe: sidecarProbe.DeepCopy(),
	})
	base.Spec.Template.Spec.InitContainers = []corev1.Container{{Name: "setup", Image: "setup", Command: []string{"true"}}}
	res := resourceCreators[gvkDeployment](base)
	for _, tt := range []struct {
		name, script, tag string
		port              int32
	}{
		{"old", "clusterd -logLevel=0", "v7.3.0", 0},
		{"enable", "clusterd --enable-healthz --healthz-address=11253", "v26.1.1", 11253},
		{"change port", "clusterd --enable-healthz --healthz-address=12000", "v26.1.1", 12000},
		{"disable", "clusterd --enable-healthz=false --healthz-address=12000", "v26.1.1", 0},
		{"re-enable", "clusterd --enable-healthz --healthz-address=11253", "v26.1.1", 11253},
		{"downgrade", "clusterd -logLevel=0", "v7.3.0", 0},
		{"enable again", "clusterd --enable-healthz --healthz-address=11253", "v26.1.1", 11253},
		{"external script", "/start.sh", "custom", 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r.instance.Spec.ClusterD.Command.Args = []string{tt.script}
			r.instance.Spec.ClusterD.Image.Tag = tt.tag
			_, err := res.clone().reconcile(ctx, &r, true)
			require.NoError(t, err)
			actual := &appsv1.Deployment{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), actual))
			container := actual.Spec.Template.Spec.Containers[0]
			require.Equal(t, []string{tt.script}, container.Args)
			require.Equal(t, base.Spec.Template.Spec.Containers[0].SecurityContext, container.SecurityContext)
			require.Equal(t, base.Spec.Template.Spec.Containers[0].VolumeMounts, container.VolumeMounts)
			require.Equal(t, sidecarProbe, actual.Spec.Template.Spec.Containers[1].LivenessProbe)
			require.Equal(t, base.Spec.Template.Spec.InitContainers, actual.Spec.Template.Spec.InitContainers)
			if tt.port == 0 {
				require.Nil(t, container.LivenessProbe)
			} else {
				require.NotNil(t, container.LivenessProbe)
				require.Equal(t, tt.port, container.LivenessProbe.HTTPGet.Port.IntVal)
			}
			// Repeated reconciliation must not change the persisted spec.
			_, err = res.clone().reconcile(ctx, &r, true)
			require.NoError(t, err)
			repeated := &appsv1.Deployment{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(base), repeated))
			require.Equal(t, actual.Spec, repeated.Spec)
		})
	}
}
