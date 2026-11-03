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
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

type healthProbeFlags struct {
	enabled bool
	version bool
	port    string
	cert    string
	key     string
}

func newHealthProbeFlags(profile healthProbeProfile) (*flag.FlagSet, *healthProbeFlags) {
	values := &healthProbeFlags{}
	flags := flag.NewFlagSet(profile.binary, flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.BoolVar(&values.enabled, "enable-healthz", false, "")
	flags.BoolVar(&values.version, "version", false, "")
	if profile.binary == "k8s-rdma-shared-dp" {
		flags.BoolVar(&values.version, "v", false, "")
	}
	flags.StringVar(&values.port, "healthz-address", "", "")
	flags.StringVar(&values.cert, "tls-cert-file", "", "")
	flags.StringVar(&values.key, "tls-private-key-file", "", "")
	for _, name := range strings.Fields("logFile logLevel maxAge maxBackups " + profile.valueFlags) {
		flags.String(name, "", "")
	}
	for _, name := range strings.Fields(profile.boolFlags) {
		flags.Bool(name, false, "")
	}
	return flags, values
}

func resolveHealthProbe(profile healthProbeProfile, command, args []string) (*corev1.Probe, string, error) {
	argv, supported := healthProbeCommandArgs(profile.binary, command, args)
	if !supported {
		return nil, "command is not a supported literal invocation", nil
	}
	flags, values := newHealthProbeFlags(profile)
	prefix, supported := healthProbeKnownFlagPrefix(flags, argv)
	// Use the upstream Go flag semantics, including stop-on-positional and repeated flags.
	if err := flags.Parse(prefix); err != nil {
		return nil, "", fmt.Errorf("invalid component arguments: %w", err)
	}
	if !supported {
		return nil, "unknown flag; cannot infer argument boundaries", nil
	}
	if !values.enabled || values.version {
		return nil, "", nil
	}
	probe, err := values.probe(profile)
	return probe, "", err
}

func (v *healthProbeFlags) probe(profile healthProbeProfile) (*corev1.Probe, error) {
	// The manifests override the healthz library's default 11251; never guess a port.
	port, err := strconv.Atoi(v.port)
	if err != nil || port < 1025 || port > 65535 {
		return nil, fmt.Errorf("enabled healthz requires an explicit healthz-address in [1025, 65535]")
	}
	if (v.cert == "") != (v.key == "") {
		return nil, fmt.Errorf("tls-cert-file and tls-private-key-file must both be set or both be empty")
	}
	scheme := corev1.URISchemeHTTP
	if v.cert != "" {
		scheme = corev1.URISchemeHTTPS
	}
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
			Path: "/", Port: intstr.FromInt32(int32(port)), Scheme: scheme,
		}},
		InitialDelaySeconds: profile.delay,
		PeriodSeconds:       profile.period,
		TimeoutSeconds:      profile.timeout,
		FailureThreshold:    3,
	}, nil
}

// Unknown flags may take values or be booleans. Do not guess their arity or
// accidentally interpret their values as healthz flags. Still validate the known
// prefix so a later unknown flag cannot hide an earlier flag parsing error.
func healthProbeKnownFlagPrefix(flags *flag.FlagSet, args []string) ([]string, bool) {
	skipValue := false
	for i, arg := range args {
		if skipValue {
			skipValue = false
			continue
		}
		if arg == "--" || arg == "-" || !strings.HasPrefix(arg, "-") {
			break
		}
		name, _, hasValue := strings.Cut(strings.TrimPrefix(strings.TrimPrefix(arg, "-"), "-"), "=")
		f := flags.Lookup(name)
		if f == nil {
			return args[:i], false
		}
		boolean, ok := f.Value.(interface{ IsBoolFlag() bool })
		if !hasValue && !(ok && boolean.IsBoolFlag()) {
			skipValue = true
		}
	}
	return args, true
}
