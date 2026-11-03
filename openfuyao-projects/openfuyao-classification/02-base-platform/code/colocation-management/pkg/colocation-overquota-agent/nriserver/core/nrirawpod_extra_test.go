/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package core

import (
	"testing"

	"github.com/containerd/nri/pkg/api"
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestPrefix(t *testing.T) {
	tests := []struct {
		name   string
		engine ContainerEngineType
		want   string
	}{
		{name: "docker", engine: DOCKER, want: "docker://"},
		{name: "containerd", engine: CONTAINERD, want: "containerd://"},
		{name: "isulad", engine: ISULAD, want: "iSulad://"},
		{name: "crio", engine: CRIO, want: "cri-o://"},
		{name: "undefined returns empty", engine: UNDEFINED, want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := tt.engine
			assert.Equal(t, tt.want, e.Prefix())
		})
	}
}

func TestGetContainerEnv(t *testing.T) {
	tests := []struct {
		name string
		env  []string
		want map[string]string
	}{
		{name: "empty", env: []string{}, want: map[string]string{}},
		{
			name: "valid key value pairs",
			env:  []string{"KEY1=VALUE1", "KEY2=VALUE2"},
			want: map[string]string{"KEY1": "VALUE1", "KEY2": "VALUE2"},
		},
		{
			name: "value with equals sign",
			env:  []string{"CONN=host=user pass=secret"},
			want: map[string]string{"CONN": "host=user pass=secret"},
		},
		{
			name: "entry without equals is dropped",
			env:  []string{"NOEQUALS", "KEY=VALUE"},
			want: map[string]string{"KEY": "VALUE"},
		},
		{
			name: "empty key is still stored",
			env:  []string{"=EMPTYKEY"},
			want: map[string]string{"": "EMPTYKEY"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetContainerEnv(tt.env)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGetPodQosLevel(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		want        common.QosLevel
	}{
		{name: "nil map", annotations: nil, want: common.QosNone},
		{name: "no qos key", annotations: map[string]string{}, want: common.QosNone},
		{name: "be", annotations: map[string]string{common.QosClassAnnotationKey: "BE"}, want: common.QosBe},
		{name: "ls", annotations: map[string]string{common.QosClassAnnotationKey: "LS"}, want: common.QosLs},
		{name: "hls", annotations: map[string]string{common.QosClassAnnotationKey: "HLS"}, want: common.QosHLS},
		{name: "invalid", annotations: map[string]string{common.QosClassAnnotationKey: "garbage"}, want: common.QosNone},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, GetPodQosLevel(tt.annotations))
		})
	}
}

func TestVerifyPodPriorityAnnotation(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		want        bool
	}{
		{name: "nil", annotations: nil, want: false},
		{name: "has priority", annotations: map[string]string{common.PriorityAnnotationKey: "true"}, want: true},
		{name: "missing priority", annotations: map[string]string{"other": "v"}, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, VerifyPodPriorityAnnotation(tt.annotations))
		})
	}
}

func TestCheckAnnotationAdmit(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		wantAdmit   bool
		wantReason  string
	}{
		{name: "nil admit", annotations: nil, wantAdmit: true},
		{name: "no extender annotation admit", annotations: map[string]string{"a": "b"}, wantAdmit: true},
		{
			name:        "extender without qos denied",
			annotations: map[string]string{common.ExtenderResourceConfigurationAnnotation: "{}"},
			wantAdmit:   false,
			wantReason:  "offline resource pod  must have BE QosLevel annotation",
		},
		{
			name: "extender with be but no priority",
			annotations: map[string]string{
				common.ExtenderResourceConfigurationAnnotation: "{}",
				common.QosClassAnnotationKey:                   string(common.QosBe),
			},
			wantAdmit:  false,
			wantReason: "offline resource pod  must have volcano.sh/preemptable annotation",
		},
		{
			name: "fully admitted",
			annotations: map[string]string{
				common.ExtenderResourceConfigurationAnnotation: "{}",
				common.QosClassAnnotationKey:                   string(common.QosBe),
				common.PriorityAnnotationKey:                   "true",
			},
			wantAdmit: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := CheckAnnotationAdmit(tt.annotations)
			assert.Equal(t, tt.wantAdmit, r.Admit)
			if tt.wantReason != "" {
				assert.Equal(t, tt.wantReason, r.Reason)
			}
		})
	}
}

func TestGetExtendedResourceConfiguration(t *testing.T) {
	t.Run("empty annotations", func(t *testing.T) {
		cfg, err := GetExtendedResourceConfiguration(map[string]string{})
		assert.NoError(t, err)
		assert.NotNil(t, cfg)
	})

	t.Run("nil annotations", func(t *testing.T) {
		cfg, err := GetExtendedResourceConfiguration(nil)
		assert.NoError(t, err)
		assert.NotNil(t, cfg)
	})

	t.Run("valid json", func(t *testing.T) {
		cfg, err := GetExtendedResourceConfiguration(map[string]string{
			common.ExtenderResourceConfigurationAnnotation: `{"containers":{"c1":{}}}`,
		})
		assert.NoError(t, err)
		assert.Contains(t, cfg.Containers, "c1")
	})

	t.Run("invalid json", func(t *testing.T) {
		cfg, err := GetExtendedResourceConfiguration(map[string]string{
			common.ExtenderResourceConfigurationAnnotation: "{not json}",
		})
		assert.Error(t, err)
		assert.Nil(t, cfg)
	})

	t.Run("no extender annotation returns empty config", func(t *testing.T) {
		cfg, err := GetExtendedResourceConfiguration(map[string]string{"other": "v"})
		assert.NoError(t, err)
		assert.NotNil(t, cfg)
		assert.Empty(t, cfg.Containers)
	})
}

func TestGetPodCPUset(t *testing.T) {
	t.Run("empty annotations", func(t *testing.T) {
		s, err := GetPodCPUset(map[string]string{})
		assert.NoError(t, err)
		assert.Equal(t, "", s)
	})

	t.Run("nil annotations", func(t *testing.T) {
		s, err := GetPodCPUset(nil)
		assert.NoError(t, err)
		assert.Equal(t, "", s)
	})

	t.Run("valid numa resource", func(t *testing.T) {
		s, err := GetPodCPUset(map[string]string{
			common.NumaResourceAnnotation: `{"CPUset":"0-3,5"}`,
		})
		assert.NoError(t, err)
		assert.Equal(t, "0-3,5", s)
	})

	t.Run("invalid json returns cpuset with error", func(t *testing.T) {
		_, err := GetPodCPUset(map[string]string{
			common.NumaResourceAnnotation: "{not json}",
		})
		assert.Error(t, err)
	})

	t.Run("no numa annotation returns empty", func(t *testing.T) {
		s, err := GetPodCPUset(map[string]string{"other": "v"})
		assert.NoError(t, err)
		assert.Equal(t, "", s)
	})
}

func TestGetKubeQoSByCgroupParent(t *testing.T) {
	tests := []struct {
		name   string
		parent string
		want   v1.PodQOSClass
	}{
		{name: "besteffort", parent: "kubepods-besteffort.slice/abc", want: v1.PodQOSBestEffort},
		{name: "burstable", parent: "kubepods-burstable.slice/abc", want: v1.PodQOSBurstable},
		{name: "guaranteed", parent: "kubepods-pod.slice/abc", want: v1.PodQOSGuaranteed},
		{name: "empty", parent: "", want: v1.PodQOSGuaranteed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, GetKubeQoSByCgroupParent(tt.parent))
		})
	}
}

func TestSetGetCgroupDriver(t *testing.T) {
	original := GetCgroupDriver()
	defer SetCgroupDriver(original)

	SetCgroupDriver(common.CgroupDriverCgroupfs)
	assert.Equal(t, common.CgroupDriverCgroupfs, GetCgroupDriver())

	SetCgroupDriver(common.CgroupDriverSystemd)
	assert.Equal(t, common.CgroupDriverSystemd, GetCgroupDriver())
}

func TestFixCurrentContainerEngine(t *testing.T) {
	originalEngine := currentContainerEngines
	defer func() { currentContainerEngines = originalEngine }()

	tests := []struct {
		name        string
		input       string
		wantPrefix  string
		wantDefault bool
	}{
		{name: "containerd prefix", input: "cri-containerd", wantPrefix: "cri-containerd"},
		{name: "isulad prefix", input: "isulad", wantPrefix: "isulad"},
		{name: "docker prefix", input: "docker", wantPrefix: "docker"},
		{name: "crio prefix", input: "crio", wantPrefix: "crio"},
		{name: "unknown falls back to default", input: "unknown-engine", wantDefault: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FixCurrentContainerEngine(tt.input)
			if tt.wantDefault {
				assert.Equal(t, supportEnginesPrefixMap[defaultContainerEngines], got)
			} else {
				assert.Equal(t, tt.wantPrefix, got)
			}
		})
	}
}

func TestGetContainerEngine(t *testing.T) {
	originalEngine := currentContainerEngines
	defer func() { currentContainerEngines = originalEngine }()

	t.Run("with annotation sets engine", func(t *testing.T) {
		GetContainerEngine(map[string]string{common.ContainerEngineAnnotation: "docker"})
		assert.Equal(t, DOCKER, currentContainerEngines)
	})

	t.Run("without annotation sets default", func(t *testing.T) {
		GetContainerEngine(map[string]string{})
		assert.Equal(t, defaultContainerEngines, currentContainerEngines)
	})
}

func makeContainer(cgroupsPath, id string) *api.Container {
	return &api.Container{
		Id: id,
		Linux: &api.LinuxContainer{
			CgroupsPath: cgroupsPath,
		},
	}
}

func TestFixContainerCgroupParentDrivers(t *testing.T) {
	originalDriver := cgroupDriver
	originalEngine := currentContainerEngines
	defer func() {
		cgroupDriver = originalDriver
		currentContainerEngines = originalEngine
	}()

	t.Run("cgroupfs with containerd besteffort", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverCgroupfs)
		currentContainerEngines = CONTAINERD
		c := makeContainer("/kubepods/besteffort/poduid/containerid", "containerid")
		parent, err := FixContainerCgroupParent(c)
		assert.NoError(t, err)
		assert.Contains(t, parent, "kubepods/besteffort/")
		assert.Contains(t, parent, "containerid")
	})

	t.Run("cgroupfs with crio guaranteed", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverCgroupfs)
		currentContainerEngines = CRIO
		c := makeContainer("/kubepods/poduid/crio-containerid", "containerid")
		parent, err := FixContainerCgroupParent(c)
		assert.NoError(t, err)
		// crio engine prepends "crio-"
		assert.Contains(t, parent, "crio-containerid")
	})

	t.Run("cgroupfs with docker burstable", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverCgroupfs)
		currentContainerEngines = DOCKER
		c := makeContainer("/kubepods/burstable/poduid/containerid", "containerid")
		parent, err := FixContainerCgroupParent(c)
		assert.NoError(t, err)
		assert.Contains(t, parent, "containerid")
	})

	t.Run("systemd guaranteed containerd", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverSystemd)
		currentContainerEngines = CONTAINERD
		c := makeContainer("kubepods-poduid.slice:cri-containerd:containerid", "containerid")
		parent, err := FixContainerCgroupParent(c)
		assert.NoError(t, err)
		assert.Contains(t, parent, "kubepods.slice")
		assert.Contains(t, parent, "cri-containerd-containerid.scope")
	})

	t.Run("systemd besteffort crio", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverSystemd)
		currentContainerEngines = CRIO
		c := makeContainer("kubepods-besteffort-poduid.slice:crio:containerid", "containerid")
		parent, err := FixContainerCgroupParent(c)
		assert.NoError(t, err)
		assert.Contains(t, parent, "kubepods-besteffort.slice")
	})

	t.Run("isulad cgroupfs unsupported engine returns empty", func(t *testing.T) {
		// ISULAD maps to "isulad" which does not match
		// common.ContainerEngineIsula ("isula"), so the engine is treated as
		// unsupported and the built path is empty.
		SetCgroupDriver(common.CgroupDriverCgroupfs)
		currentContainerEngines = ISULAD
		c := makeContainer("/kubepods/poduid/containerid", "containerid")
		parent, err := FixContainerCgroupParent(c)
		assert.NoError(t, err)
		assert.Equal(t, "", parent)
	})
}

func TestGetQos(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantQos string
	}{
		{name: "guaranteed", path: "kubepods-pod.slice:abc", wantQos: "Guaranteed"},
		{name: "burstable", path: "/kubepods/burstable/poduid/cid", wantQos: "Burstable"},
		{name: "besteffort", path: "/kubepods/besteffort/poduid/cid", wantQos: "BestEffort"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := makeContainer(tt.path, "cid")
			assert.Equal(t, tt.wantQos, getQos(c))
		})
	}
}

func TestGetQosPath(t *testing.T) {
	tests := []struct {
		name      string
		qosClass  string
		wantPath  string
		wantError bool
	}{
		{name: "guaranteed", qosClass: "Guaranteed", wantPath: ""},
		{name: "burstable", qosClass: "Burstable", wantPath: "burstable"},
		{name: "besteffort", qosClass: "BestEffort", wantPath: "besteffort"},
		{name: "unknown error", qosClass: "Unknown", wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := getQosPath(tt.qosClass)
			if tt.wantError {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.wantPath, p)
		})
	}
}

func TestGetPodCgroupDir(t *testing.T) {
	originalDriver := cgroupDriver
	defer func() { cgroupDriver = originalDriver }()

	t.Run("systemd driver splits on colon", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverSystemd)
		c := makeContainer("kubepods-poduid.slice:cri-containerd:cid", "cid")
		assert.Equal(t, "kubepods-poduid.slice", getPodCgroupDir(c))
	})

	t.Run("cgroupfs driver splits on slash", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverCgroupfs)
		c := makeContainer("/kubepods/burstable/poduid/cid", "cid")
		// second from end = poduid
		assert.Equal(t, "poduid", getPodCgroupDir(c))
	})

	t.Run("cgroupfs minimal two-segment path", func(t *testing.T) {
		SetCgroupDriver(common.CgroupDriverCgroupfs)
		// podPathIndexFromEnd == 2, so pathInfo[len-2] requires len >= 2.
		c := makeContainer("poduid/cid", "cid")
		assert.Equal(t, "poduid", getPodCgroupDir(c))
	})
}

func TestIsSupportedContainerEngine(t *testing.T) {
	originalEngine := currentContainerEngines
	defer func() { currentContainerEngines = originalEngine }()

	// DOCKER, CONTAINERD, CRIO are supported. ISULAD maps to "isulad" in
	// containerEngineScopes but common.ContainerEngineIsula is "isula", so
	// ISULAD is effectively unsupported by isSupportedContainerEngine (this
	// is a pre-existing spelling mismatch in the source, not a test bug).
	for _, e := range []ContainerEngineType{DOCKER, CONTAINERD, CRIO} {
		currentContainerEngines = e
		assert.True(t, isSupportedContainerEngine(), "engine %v should be supported", e)
	}

	for _, e := range []ContainerEngineType{UNDEFINED, ISULAD} {
		currentContainerEngines = e
		assert.False(t, isSupportedContainerEngine(), "engine %v should be unsupported", e)
	}
}

func TestGetContainerID(t *testing.T) {
	originalEngine := currentContainerEngines
	defer func() { currentContainerEngines = originalEngine }()

	currentContainerEngines = CRIO
	assert.Equal(t, "crio-cid", getContainerID(&api.Container{Id: "cid"}))

	currentContainerEngines = CONTAINERD
	assert.Equal(t, "cid", getContainerID(&api.Container{Id: "cid"}))
}

func TestBuildCgroupfsPathUnsupportedEngine(t *testing.T) {
	originalEngine := currentContainerEngines
	defer func() { currentContainerEngines = originalEngine }()

	currentContainerEngines = UNDEFINED
	c := makeContainer("/kubepods/burstable/poduid/cid", "cid")
	p, err := buildCgroupfsPath(c, "burstable")
	assert.NoError(t, err)
	assert.Equal(t, "", p)
}

func TestBuildSystemdCgroupPathUnsupportedEngine(t *testing.T) {
	originalEngine := currentContainerEngines
	defer func() { currentContainerEngines = originalEngine }()

	currentContainerEngines = UNDEFINED
	c := makeContainer("kubepods-poduid.slice:abc:cid", "cid")
	p, err := buildSystemdCgroupPath(c, "")
	assert.NoError(t, err)
	assert.Equal(t, "", p)
}
