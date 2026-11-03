/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgradepath

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	upv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestServiceFindPathReturnsShortestPath(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
		{From: "v1.1.0", To: "v1.2.0"},
		{From: "v1.0.0", To: "v1.2.0"},
	}

	require.NoError(t, service.Load(paths, nil, "digest-a"))

	path, err := service.FindPath("v1.0.0", "v1.2.0")
	require.NoError(t, err)
	require.Len(t, path, 1)
	assert.Equal(t, "v1.2.0", path[0].To)
	assert.Equal(t, "digest-a", service.Digest())
	assert.Equal(t, 3, service.PathCount())
	assert.Equal(t, []string{"v1.0.0", "v1.1.0", "v1.2.0"}, service.AllVersions())
}

func TestServiceFindPathSkipsBlockedEdges(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.2.0", Blocked: true},
		{From: "v1.0.0", To: "v1.1.0"},
		{From: "v1.1.0", To: "v1.2.0"},
	}

	require.NoError(t, service.Load(paths, nil, "digest-a"))

	path, err := service.FindPath("v1.0.0", "v1.2.0")
	require.NoError(t, err)
	require.Len(t, path, 2)
	assert.Equal(t, "v1.1.0", path[0].To)
	assert.Equal(t, "v1.2.0", path[1].To)
}

func TestServiceFindPathReturnsNoPathWhenOnlyPathBlocked(t *testing.T) {
	service := NewService()
	require.NoError(t, service.Load([]upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0", Blocked: true},
	}, nil, "digest-a"))

	_, err := service.FindPath("v1.0.0", "v1.1.0")
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrNoPath))
}

func TestServiceLoadRejectsCycle(t *testing.T) {
	service := NewService()
	err := service.Load([]upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
		{From: "v1.1.0", To: "v1.2.0"},
		{From: "v1.2.0", To: "v1.0.0"},
	}, nil, "digest-a")

	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrCycleDetected))
	assert.Equal(t, 0, service.PathCount())
}

func TestServiceGetInstallableVersions(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
		{From: "v1.1.0", To: "v1.2.0"},
	}
	versions := []upv1alpha1.VersionEntry{
		{Version: "v1.0.0", Installable: true, Deprecated: true},
		{Version: "v1.1.0", Installable: true, Deprecated: false},
		{Version: "v1.2.0", Installable: true, Deprecated: false},
	}

	require.NoError(t, service.Load(paths, versions, "digest-a"))

	result := service.GetInstallableVersions()
	assert.Equal(t, []string{"v1.1.0", "v1.2.0"}, result)
}

func TestServiceGetUpgradeableVersions(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
		{From: "v1.0.0", To: "v1.2.0", Blocked: true},
		{From: "v1.1.0", To: "v1.2.0"},
	}

	require.NoError(t, service.Load(paths, nil, "digest-a"))

	result := service.GetUpgradeableVersions("v1.0.0")
	assert.Equal(t, []string{"v1.1.0", "v1.2.0"}, result)

	result2 := service.GetUpgradeableVersions("v1.1.0")
	assert.Equal(t, []string{"v1.2.0"}, result2)
}

func TestServiceGetUpgradeableVersionsNotFound(t *testing.T) {
	service := NewService()
	require.NoError(t, service.Load([]upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
	}, nil, "digest-a"))

	result := service.GetUpgradeableVersions("v9.0.0")
	assert.Empty(t, result)
}

func TestServiceGetUpgradeableVersionsBlockedBlocksReachability(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0", Blocked: true},
		{From: "v1.1.0", To: "v1.2.0"},
	}

	require.NoError(t, service.Load(paths, nil, "digest-a"))

	result := service.GetUpgradeableVersions("v1.0.0")
	assert.Empty(t, result)
}

func TestServiceVersionEntryMergedWithGraphNodes(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
	}
	versions := []upv1alpha1.VersionEntry{
		{Version: "v1.0.0", Installable: true},
	}

	require.NoError(t, service.Load(paths, versions, "digest-a"))

	installable := service.GetInstallableVersions()
	assert.Contains(t, installable, "v1.0.0")

	all := service.AllVersions()
	assert.Contains(t, all, "v1.1.0")
}

func TestValidateRulesRejectsInvalidRules(t *testing.T) {
	tests := []struct {
		name  string
		paths []upv1alpha1.UpgradePathRule
	}{
		{
			name:  "self loop",
			paths: []upv1alpha1.UpgradePathRule{{From: "v1.0.0", To: "v1.0.0"}},
		},
		{
			name: "duplicate edge",
			paths: []upv1alpha1.UpgradePathRule{
				{From: "v1.0.0", To: "v1.1.0"},
				{From: "v1.0.0", To: "v1.1.0"},
			},
		},
		{
			name: "empty precheck name",
			paths: []upv1alpha1.UpgradePathRule{{
				From:     "v1.0.0",
				To:       "v1.1.0",
				PreCheck: []upv1alpha1.CheckStep{{Required: true}},
			}},
		},
		{
			name: "empty postcheck name",
			paths: []upv1alpha1.UpgradePathRule{{
				From:      "v1.0.0",
				To:        "v1.1.0",
				PostCheck: []upv1alpha1.CheckStep{{Required: true}},
			}},
		},
		{
			name:  "empty from",
			paths: []upv1alpha1.UpgradePathRule{{From: "", To: "v1.1.0"}},
		},
		{
			name:  "empty to",
			paths: []upv1alpha1.UpgradePathRule{{From: "v1.0.0", To: ""}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Error(t, ValidateRules(tt.paths))
		})
	}
}

func TestValidateRulesAcceptsNonSemverVersions(t *testing.T) {
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v26.07-rc.1", To: "v26.07"},
		{From: "latest", To: "v1.1.0"},
	}
	assert.NoError(t, ValidateRules(paths))
}

func TestServiceClear(t *testing.T) {
	service := NewService()
	require.NoError(t, service.Load([]upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
	}, nil, "digest-a"))

	assert.Equal(t, 1, service.PathCount())
	service.Clear()
	assert.Equal(t, 0, service.PathCount())
	assert.Equal(t, "", service.Digest())
}

func TestCompareVersionFallsBackToLexical(t *testing.T) {
	assert.Equal(t, -1, compareVersion("v1.0.0", "v1.1.0"))
	assert.Equal(t, 1, compareVersion("z", "a"))
	assert.Equal(t, 0, compareVersion("same", "same"))
}

func TestServiceConcurrentAccess(t *testing.T) {
	service := NewService()
	paths := []upv1alpha1.UpgradePathRule{
		{From: "v1.0.0", To: "v1.1.0"},
		{From: "v1.1.0", To: "v1.2.0"},
	}
	versions := []upv1alpha1.VersionEntry{
		{Version: "v1.0.0", Installable: true},
		{Version: "v1.1.0", Installable: true},
		{Version: "v1.2.0", Installable: true},
	}

	require.NoError(t, service.Load(paths, versions, "digest-a"))

	done := make(chan bool, 6)

	go func() {
		for i := 0; i < 100; i++ {
			service.FindPath("v1.0.0", "v1.2.0")
		}
		done <- true
	}()
	go func() {
		for i := 0; i < 100; i++ {
			service.AllVersions()
		}
		done <- true
	}()
	go func() {
		for i := 0; i < 100; i++ {
			service.HasVersion("v1.0.0")
		}
		done <- true
	}()
	go func() {
		for i := 0; i < 100; i++ {
			service.GetInstallableVersions()
		}
		done <- true
	}()
	go func() {
		for i := 0; i < 100; i++ {
			service.GetUpgradeableVersions("v1.0.0")
		}
		done <- true
	}()
	go func() {
		for i := 0; i < 100; i++ {
			service.PathCount()
		}
		done <- true
	}()

	for i := 0; i < 6; i++ {
		<-done
	}
}
