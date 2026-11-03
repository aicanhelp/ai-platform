/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package helm

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/release"
	time2 "helm.sh/helm/v3/pkg/time"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"marketplace-service/pkg/models/helm"
)

func TestGetMinInt(t *testing.T) {
	type args struct {
		a int
		b int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "TestGetMinInt",
			args: args{
				a: 3,
				b: 2,
			},
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetMinInt(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("GetMinInt() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestConvertToRelease 测试Release转换
func TestConvertToRelease(t *testing.T) {
	// 显式使用helm的time类型，避免与标准库冲突
	helmTime := metav1.NewTime(time.Now())
	rel := &release.Release{
		Name:      "test-rel",
		Namespace: "default",
		Version:   1,
		Info: &release.Info{
			Status:      release.StatusDeployed,
			Notes:       "test notes",
			Deleted:     time2.Time(helmTime), // 修复time类型不匹配
			Description: "test desc",
		},
	}
	res := ConvertToRelease(nil, rel, false)
	assert.Equal(t, "test-rel", res.Name)
	assert.Equal(t, "deployed", res.Info.Status)
	assert.Equal(t, 1, res.Version)
}

// TestConvertChart 测试Chart转换
func TestConvertChart(t *testing.T) {
	rel := &release.Release{
		Chart: &chart.Chart{
			Metadata: &chart.Metadata{Name: "test-chart"},
			Files:    []*chart.File{{Name: "README.md", Data: []byte("readme")}},
		},
	}
	info := &helm.Info{}
	chartRes := ConvertChart(rel, info, true)
	assert.Equal(t, "test-chart", chartRes.Metadata.Name)
	assert.Equal(t, "readme", info.Readme)
}

// TestConvertToListInterface 测试切片转换
func TestConvertToListInterface(t *testing.T) {
	const len = 1
	const len3 = 3
	// 正常切片
	slice := []int{1, 2, 3}
	res := ConvertToListInterface(slice)
	assert.Len(t, res, len3)
	assert.Equal(t, len, res[0])

	// 非切片
	nonSlice := "test"
	res2 := ConvertToListInterface(nonSlice)
	assert.Nil(t, res2)
}

// TestConvertToReleaseHistory 测试Release历史转换
func TestConvertToReleaseHistory(t *testing.T) {
	rel := &release.Release{
		Name:      "test-rel",
		Namespace: "default",
		Version:   1,
		Chart: &chart.Chart{Metadata: &chart.Metadata{
			Name:       "test-chart",
			Version:    "1.0.0",
			AppVersion: "v1",
		}},
	}
	history := ConvertToReleaseHistory(rel)
	assert.Equal(t, "test-rel", history.Name)
	assert.Equal(t, "test-chart", history.ChartName)
	assert.Equal(t, "1.0.0", history.ChartVersion)
}

// TestParseQueryInt 测试字符串转整数
func TestParseQueryInt(t *testing.T) {
	const num = 10
	assert.Equal(t, num, ParseQueryInt("10"))
	assert.Equal(t, 0, ParseQueryInt("abc")) // 非法字符串返回0
}
