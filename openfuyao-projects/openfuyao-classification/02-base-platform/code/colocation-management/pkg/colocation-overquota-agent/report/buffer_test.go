/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
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

package report

import (
	"testing"
	"time"

	"gotest.tools/v3/assert"
	"k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/metric"
)

func TestBufferNotFull(t *testing.T) {
	bufCap := 10
	buf := NewReportBuffer(bufCap)

	for i := 0; i < 4; i++ {
		buf.Put(metric.NodeUsageReport{
			Timestamp:  v1.NewTime(time.Unix(int64(1+i), 0)),
			Window:     v1.Duration{Duration: 1 * time.Second},
			Pods:       nil,
			Containers: nil,
		})
	}

	allElem, err := buf.PeekAll(v1.Time{})
	assert.NilError(t, err)
	t.Log(allElem)
	assert.Equal(t, 4, buf.CurSize())
	assert.Equal(t, len(allElem), buf.CurSize())
	for i := 0; i < len(allElem); i++ {
		assert.Equal(t, allElem[i].Timestamp, v1.NewTime(time.Unix(int64(1+buf.tail+i), 0)))
	}
}

func TestBufferOverride(t *testing.T) {
	bufCap := 10
	buf := NewReportBuffer(bufCap)

	for i := 0; i < 11; i++ {
		buf.Put(metric.NodeUsageReport{
			Timestamp:  v1.NewTime(time.Unix(int64(1+i), 0)),
			Window:     v1.Duration{Duration: 1 * time.Second},
			Pods:       nil,
			Containers: nil,
		})
	}

	t.Logf("head=%v, tail=%v, size=%v, cap=%v\n", buf.head, buf.tail, buf.size, buf.capacity)
	assert.Equal(t, buf.head, 1)
	assert.Equal(t, buf.tail, 1)
	assert.Equal(t, buf.size, 10)
	assert.Equal(t, buf.capacity, 10)
	printAll(t, buf)

	e1, err := buf.Peek()
	assert.NilError(t, err)
	t.Log(e1)
	assert.Equal(t, e1.Timestamp, v1.NewTime(time.Unix(int64(1+1), 0)))

	allElem, err := buf.PeekAll(v1.Time{})
	assert.NilError(t, err)
	t.Log(allElem)
	assert.Equal(t, len(allElem), 10)
	for i := 0; i < len(allElem); i++ {
		assert.Equal(t, allElem[i].Timestamp, v1.NewTime(time.Unix(int64(1+buf.tail+i), 0)))
	}
}

func printAll(t *testing.T, buf *Buffer) {
	for i, e := range buf.buf {
		t.Logf("i=%v, e=%v\n", i, e)
	}
}

// TestBufferPeekAllWithFilter exercises the non-zero lastUsageTimestampCollected
// branch of PeekAll, which filters out reports at-or-before that timestamp.
func TestBufferPeekAllWithFilter(t *testing.T) {
	buf := NewReportBuffer(5)

	// Put reports at t=1,2,3,4
	for i := 1; i <= 4; i++ {
		buf.Put(metric.NodeUsageReport{
			Timestamp: v1.NewTime(time.Unix(int64(i), 0)),
		})
	}

	// Filter: only keep reports strictly after t=2
	filter := v1.NewTime(time.Unix(2, 0))
	all, err := buf.PeekAll(filter)
	assert.NilError(t, err)
	assert.Equal(t, 2, len(all)) // t=3 and t=4 remain
	for _, r := range all {
		assert.Assert(t, r.Timestamp.Time.After(filter.Time))
	}
}

// TestBufferPeekAllFilterExcludesAll checks that a filter newer than every
// report yields an empty (but non-error) slice.
func TestBufferPeekAllFilterExcludesAll(t *testing.T) {
	buf := NewReportBuffer(3)
	buf.Put(metric.NodeUsageReport{Timestamp: v1.NewTime(time.Unix(1, 0))})

	filter := v1.NewTime(time.Unix(10, 0))
	all, err := buf.PeekAll(filter)
	assert.NilError(t, err)
	assert.Equal(t, 0, len(all))
}
