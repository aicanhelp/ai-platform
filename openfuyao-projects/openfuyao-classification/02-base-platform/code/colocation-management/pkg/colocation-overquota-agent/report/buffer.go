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

// Package report
package report

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/metric"
	"openfuyao.com/colocation-management/pkg/common"
)

// Buffer 缓冲区结构体
type Buffer struct {
	sync.Mutex
	buf      []metric.NodeUsageReport
	head     int // 下一个写入的位置
	tail     int // 下一个读取的位置
	size     int // 当前存储的数据数量
	capacity int // 缓冲区的最大容量
}

// NewReportBuffer 创建一个新的缓冲区
func NewReportBuffer(capacity int) *Buffer {
	return &Buffer{
		buf:      make([]metric.NodeUsageReport, capacity),
		size:     0,
		capacity: capacity,
	}
}

// Put 向缓冲区中写入数据，如果缓冲区已满则覆盖旧数据
func (b *Buffer) Put(data metric.NodeUsageReport) {
	b.Lock()
	defer b.Unlock()

	b.buf[b.head] = data
	b.head = (b.head + 1) % b.capacity

	if b.size == b.capacity {
		b.tail = b.head
	} else {
		b.size++
	}
	klog.V(common.MicroscopicDebugLog).Infof("Put: head=%v, tail=%v, size=%v\n", b.head, b.tail, b.size)
}

// Peek 读取缓冲区中的数据而不移除数据
func (b *Buffer) Peek() (metric.NodeUsageReport, error) {
	b.Lock()
	defer b.Unlock()

	if b.size == 0 {
		return metric.NodeUsageReport{}, fmt.Errorf("buffer is empty")
	}

	data := b.buf[b.tail]

	return data, nil
}

// PeekAll 读取并返回缓冲区中的所有数据，但不移除数据
func (b *Buffer) PeekAll(lastUsageTimestampCollected v1.Time) ([]metric.NodeUsageReport, error) {
	b.Lock()
	defer b.Unlock()

	if b.size == 0 {
		return nil, fmt.Errorf("buffer is empty")
	}

	// 创建一个切片来保存所有数据
	allData := make([]metric.NodeUsageReport, 0, b.size)

	tail := b.tail
	for count := b.size; count > 0; count-- {
		if lastUsageTimestampCollected.IsZero() {
			allData = append(allData, b.buf[tail])
		} else if b.buf[tail].Timestamp.Time.After(lastUsageTimestampCollected.Time) {
			allData = append(allData, b.buf[tail])
		}
		tail = (tail + 1) % b.capacity
	}

	return allData, nil
}

// CurSize 返回当前缓冲区中数据的数量
func (b *Buffer) CurSize() int {
	b.Lock()
	defer b.Unlock()

	return b.size
}
