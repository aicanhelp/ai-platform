/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "LocalMemoryAllocator.h"
#include "RackMemoryAllocator.h"

#include <list>
#include <mutex>

class MemoryManager {
private:
    LocalMemoryAllocator* localAllocator;   // 本地内存分配器
    RackMemoryAllocator*  rackAllocator;    // 远程内存分配器
    size_t                memoryLimit;      // 内存申请限制
    std::mutex            mtx;              // 互斥锁
    size_t                currentUsage;     // 本地内存使用量
    std::list<void*>      localPtrs;        // 本地内存指针列表
    std::list<void*>      remotePtrs;       // 远程内存指针列表

public:
    /**
     * @brief 构造函数
     * @param localAlloc 本地内存分配器
     * @param rackAlloc 远程内存分配器
     * @param limit 内存申请限制
     */
    MemoryManager(LocalMemoryAllocator* localAlloc, RackMemoryAllocator* rackAlloc, size_t limit);

    /**
     * @brief 析构函数
     */
    ~MemoryManager();

    /**
     * @brief 申请远程内存
     * @param size 申请内存大小
     * @return 返回申请内存地址
     */
    void* allocRemotrMemory(size_t size);

    /**
     * @brief 申请远程内存
     * @param size 申请内存大小
     * @return 返回申请内存地址
     */
    void* allocateMemory(size_t size);

    /**
     * @brief 释放内存
     */
    void freeMemory();

    /**
     * @brief 打印当前系统内存使用信息
     */
    void printMemoryUsage();
};

#endif   // MEMORY_MANAGER_H