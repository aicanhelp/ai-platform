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

#include "MemoryManager.h"

#include <iostream>
#include <sys/sysinfo.h>

MemoryManager::MemoryManager(LocalMemoryAllocator* localAlloc, RackMemoryAllocator* rackAlloc, size_t limit)
    : localAllocator(localAlloc), rackAllocator(rackAlloc), memoryLimit(limit), currentUsage(0) {}

MemoryManager::~MemoryManager() {
    // 释放所有本地内存块
    for (auto ptr : localPtrs) {
        localAllocator->free_mem(ptr);
    }
    // 释放所有远程内存块
    for (auto ptr : remotePtrs) {
        rackAllocator->free_mem(ptr);
    }
}

// 申请远端内存
void* MemoryManager::allocRemotrMemory(size_t size) {
    printf("借用远程内存...\n");
    void* remotePtr = rackAllocator->malloc_mem(size);
    if (remotePtr == nullptr) {
        printf("借用远程内存失败!\n");
        return nullptr;
    }
    printf("借用远程内存成功\n");
    remotePtrs.push_back(remotePtr);
    return remotePtr;
}

// 模拟内存分配函数
void* MemoryManager::allocateMemory(size_t size) {
    std::lock_guard<std::mutex> lock(mtx);
    if (currentUsage + size <= memoryLimit) {
        void* localPtr = localAllocator->malloc_mem(size);
        if (localPtr == nullptr) {
            printf("本地内存申请失败!\n");
            return nullptr;
        }
        currentUsage += size;
        printf("本地内存申请成功\n");
        localPtrs.push_back(localPtr);
        return localPtr;
    } else {
        printf("申请内存超过本地阈值\n");
        // 当没有UB时, 内存不足返回空指针
        // return nullptr;
        // 当有UB时, 可以借用远程内存
        return allocRemotrMemory(size);
    }
}

// 释放内存
void MemoryManager::freeMemory() {
    printf("释放内存\n");
    std::lock_guard<std::mutex> lock(mtx);
    for (auto ptr : localPtrs) {
        localAllocator->free_mem(ptr);
    }
    for (auto ptr : remotePtrs) {
        rackAllocator->free_mem(ptr);
    }
    // 清空内存指针, 防止析构时失败
    localPtrs.clear();
    remotePtrs.clear();
}

void MemoryManager::printMemoryUsage() {
    struct sysinfo si;
    sysinfo(&si);
    printf("------------------------\n");
    printf("Total RAM: %lu MB\n", si.totalram / (1024 * 1024));
    printf("Free RAM: %lu MB\n", si.freeram / (1024 * 1024));
    printf("Used RAM: %lu MB\n", (si.totalram - si.freeram) / (1024 * 1024));
    printf("------------------------\n");
}
