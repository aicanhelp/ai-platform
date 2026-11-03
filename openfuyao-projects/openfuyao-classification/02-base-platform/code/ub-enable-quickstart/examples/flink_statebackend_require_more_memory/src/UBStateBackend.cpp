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

#include "UBStateBackend.h"

#include <cstring>
#include <thread>
#include <vector>

const size_t ALLOCATION_SIZE = 1ULL << 30;   // 单次申请内存大小：1GB

UBStateBackend::UBStateBackend(MemoryManager* mm) : memoryManager(mm) {}

// 申请内存
void* UBStateBackend::requestMemory(size_t size) {
    return memoryManager->allocateMemory(size);
}

// 释放内存
void UBStateBackend::releaseMemory() {
    memoryManager->freeMemory();
}

// 打印内存使用情况
void UBStateBackend::showMemeoryUsage() {
    memoryManager->printMemoryUsage();
}

// 模拟内存使用
void mockMemoryUsed(void* ptr, size_t size, size_t numThreads) {
    size_t                   chunkSize = size / numThreads;
    std::vector<std::thread> threads;

    for (size_t i = 0; i < numThreads; i++) {
        void* chunkPtr = static_cast<uint8_t*>(ptr) + i * chunkSize;
        threads.emplace_back([chunkPtr, chunkSize]() { std::memset(chunkPtr, 0, chunkSize); });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// 保存状态
bool UBStateBackend::saveState(const std::string& key, const std::string& value) {
    // 分配内存存储状态
    void* ptr = requestMemory(ALLOCATION_SIZE);
    if (ptr == nullptr) {
        printf("Failed to alloc for memory. Exit 1.\n");
        return false;
    }
    mockMemoryUsed(ptr, ALLOCATION_SIZE, 32);
    printf("saveState success.\n");
    return true;
}
