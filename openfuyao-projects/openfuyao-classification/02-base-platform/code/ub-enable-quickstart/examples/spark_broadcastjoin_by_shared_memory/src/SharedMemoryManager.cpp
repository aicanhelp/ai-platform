/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "SharedMemoryManager.h"
#include "utils.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdlib.h>

SharedMemoryManager::SharedMemoryManager(const std::string& shmName, size_t shmSize, size_t shmChunkSize)
    : shmName(shmName), shmSize(shmSize), shmChunkSize(shmChunkSize), shmCreated(false) {}

SharedMemoryManager::~SharedMemoryManager() {
    // 析构时自动清理：删除共享内存+解除所有映射
    std::lock_guard<std::mutex> lock(shmMutex);
    for (auto& [name, addr] : shmAddrMap) {
        if (addr != nullptr) {
            ubsmem_shmem_unmap(addr, shmSize);
        }
    }
    shmAddrMap.clear();
    // 防止死锁，调用不加锁版本删除资源
    DeleteShmUnsafe();
}

std::string SharedMemoryManager::CreateShm() {
    std::lock_guard<std::mutex> lock(shmMutex);
    if (shmCreated) {
        return shmName;
    }

    int ret = ubsmem_shmem_allocate("default", shmName.c_str(), shmSize, 0600, UBSM_FLAG_CACHE);
    if (ret == 0) {
        shmCreated = true;
        return shmName;
    }

    std::cerr << "[SharedMemoryManager] 创建失败, 请检查系统资源" << std::endl;
    return "";
}

void* SharedMemoryManager::MapShm() {
    // 检查是否已映射
    std::lock_guard<std::mutex> lock(shmMutex);
    auto                        it = shmAddrMap.find(shmName);
    if (it != shmAddrMap.end()) {
        return it->second;
    }
	
    // 执行映射
    void* shmAddr = nullptr;
    int   ret = ubsmem_shmem_map(shmAddr, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmName.c_str(), 0, &shmAddr);
    if (!shmAddr) {
        std::cerr << "[SharedMemoryManager] 映射失败, errno=" << errno << std::endl;
        return nullptr;
    }

    // 记录映射地址
    std::lock_guard<std::mutex> lock(shmMutex);
    shmAddrMap[shmName] = shmAddr;
    return shmAddr;
}

void SharedMemoryManager::DeleteShmUnsafe() {
    if (!shmCreated) {
        return;
    }

    char shmNameBuf[128] = {0};
    strncpy(shmNameBuf, shmName.c_str(), sizeof(shmNameBuf) - 1);
    int ret = ubsmem_shmem_deallocate(shmNameBuf);
    if (ret == 0) {
        shmCreated = false;
    } else {
        std::cerr << "[SharedMemoryManager] 删除失败, ret=" << ret << std::endl;
    }
}

void SharedMemoryManager::DeleteShm() {
    std::lock_guard<std::mutex> lock(shmMutex);
    DeleteShmUnsafe();
}

void SharedMemoryManager::UnmapShm(void* shmAddr) {
    if (!shmAddr) {
        std::cerr << "[SharedMemoryManager] 解除映射失败: 地址为空" << std::endl;
        return;
    }

    // 执行解除映射
    int ret = ubsmem_shmem_unmap(shmAddr, shmSize);
    if (ret != 0) {
        std::cerr << "[SharedMemoryManager] 解除映射失败: ret=" << ret << std::endl;
        return;
    }

    // 从映射表删除
    std::lock_guard<std::mutex> lock(shmMutex);
    for (auto it = shmAddrMap.begin(); it != shmAddrMap.end(); ++it) {
        if (it->second == shmAddr) {
            shmAddrMap.erase(it);
            break;
        }
    }
}

bool SharedMemoryManager::WriteShm(const std::string& data) {
    // 1. 前置检查：共享内存是否创建
    if (CreateShm().empty()) {
        std::cerr << "[SharedMemoryManager] 写入失败: 共享内存未创建成功" << std::endl;
        return false;
    }

    // 2. 映射共享内存
    void* shmAddr = MapShm();
    if (!shmAddr) {
        std::cerr << "[SharedMemoryManager] 写入失败: 映射失败" << std::endl;
        DeleteShm();
        return false;
    }

    // 3. 切换为写权限
    int ret = ubsmem_shmem_set_ownership(shmName.c_str(), shmAddr, shmSize, PROT_READ | PROT_WRITE);
    if (ret != 0) {
        std::cerr << "[SharedMemoryManager] 写入失败: 切换写权限, ret=" << ret << std::endl;
        UnmapShm(shmAddr);
        return false;
    }

    // 4. 写入数据
    size_t   writeSize = std::min(data.size(), shmSize - 1);
    uint64_t startTime = GetTimeStampMs();
    memcpy(shmAddr, data.c_str(), writeSize);
    uint64_t elapsed = GetTimeStampMs() - startTime;

    // 5. 切换回读权限
    ret = ubsmem_shmem_set_ownership(shmName.c_str(), shmAddr, shmSize, PROT_NONE);
    if (ret != 0) {
        std::cerr << "[SharedMemoryManager] 写入警告: 切换读权限, ret=" << ret << std::endl;
    }

    // 6. 解除映射
    UnmapShm(shmAddr);
    return true;
}

std::string SharedMemoryManager::ReadShm() {
    // 1. 映射共享内存
    void* shmAddr = MapShm();
    if (!shmAddr) {
        std::cerr << "[SharedMemoryManager] 读取失败: 映射失败" << std::endl;
        DeleteShm();
        return "";
    }

    // 2. 切换为读权限
    int ret = ubsmem_shmem_set_ownership(shmName.c_str(), shmAddr, shmSize, PROT_READ);
    if (ret != 0) {
        std::cerr << "[SharedMemoryManager] 读取失败: 切换读权限, ret=" << ret << std::endl;
        UnmapShm(shmAddr);
        return "";
    }

    // 3. 读取数据
    uint64_t    startTime = GetTimeStampMs();
    std::string readData(reinterpret_cast<char*>(shmAddr));
    uint64_t    elapsed = GetTimeStampMs() - startTime;

    // 4. 切换为无访问权限
    ret = ubsmem_shmem_set_ownership(shmName.c_str(), shmAddr, shmSize, PROT_NONE);
    if (ret != 0) {
        std::cerr << "[SharedMemoryManager] 读取警告: 切换无权限, ret=" << ret << std::endl;
    }

    // 5.解除映射
    UnmapShm(shmAddr);
    return readData;
}

void SharedMemoryManager::ForceSetCreated(bool flag) {
    std::lock_guard<std::mutex> lock(shmMutex);
    shmCreated = flag;
}
