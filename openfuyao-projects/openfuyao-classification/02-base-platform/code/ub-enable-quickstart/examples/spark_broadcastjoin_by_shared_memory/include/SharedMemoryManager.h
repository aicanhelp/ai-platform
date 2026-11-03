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

#ifndef SHARED_MEMORY_MANAGER_H
#define SHARED_MEMORY_MANAGER_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <sys/mman.h>

#include "ubs_mem.h"

class SharedMemoryManager {
public:
    /**
     * @brief 构造函数：初始化共享内存配置
     * @param shmName 共享内存唯一名称
     * @param shmSize 共享内存大小（字节）
     * @param shmChunkSize 读写块大小（字节）
     */
    SharedMemoryManager(const std::string& shmName, size_t shmSize, size_t shmChunkSize);

    /**
     * @brief 析构函数：自动清理共享内存资源
     */
    ~SharedMemoryManager();

    /**
     * @brief 创建共享内存
     * @return 成功返回共享内存名称，失败返回空字符串
     */
    std::string CreateShm();

    /**
     * @brief 映射共享内存到地址空间
     * @return 成功返回映射地址，失败返回nullptr
     */
    void* MapShm();

    /**
     * @brief 删除共享内存（加锁）
     */
    void DeleteShm();

    /**
     * @brief 删除共享内存（不加锁）
     */
    void DeleteShmUnsafe();

    /**
     * @brief 解除共享内存映射
     * @param shmAddr 要解除映射的地址
     */
    void UnmapShm(void* shmAddr);

    /**
     * @brief 写入数据到共享内存
     * @param data 要写入的数据
     * @return 成功返回true，失败返回false
     */
    bool WriteShm(const std::string& data);

    /**
     * @brief 读取数据
     * @return 成功返回读取到的数据，失败返回空字符串
     */
    std::string ReadShm();

    /**
     * @brief 强制设置共享内存创建标记（用于强制清理残留）
     * @param flag 创建状态
     */
    void ForceSetCreated(bool flag);

private:
    std::string                  shmName;        // 共享内存名称
    size_t                       shmSize;        // 共享内存总大小
    size_t                       shmChunkSize;   // 单次读写块大小
    std::map<std::string, void*> shmAddrMap;     // 共享内存地址映射表
    std::mutex                   shmMutex;       // 互斥锁
    bool                         shmCreated;     // 共享内存创建标记
};

#endif   // SHARED_MEMORY_MANAGER_H