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

#ifndef SAPRK_UB_MEM_SHARE_H
#define SAPRK_UB_MEM_SHARE_H

#include "SharedMemoryManager.h"

#include <memory>
#include <string>

// 封装UB内存共享操作，对接SharedMemoryManager
class SparkUBMemShare {
public:
    /**
     * @brief 构造函数：初始化共享内存配置
     * @param shmManager 共享内存管理实例
     */
    SparkUBMemShare(SharedMemoryManager* shmManager);

    /**
     * @brief 析构函数：自动清理共享内存资源
     */
    ~SparkUBMemShare() = delete;

    /**
     * @brief 创建共享内存
     * @return 成功返回true，失败返回false
     */
    bool createShareMem();

    /**
     * @brief 向共享内存中写入数据
     * @param data 数据信息
     * @return 成功返回true，失败返回false
     */
    bool saveToShareMem(const std::string& data);

    /**
     * @brief 读取共享内存数据
     * @return 读取到的信息
     */
    std::string readFromShareMem();

    /**
     * @brief 删除共享内存
     */
    void deleteShm();

private:
    SharedMemoryManager* shmManager;   // 共享内存管理器实例
};

#endif   // SAPRK_UB_MEM_SHARE_H