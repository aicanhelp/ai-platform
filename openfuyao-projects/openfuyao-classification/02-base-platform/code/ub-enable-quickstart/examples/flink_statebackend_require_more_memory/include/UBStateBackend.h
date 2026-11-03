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

#ifndef UB_STATE_BACKEND_H
#define UB_STATE_BACKEND_H

#include "MemoryManager.h"

class UBStateBackend {
private:
    MemoryManager* memoryManager;   // 内存管理

public:
    /**
     * @brief 构造函数
     * @param mm 内存管理器
     */
    UBStateBackend(MemoryManager* mm);

    /**
     * @brief 析构函数
     */
    ~UBStateBackend() = default;

    /**
     * @brief 存储状态
     * @param key 状态键值
     * @param value 状态数据
     * @return 存储成功返回true, 失败返回false
     */
    bool saveState(const std::string& key, const std::string& value);

    /**
     * @brief 申请内存
     * @param size 申请内存大小
     * @return 内存地址
     */
    void* requestMemory(size_t size);

    /**
     * @brief 释放内存
     */
    void releaseMemory();

    /**
     * @brief 打印系统内存使用信息
     */
    void showMemeoryUsage();
};

#endif   // UB_STATE_BACKEND_H
