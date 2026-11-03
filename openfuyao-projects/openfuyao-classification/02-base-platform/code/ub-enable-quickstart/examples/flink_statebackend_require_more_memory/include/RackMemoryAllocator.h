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

#ifndef RACK_MEMORY_ALLOCATOR_H
#define RACK_MEMORY_ALLOCATOR_H

#include <iostream>

class RackMemoryAllocator {
public:
    /**
     * @brief 析构函数
     */
    ~RackMemoryAllocator() = default;

    /**
     * @brief 申请内存
     * @param size 申请内存大小
     * @return 返回内存地址
     */
    void* malloc_mem(size_t size);

    /**
     * @brief 释放内存
     * @param ptr 释放内存地址
     */
    void free_mem(void* ptr);
};

#endif   // RACK_MEMORY_ALLOCATOR_H