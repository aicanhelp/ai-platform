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

#include "RackMemoryAllocator.h"

#include "ubs_mem.h"

void* RackMemoryAllocator::malloc_mem(size_t size) {
    auto        mem_distance = static_cast<ubsmem_distance_t>(0);
    void*       attr         = nullptr;
    std::string region_name  = "default";
    int         ret          = ubsmem_lease_malloc(region_name.c_str(), size, mem_distance, false, &attr);
    if (ret != 0) {
        printf("借用远程内存失败!，错误码: %d\n", ret);
        return nullptr;
    }
    return attr;
}

void RackMemoryAllocator::free_mem(void* ptr) {
    ubsmem_lease_free(ptr);
}