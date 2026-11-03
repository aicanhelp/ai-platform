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

#include "FlinkTaskManager.h"
#include "LocalMemoryAllocator.h"
#include "MemoryManager.h"
#include "RackMemoryAllocator.h"
#include "UBStateBackend.h"

#include "ubs_mem.h"
#include "ubs_mem_def.h"


int main() {
    LocalMemoryAllocator localAlloc;                      // 本地内存分配器
    RackMemoryAllocator  rackAlloc;                       // 远程内存分配器
    const size_t         memoryThreshold  = 1ULL << 30;   // 本地内存限制阈值：1GB
    const size_t         ALLOCATION_SIZE  = 1ULL << 30;   // 单次申请内存大小：1GB
    const int            ALLOCATION_COUNT = 2;            // 内存申请次数控制：2次

    // 初始化ubs
    ubsmem_options_t options{};
    auto             ret = ubsmem_init_attributes(&options);
    if (ret != 0) {
        printf("ubsm 初始化属性失败!，错误码: %d\n", ret);
        return -1;
    }

    ret = ubsmem_initialize(&options);
    if (ret != 0) {
        printf("ubsm 初始化失败!，错误码: %d\n", ret);
        return -1;
    }

    MemoryManager*    mm          = new MemoryManager(&localAlloc, &rackAlloc, memoryThreshold);
    UBStateBackend*   sb          = new UBStateBackend(mm);
    FlinkTaskManager* taskManager = new FlinkTaskManager("UBTask", sb);

    printf("\n=======================================\n");
    printf("模拟Flink存储状态数据, 设定本地内存限制使用1GB\n");
    printf("每次申请内存1GB用于模拟数据存储, 申请2次\n");
    printf("=======================================\n");
    printf("\n执行前内存使用情况\n");
    sb->showMemeoryUsage();

    printf("模拟Flink任务, 调用Statebackend接口保存状态\n");
    for (int i = 0; i < ALLOCATION_COUNT; i++) {
        printf("模拟保存状态数据, 申请内存 %zu GB (第 %d 次)\n", ALLOCATION_SIZE / (1024 * 1024 * 1024), i + 1);
        if (!taskManager->process()) {
            break;
        }
        printf("第 %d 次, 保存数据后内存使用情况:\n", i + 1);
        sb->showMemeoryUsage();
    }
    sb->releaseMemory();
    delete sb;
    delete mm;
    delete taskManager;
    return 0;
}