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

#include "SparkDriver.h"

#include <iostream>
#include <string>

#include "ubs_mem.h"
#include "ubs_mem_def.h"

#include "utils.h"

// 全局配置参数(根据需求调整)
constexpr const char* SHM_NAME       = "Spark_Broadcast_Shm";   // 共享内存名称
constexpr size_t      SHM_SIZE       = 128ULL << 20;            // 共享内存大小(128M)
constexpr size_t      SHM_CHUNK_SIZE = 1 << 10;                 // 读写块大小(1KB)

void PrintMenu() {
    // 打印菜单
    std::cout << "\n========================================" << std::endl;
    std::cout << "           Spark_UB_Demo            " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "支持操作: " << std::endl;
    std::cout << "  1. 广播小表" << std::endl;
    std::cout << "  2. 读取小表并Join" << std::endl;
    std::cout << "  3. 强制清理共享内存" << std::endl;
    std::cout << "  4. 退出程序" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main() {
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
    // 初始化核心模块实例
    SharedMemoryManager* shmManager    = new SharedMemoryManager(SHM_NAME, SHM_SIZE, SHM_CHUNK_SIZE);
    SparkUBMemShare*     ubMemShare    = new SparkUBMemShare(shmManager);
    SparkBroadcastJoin*  broadcastJoin = new SparkBroadcastJoin(ubMemShare);
    SparkDriver*         driver        = new SparkDriver(broadcastJoin);

    int choice = 0;
    while (true) {
        PrintMenu();
        std::cin >> choice;
        ClearInputBuffer();

        switch (choice) {
            case 1: {
                // 广播小表
                driver->broadcastSmallTable();
                broadcastJoin->setLongSlicce(true);
                break;
            }
            case 2: {
                // 读取小表并Join
                broadcastJoin->join();
                break;
            }
            case 3: {
                std::cout << "\n[main] 开始强制清理共享内存..." << std::endl;
                shmManager->ForceSetCreated(true);
                shmManager->DeleteShm();
                std::cout << "[main] 强制清理共享内存完成" << std::endl;
                break;
            }
            case 4: {
                // 退出程序(清理资源)
                broadcastJoin->done();
                std::cout << "[main] 已退出, 感谢使用! " << std::endl;
                return 0;
            }
            default: {
                std::cerr << "无效操作编号, 请输入1-4之间的数字! " << std::endl;
                break;
            }
        }
    }
    return 0;
}
