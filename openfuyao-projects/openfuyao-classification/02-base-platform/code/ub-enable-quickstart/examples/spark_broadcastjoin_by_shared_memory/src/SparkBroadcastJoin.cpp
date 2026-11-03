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

#include "SparkBroadcastJoin.h"
#include "utils.h"

#include <iostream>
#include <sstream>
#include <thread>

SparkBroadcastJoin::SparkBroadcastJoin(SparkUBMemShare* ubMemShare) : ubMemShare(ubMemShare), isRunning(false) {}

void SparkBroadcastJoin::setLongSlicce(bool isLongSlice) {
    this->isLongSlice = isLongSlice;
}

void SparkBroadcastJoin::createShareMemAndWriteSmallTable(const std::string& smallTableData) {
    bool createOK = ubMemShare->createShareMem();
    if (!createOK) {
        std::cerr << "[SparkBroadcastJoin] 共享内存创建失败!" << std::endl;
        return;
    }
    bool saveOK = ubMemShare->saveToShareMem(smallTableData);
    if (!saveOK) {
        std::cerr << "[SparkBroadcastJoin] 小表写入共享内存失败!" << std::endl;
        return;
    }
    std::cout << "[SparkBroadcastJoin] 接收小表并将其写入共享内存完毕。写入共享内存中小表数据: " << smallTableData
              << std::endl;
}

void SparkBroadcastJoin::join() {
    isRunning           = true;
    int       pollCount = 0;
    const int MAX_POLL  = 3;

    while (isRunning && pollCount < MAX_POLL) {
        pollCount++;
        // 读取共享内存中的小表
        std::string smallTable = ubMemShare->readFromShareMem();
        if (!smallTable.empty()) {
            joinSmallTable(smallTable);
            isRunning = false;
        } else {
            std::cout << "[SparkBroadcastJoin] 未读取到数据, 再次尝试" << (MAX_POLL - pollCount) << "次后退出"
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

void SparkBroadcastJoin::joinSmallTable(const std::string& smallTable) {
    std::vector<std::string> localBigTableSlices;
    if (isLongSlice) {
        localBigTableSlices = {"id1:aa", "id2:bb", "id3:cc", "id2:dd", "id3:ee"};
    } else {
        localBigTableSlices = {"id1:aa", "id2:bb", "id3:cc"};
    }

    std::cout << "[SparkBroadcastJoin] 启动Join流程, 读取共享内存中小标数据: " << smallTable
              << "本地大表切片数量: " << localBigTableSlices.size() << std::endl;
    // 模拟本地大表切片执行Join操作
    for (int i = 0; i < localBigTableSlices.size(); i++) {
        // 示例程序，仅打印日志
        std::cout << "[SparkBroadcastJoin] Join: 第" << (i + 1) << "次" << std::endl;
    }
    std::cout << "[SparkBroadcastJoin] Join: 所有" << localBigTableSlices.size() << "个切片Join流程全部完成"
              << std::endl;
}

void SparkBroadcastJoin::done() {
    ubMemShare->deleteShm();
}
