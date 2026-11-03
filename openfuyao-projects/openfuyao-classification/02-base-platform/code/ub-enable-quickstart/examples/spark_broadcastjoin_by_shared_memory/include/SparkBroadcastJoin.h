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

#ifndef SPARK_BROADCAST_JOIN_H
#define SPARK_BROADCAST_JOIN_H

#include "SparkUBMemShare.h"

#include <atomic>
#include <map>
#include <string>
#include <vector>

// 广播Join核心逻辑，广播小表，执行Join
class SparkBroadcastJoin {
public:
    /**
     * @brief 构造函数
     * @param ubMemShare 共享内存操作器
     */
    SparkBroadcastJoin(SparkUBMemShare* ubMemShare);

    /**
     * @brief 析构函数
     */
    ~SparkBroadcastJoin() = delete;

    /**
     * @brief 广播小表存储共享内存
     * @param smallTableData 小表数据
     */
    void createShareMemAndWriteSmallTable(const std::string& smallTableData);

    /**
     * @brief Join 表
     */
    void join();

    /**
     * @brief 结束流程
     */
    void done();

    /**
     * @brief 设置本地大表长度标记
     * @param isLongSlicce true 切片较多
     */
    void setLongSlicce(bool isLongSlicce);

private:

    /**
     * @brief 执行Join操作
     */
    void joinSmallTable(const std::string& smallTable);

private:
    bool              isLongSlice = false;   // 本地大表长度标记
    SparkUBMemShare*  ubMemShare;            // 共享内存操作器
    std::atomic<bool> isRunning;             // 轮训控制标志
};

#endif   // SPARK_BROADCAST_JOIN_H