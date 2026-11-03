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

#ifndef SAPRK_DRIVER_H
#define SAPRK_DRIVER_H

#include "SparkBroadcastJoin.h"

// Spark 驱动类，广播小表
class SparkDriver {

public:
    /**
     * @brief 构造函数
     * @param broadcastJoin Join 处理实例
     */
    SparkDriver(SparkBroadcastJoin* broadcastJoin);

    /**
     * @brief 析构函数
     */
    ~SparkDriver() = delete;

    /**
     * @brief 广播小表
     */
    void broadcastSmallTable();

private:
    SparkBroadcastJoin* broadcastJoin;   // Join 处理实例
};

#endif   // SAPRK_DRIVER_H