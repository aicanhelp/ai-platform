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

#ifndef FLINK_TASK_MANAGER_H
#define FLINK_TASK_MANAGER_H

#include "UBStateBackend.h"

#include <iostream>

class FlinkTaskManager {
private:
    const std::string key;            // 存储key
    UBStateBackend*   stateBackend;   // 存储后端UBStateBackend

public:
    /**
     * @brief 构造函数
     * @param key 存储key
     * @param backend 存储后端
     */
    FlinkTaskManager(const std::string& key, UBStateBackend* backend);


    /**
     * @brief 析构函数
     */
    ~FlinkTaskManager() = default;

    /**
     * @brief 模拟Flink任务处理
     */
    bool process();
};

#endif   // FLINK_TASK_MANAGER_H