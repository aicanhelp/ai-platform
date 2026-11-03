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

SparkDriver::SparkDriver(SparkBroadcastJoin* broadcastJoin):broadcastJoin(broadcastJoin){}

void SparkDriver::broadcastSmallTable(){
    // 模拟小表数据
    std::cout<<"[SparkDriver] 开始广播小表..."<<std::endl;
    std::string smallTable="id1:Alice,id2:Bob,id3:Charlie";
    broadcastJoin->createShareMemAndWriteSmallTable(smallTable);
}