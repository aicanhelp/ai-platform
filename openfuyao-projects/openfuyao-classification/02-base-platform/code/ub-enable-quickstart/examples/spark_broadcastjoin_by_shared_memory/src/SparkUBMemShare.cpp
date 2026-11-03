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

#include "SparkUBMemShare.h"

#include <iostream>

SparkUBMemShare::SparkUBMemShare(SharedMemoryManager* shmManager) : shmManager(shmManager) {}

bool SparkUBMemShare::createShareMem() {
    std::string createdName = shmManager->CreateShm();
    return !createdName.empty();
}

bool SparkUBMemShare::saveToShareMem(const std::string& smallTableData) {
    return shmManager->WriteShm(smallTableData);
}

std::string SparkUBMemShare::readFromShareMem() {
    return shmManager->ReadShm();
}

void SparkUBMemShare::deleteShm() {
    shmManager->DeleteShm();
}