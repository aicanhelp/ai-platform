/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
const config = {
  menu: {
    entrypoint: '/container_platform',
    pluginName: 'scheduling',
    displayName: 'NUMA亲和调度',
    subPages: [
      {
        pageName: 'NumaOverview',
        displayName: '概览',
      },
      {
        pageName: 'AffinityPolicyConfiguration',
        displayName: '亲和策略配置',
      },
      {
        pageName: 'numaMonitor',
        displayName: '集群NUMA监控',
      },
    ],
  },
};

export default config;
