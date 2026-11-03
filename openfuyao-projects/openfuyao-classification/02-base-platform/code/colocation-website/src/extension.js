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
    pluginName: 'colocation',
    displayName: '在离线混部',
    subPages: [
      {
        pageName: 'ColocationOverview',
        displayName: '概览',
      },
      {
        pageName: 'ColocationConfiguration',
        displayName: '混部策略配置',
      },
      {
        pageName: 'ColocationMonitor',
        displayName: '混部监控',
      },
    ],
  },
};

export default config;