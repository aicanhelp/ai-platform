/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { createStore } from 'openinula';

const infoStore = createStore({
  id: 'info',
  state: {
    timeZone: {
      server: 'UTC+08:00',
      client: 'UTC+8',
    },
    offsetCompareLocal: NaN,
    locale: 'zh-CN',
  },
  actions: {
    setInfoTimeZone(state, payload) {
      state.timeZone = { ...state.timeZone, ...payload };
    },
    setOffsetCompareLocal(state, payload) {
      state.offsetCompareLocal = payload;
    },
    setInfoLocale(state, payload) {
      state.locale = payload;
    },
  },
});

export default infoStore;
