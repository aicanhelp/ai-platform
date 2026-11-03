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

const themeStore = createStore({
  id: 'theme',
  state: {
    theme: (() => {
      let localTheme = localStorage.getItem('theme');
      if (!localTheme) {
        localTheme = 'light';
        localStorage.setItem('theme', 'light');
      }
      return localTheme;
    })(),
  },
  actions: {
    setTheme(state, payload) {
      const event = new CustomEvent('localStorageChanged', {
        detail: {
          key: 'theme',
          oldValue: state.theme,
          newValue: payload.theme,
        },
      });
      localStorage.setItem('theme', payload.theme);
      state.theme = payload.theme;
      // 触发自定义事件
      window.dispatchEvent(event);
    },
  },
});

export default themeStore;
