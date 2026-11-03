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

const userStore = createStore({
  id: 'user',
  state: {
    user: (() => {
      let localUser = localStorage.getItem('user');
      if (!localUser) {
        localUser = JSON.stringify({ name: '', avatar: '' });
      }
      localStorage.setItem('user', localUser);
      return JSON.parse(localUser);
    })(),
  },
  actions: {
    setUserInfo(state, payload) {
      state.user = { ...state.user, ...payload };
      localStorage.setItem('user', JSON.stringify({
        name: payload.name,
        avatar: payload.avatar,
      }));
    },
  },
});

export default userStore;
