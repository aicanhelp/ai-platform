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

const terminalStore = createStore({
  id: 'terminal',
  state: {
    type: '',
    name: '',
    screenSize: 'middle',
    isOpen: false,
    podDescription: {
      namespace: '',
      pod: '',
    },
  },
  actions: {
    setTerminalInit(state, payload) {
      state.type = payload.type;
      state.name = payload.name;
    },
    setTerminalScreenSize(state, payload) {
      state.screenSize = payload;
    },
    setTerminalOpen(state, payload) {
      state.isOpen = payload;
    },
    setTerminalPodDescription(state, payload) {
      state.podDescription = { ...payload };
    },
  },
});

export default terminalStore;