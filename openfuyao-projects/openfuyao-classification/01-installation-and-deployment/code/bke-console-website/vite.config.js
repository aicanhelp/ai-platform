/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import react from '@vitejs/plugin-react';
import path from 'path';
import { loadEnv } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';

let alias = {
  react: 'openinula', // 新增
  'react-dom': 'openinula', // 新增
  'react/jsx-dev-runtime': 'openinula/jsx-dev-runtime',
  '@': path.resolve(__dirname, './src'),
};
const env = loadEnv(process.env.NODE_ENV, process.cwd());
export default {
  plugins: [
    react(),
    // visualizer({
    //   gzipSize: true,
    // }),
  ],
  resolve: {
    alias,
  },
  server: {
    host: '0.0.0.0',
    port: 7777,
    proxy: {
      '/clusters': {
        target: env.VITE_API_URL,
        changeOrigin: true,
        ws: true,
        secure: false,
      },
      '/ws/': {
        target: env.VITE_API_URL,
        changeOrigin: true,
        ws: true,
        secure: false,
      },
      '/rest/': {
        target: env.VITE_API_URL,
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      onwarn(warning, warn) {
        if (warning.code === 'MODULE_LEVEL_DIRECTIVE') {
          return;
        }
        warn(warning);
      },
    },
  },
};
