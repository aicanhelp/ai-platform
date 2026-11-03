/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import fs from 'fs';
import path from 'path';
import react from '@vitejs/plugin-react';

export default {
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
  // Public directory configuration - files here will be copied to dist as-is
  publicDir: 'public',
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      react: 'openinula',
      'react-dom': 'openinula',
      'react/jsx-dev-runtime': 'openinula/jsx-dev-runtime',
    },
  },
  server: {
    host: '0.0.0.0',
    port: 8080,
    // https is enabled by default in openfuyao cluster
    // DEV_POD is set in console-website-dev pod menifest
    https: process.env.DEV_POD === 'true' && {
      key: fs.readFileSync('/cert/server.key'),
      cert: fs.readFileSync('/cert/server.crt'),
      ca: fs.readFileSync('/cert/ca.crt'),
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 8080,
    // Reuse the same certs in pod to serve build artifacts over HTTPS
    https: process.env.DEV_POD === 'true' && {
      key: fs.readFileSync('/cert/server.key'),
      cert: fs.readFileSync('/cert/server.crt'),
      ca: fs.readFileSync('/cert/ca.crt'),
    },
  },
};
