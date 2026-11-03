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
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';
import extension from './src/extension.js';
import postcssPrefixSelector from 'postcss-prefix-selector';
import 'dotenv/config';

let alias = {
  react: 'openinula', // 新增
  'react-dom': 'openinula', // 新增
  'react/jsx-dev-runtime': 'openinula/jsx-dev-runtime',
  '@': path.resolve(__dirname, './src'),
};

const { pluginName } = extension.menu;

const pluginBuildConfig = {
  build: {
    lib: {
      entry: path.resolve(__dirname, 'index.html'),
      name: pluginName,
      formats: ['es'],
    },
    rollupOptions: {
      output: {
        entryFileNames: `${pluginName}/${pluginName}.mjs`,
      },
      onwarn(warning, warn) {
        if (warning.code === 'MODULE_LEVEL_DIRECTIVE') {
          return;
        }
        warn(warning);
      },
    },
  },
  css: {
    postcss: {
      plugins: [
        postcssPrefixSelector({
          prefix: `#${pluginName}_root`,
          transform(prefix, selector, prefixedSelector) {
            if (selector.startsWith('body') || selector.startsWith('html')) {
              return selector;
            }
            return prefixedSelector;
          },
        }),
      ],
    },
  },
  define: {
    'process.env': 'new Object({ NODE_ENV: "production" })',
  },
  base: './',
};

export default {
  ...pluginBuildConfig,
  plugins: [
    react(),
    cssInjectedByJsPlugin(),
  ],
  resolve: {
    alias,
  },
  server: {
    host: '0.0.0.0',
    port: 7777,
    proxy: {
      '/rest/multicluster': {
        target: process.env.DEV_BACKEND_HOST, // specify dev backend host in .env file
        changeOrigin: true,
        secure: false,
      },
    },
  },
};
