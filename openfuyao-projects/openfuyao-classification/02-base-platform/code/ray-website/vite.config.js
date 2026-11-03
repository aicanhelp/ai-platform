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
            if (selector.startsWith('body') || selector.startsWith('html') || selector.startsWith('.ray_rules_col_popver')) {
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
      '/rest/ray/rest/v1': {
        target: process.env.VITE_RAY_API_URL,
        changeOrigin: true,
        secure: false,
        rewrite: queryPath => queryPath.replace(/^\/rest\/ray\/rest\/v1/, '/rest/v1/'),
        configure: (proxy, options) => {
          proxy.on('proxyRes', (proxyRes, req, res) => {
            delete proxyRes.headers['Content-Security-Policy'];
            delete proxyRes.headers['X-Content-Type-Options'];
            delete proxyRes.headers['X-Frame-Options'];
            delete proxyRes.headers['X-Xss-Protection'];

          })
        }
      },
      '/api/kubernetes': {
        target: process.env.VITE_KUBERNETES_API_URL,
        changeOrigin: true,
      },
      '/rest/ray/grafana': {
        target: process.env.VITE_GRAFANA_URL,
        changeOrigin: true,
      },
      '/rest/ray/prometheus': {
        target: process.env.VITE_PROMETHEUS_URL,
        secure: false,
        changeOrigin: true,
        rewrite: queryPath => queryPath.replace(/^\/rest\/ray\/prometheus/, ''),
      },
      '/rest/ray/loki':{
        target: process.env.VITE_LOKI_URL,
        secure: false,
        changeOrigin: true,
        rewrite: queryPath => queryPath.replace(/^\/rest\/ray/, ''),
      }
    },
  },
};
