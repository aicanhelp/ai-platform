/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

/**
 * i18n Configuration
 * Configuration for i18next instance
 */

export const i18nConfig = {
  // 关闭跨语言 fallback，避免英文模式下先闪中文再变英文
  fallbackLng: false,
  // 只加载当前语言包，不预拉 fallback 语言
  load: 'currentOnly',
  // JSON 中值为空字符串时仍返回 ''，不继续 fallback
  returnEmptyString: true,
  // 命名空间未加载或缺 key 时返回空字符串，避免短暂显示 plugin.xxx / menu.xxx
  parseMissingKeyHandler: () => '',

  // Default namespace
  defaultNS: 'common',

  /**
   * 命名空间不要在这里全量列出，避免初次加载时获取全部命名空间，降低首屏渲染时间。
   * 让页面通过 useTranslation('xxx') 按需加载对应 ns（HTTP backend 会按需拉取）。
   * 仅加载 UI 框架缩必须的命名空间：common、nav、menu、overview
   */
  ns: ['common', 'nav', 'menu', 'overview'],

  // Supported languages
  supportedLngs: ['zh-CN', 'en-US'],

  // Interpolation settings
  interpolation: {
    escapeValue: true,
  },

  // Language detection settings
  detection: {
    order: ['localStorage', 'navigator'],
    caches: ['localStorage'],
    lookupLocalStorage: 'i18nextLng',
  },

  // React-i18next settings
  react: {
    useSuspense: false, // Required for OpenInula/React 17 compatibility
    bindI18n: 'languageChanged loaded',
    bindI18nStore: 'added removed',
    transEmptyNodeValue: '',
    transSupportBasicHtmlNodes: true,
    transKeepBasicHtmlNodesFor: ['br', 'strong', 'i', 'p'],
  },

  // Backend settings for loading translations via HTTP
  backend: {
    loadPath: '/locales/{{lng}}/{{ns}}.json',
    // Allow cross-origin requests if needed
    crossDomain: false,
    // Cache translation files
    cacheHitMode: 'none',
    // Add version query for cache busting in production
    queryStringParams: { v: '1.0.0' },
  },

  // Missing key handling
  saveMissing: false,
};

// Language display names for the UI
export const languageOptions = [
  {
    code: 'zh-CN',
    name: '简体中文',
    shortName: '中',
  },
  {
    code: 'en-US',
    name: 'English',
    shortName: 'EN',
  },
];

// Get language display name
export const getLanguageDisplayName = (code) => {
  const lang = languageOptions.find((l) => l.code === code);
  return lang?.name || code;
};

// Get language short name
export const getLanguageShortName = (code) => {
  const lang = languageOptions.find((l) => l.code === code);
  return lang?.shortName || code;
};
