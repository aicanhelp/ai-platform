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
 * i18n Utilities
 * Helper functions for internationalization
 */

import i18n from './index';
import { languageOptions, getLanguageDisplayName, getLanguageShortName } from './config';

export { getDayjsLocaleForI18nLng, syncDayjsLocale } from './dayjsSync';

/**
 * Change the current language
 * @param {string} langCode - Language code (e.g., 'zh-CN', 'en-US')
 */
export const changeLanguage = (langCode) => {
  if (i18n.language !== langCode) {
    i18n.changeLanguage(langCode);
    // Store in localStorage for persistence
    localStorage.setItem('i18nextLng', langCode);
  }
};

/**
 * Get the current language code
 * @returns {string} Current language code
 */
export const getCurrentLanguage = () => {
  return i18n.language || 'zh-CN';
};

/**
 * Toggle between available languages
 * Cycles through the available languages
 */
export const toggleLanguage = () => {
  const currentLang = getCurrentLanguage();
  const availableLangs = languageOptions.map((l) => l.code);
  const currentIndex = availableLangs.indexOf(currentLang);
  const nextIndex = (currentIndex + 1) % availableLangs.length;
  changeLanguage(availableLangs[nextIndex]);
};

/**
 * Check if a language is supported
 * @param {string} langCode - Language code to check
 * @returns {boolean}
 */
export const isLanguageSupported = (langCode) => {
  return languageOptions.some((l) => l.code === langCode);
};

/**
 * Get all available languages
 * @returns {Array} Array of language objects
 */
export const getAvailableLanguages = () => {
  return languageOptions;
};

// Re-export config helpers
export { getLanguageDisplayName, getLanguageShortName };

/**
 * Translate with fallback value
 * @param {string} key - Translation key
 * @param {string} fallback - Fallback value if translation is missing
 * @param {object} options - i18n options
 * @returns {string} Translated string or fallback
 */
export const tWithFallback = (key, fallback, options = {}) => {
  const translated = i18n.t(key, options);
  if (!translated || translated === key) {
    return fallback;
  }
  return translated;
};

/**
 * 临时硬编码：后端 displayNameI18n 未就绪时，按 pluginName / pageName 覆盖显示名。
 * TODO: 后端 displayNameI18n 稳定且各扩展组件 ConsolePlugin 资源刷新后移除。
 * @type {Array<{ key: string, 'zh-CN'?: string, 'en-US'?: string }>}
 */
export const CONSOLE_PLUGIN_DISPLAY_NAME_OVERRIDES = [
  { key: 'colocation', 'zh-CN': '在离线混部', 'en-US': 'Colocation' },
  { key: 'colocation_overview', 'zh-CN': '概览', 'en-US': 'Overview' },
  { key: 'colocation_configuration', 'zh-CN': '混部策略配置', 'en-US': 'Configuration' },
  { key: 'colocation_monitoring', 'zh-CN': '混部监控', 'en-US': 'Colocation Monitoring' },
  { key: 'cluster', 'zh-CN': '集群生命周期管理', 'en-US': 'Cluster Lifecycle Management' },
  { key: 'log', 'zh-CN': '日志', 'en-US': 'Logs' },
  { key: 'log_query', 'zh-CN': '日志查询', 'en-US': 'Query' },
  { key: 'log_config', 'zh-CN': '日志配置', 'en-US': 'Configuration' },
  { key: 'monitoringDashboard', 'zh-CN': '自定义监控看板', 'en-US': 'Custom Monitoring' },
  { key: 'numa', 'zh-CN': 'NUMA亲和调度', 'en-US': 'NUMA Affinity' },
  { key: 'numa_overview', 'zh-CN': '概览', 'en-US': 'Overview' },
  { key: 'numa_configuration', 'zh-CN': '亲和策略配置', 'en-US': 'Affinity Policy Configuration' },
  { key: 'numa_monitoring', 'zh-CN': '集群NUMA监控', 'en-US': 'Cluster NUMA Monitoring' },
  { key: 'ray', 'zh-CN': 'openFuyao Ray', 'en-US': 'openFuyao Ray' },
  { key: 'ray_overview', 'zh-CN': '概览', 'en-US': 'Overview' },
  { key: 'ray_cluster', 'zh-CN': 'RayCluster', 'en-US': 'RayCluster' },
  { key: 'ray_service', 'zh-CN': 'RayService', 'en-US': 'RayService' },
  { key: 'ray_job', 'zh-CN': 'RayJob', 'en-US': 'RayJob' },
];

const getHardcodedConsolePluginDisplayName = (item, language) => {
  if (language === 'zh-CN') {
    return '';
  }
  const lookupKey = item.displayName;
  if (!lookupKey) {
    return '';
  }
  const entry = CONSOLE_PLUGIN_DISPLAY_NAME_OVERRIDES.find(({ 'zh-CN': zhCN }) => zhCN === lookupKey);
  return entry?.['en-US'] || '';
};

/**
 * Resolve console plugin display name for the current locale.
 * Prefers displayNameI18n[locale], then hardcoded overrides, then displayName / pluginName / pageName.
 * @param {{ displayName?: string, displayNameI18n?: Record<string, string>, pluginName?: string, pageName?: string }} item
 * @param {string} [language]
 * @returns {string}
 */
export const getConsolePluginDisplayName = (item, language = getCurrentLanguage()) => {
  if (!item) {
    return '';
  }
  const localized = item.displayNameI18n?.[language];
  if (localized) {
    return localized;
  }
  const hardcoded = getHardcodedConsolePluginDisplayName(item, language);
  if (hardcoded) {
    return hardcoded;
  }
  return item.displayName || item.pluginName || item.pageName || '';
};
