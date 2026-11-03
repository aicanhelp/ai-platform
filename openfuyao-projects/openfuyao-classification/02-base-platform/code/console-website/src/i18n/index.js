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
 * i18n Initialization
 * Initialize i18next with react-i18next for OpenInula (React 17 compatible)
 * Uses HTTP backend to load translation files dynamically
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import HttpBackend from 'i18next-http-backend';
import { i18nConfig } from './config';
import { syncDayjsLocale } from './dayjsSync';

// Initialize i18next with HTTP backend for loading translations
const i18nReady = i18n
  .use(HttpBackend) // Load translations via HTTP
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    ...i18nConfig,
    // Resources will be loaded dynamically via HTTP backend
    // from /locales/{{lng}}/{{ns}}.json
  })
  .then(() => {
    syncDayjsLocale(i18n.language);
  });

i18n.on('languageChanged', (lng) => {
  syncDayjsLocale(lng);
  // Propagate locale to monitoring-dashboard and other plugin sub-apps
  window.__OPENFUYAO_LOCALE__ = lng;
  localStorage.setItem('locale', lng);
  document.cookie = `locale=${lng};path=/;max-age=31536000`;
  window.dispatchEvent(new CustomEvent('localeChange', { detail: { locale: lng } }));
});

export default i18n;
export { i18nReady };
