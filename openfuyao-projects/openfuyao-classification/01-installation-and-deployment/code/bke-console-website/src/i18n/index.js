import zhCN from './locales/zh-CN';
import enUS from './locales/en-US';

export const SUPPORTED_LOCALES = {
  'zh-CN': { label: '中文', messages: zhCN },
  'en-US': { label: 'English', messages: enUS },
};

export const DEFAULT_LOCALE = 'zh-CN';

export const getMessages = (locale) => {
  return SUPPORTED_LOCALES[locale]?.messages || SUPPORTED_LOCALES[DEFAULT_LOCALE].messages;
};

export const getInitialLocale = () => {
  return localStorage.getItem('locale') || DEFAULT_LOCALE;
};

export const setLocaleToStorage = (locale) => {
  localStorage.setItem('locale', locale);
  window.__OPENFUYAO_LOCALE__ = locale;
  window.dispatchEvent(new CustomEvent('localeChange', { detail: { locale } }));
};
