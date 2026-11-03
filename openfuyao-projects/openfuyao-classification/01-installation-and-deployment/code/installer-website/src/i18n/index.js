import zhCN from './locales/zh-CN';
import enUS from './locales/en-US';

export const DEFAULT_LOCALE = 'zh-CN';

const messages = { 'zh-CN': zhCN, 'en-US': enUS };

export const getMessages = (locale) => messages[locale] || messages[DEFAULT_LOCALE];

export const getInitialLocale = () => window.__OPENFUYAO_LOCALE__ || localStorage.getItem('locale') || DEFAULT_LOCALE;

// Re-export useIntl from inula-intl for convenience
export { useIntl } from 'inula-intl';
