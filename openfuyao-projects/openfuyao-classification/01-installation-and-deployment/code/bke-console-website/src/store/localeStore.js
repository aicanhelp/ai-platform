import { createStore } from 'openinula';
import { DEFAULT_LOCALE } from '@/i18n';

const localeStore = createStore({
  id: 'locale',
  state: {
    locale: (() => {
      const saved = localStorage.getItem('locale');
      if (saved && (saved === 'zh-CN' || saved === 'en-US')) {
        return saved;
      }
      localStorage.setItem('locale', DEFAULT_LOCALE);
      return DEFAULT_LOCALE;
    })(),
  },
  actions: {
    setLocale(state, payload) {
      localStorage.setItem('locale', payload.locale);
      state.locale = payload.locale;
      window.__OPENFUYAO_LOCALE__ = payload.locale;
      window.dispatchEvent(new CustomEvent('localeChange', { detail: { locale: payload.locale } }));
    },
  },
});

export default localeStore;
