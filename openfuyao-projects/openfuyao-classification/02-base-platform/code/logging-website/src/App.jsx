/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import routers from './routers/routers';
import { BrowserRouter, Switch, Route, Redirect } from 'inula-router';
import { useState, useEffect } from 'openinula';
import { IntlProvider } from 'inula-intl';
import { getMessages, getInitialLocale } from './i18n';
import extensionConfig from './extension';
import NavBar from './components/NavBar';
import useLocalStorage from '@/hooks/useLocalStorage';
import { ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import { containerStylePrefix } from './common/constants';
import '@/styles/components/containerIndex.less';
import '@/styles/dark.less';

export default function App() {
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [locale, setLocale] = useState(getInitialLocale());

  useEffect(() => {
    const handleLocaleChange = (e) => {
      if (e.data?.type === 'localeChange' && e.data?.locale) {
        setLocale(e.data.locale);
      }
    };
    window.addEventListener('message', handleLocaleChange);
    // Also listen for the custom event from host
    const handleCustomLocaleChange = (e) => {
      if (e.detail?.locale) {
        setLocale(e.detail.locale);
      }
    };
    window.addEventListener('localeChange', handleCustomLocaleChange);
    return () => {
      window.removeEventListener('message', handleLocaleChange);
      window.removeEventListener('localeChange', handleCustomLocaleChange);
    };
  }, []);

  return (
    <IntlProvider key={locale} locale={locale} messages={getMessages(locale)}>
      <div className={`content ${currentTheme === 'dark' ? 'dark_box' : ''}`}>
        <ConfigProvider
          prefixCls={containerStylePrefix}
          theme={{ algorithm: currentTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}
          locale={locale === 'en-US' ? enUS : zhCN}
        >
          <BrowserRouter basename={`${window.__OPENFUYAO__ ? extensionConfig.menu.entrypoint : ''}/logging`}>
            {!window.__OPENFUYAO__ && <NavBar />}
            <Switch>
              {routers.map((item) => (
                <Route path={item.path} component={item.component} exact />
              ))}
              <Redirect to="/logSearch" />
            </Switch>
          </BrowserRouter>
        </ConfigProvider>
      </div>
    </IntlProvider>
  );
}
