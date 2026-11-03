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
import '@/styles/components/containerIndex.less';
import NavBar from '@/components/NavBar';
import useLocalStorage from '@/hooks/useLocalStorage';
import { ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import LocalMenu from '@/components/LocalMenu';
import { containerStylePrefix } from '@/common/constants';
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
      <div className="content">
        <ConfigProvider
          prefixCls={containerStylePrefix}
          theme={{ algorithm: currentTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}
          locale={locale === 'en-US' ? enUS : zhCN}
        >
          <BrowserRouter
            basename={`${window.__OPENFUYAO__ ? extensionConfig.menu.entrypoint : ''}/monitoring-dashboard`}
          >
            {!window.__OPENFUYAO__ && <NavBar />}
            <div className={`container_platform_box ${currentTheme === 'light' ? '' : 'dark_box'}`}>
              {!window.__OPENFUYAO__ && (
                <div className="container_nav_bar">
                  <LocalMenu />
                </div>
              )}
              <div className="container_content">
                <Switch>
                  {routers.map((item) => (
                    <Route path={item.path} component={item.component} exact />
                  ))}
                  <Redirect to="/dashboard" />
                </Switch>
              </div>
            </div>
          </BrowserRouter>
        </ConfigProvider>
      </div>
    </IntlProvider>
  );
}
