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
import { BrowserRouter, Switch, Route, Redirect, useHistory } from 'inula-router';
import extensionConfig from './extension';
import '@/styles/components/containerIndex.less';
import NavBar from '@/components/NavBar';
import LocalMenu from '@/components/LocalMenu';
import useLocalStorage from '@/hooks/useLocalStorage';
import { ConfigProvider, theme } from 'antd';
import { containerStylePrefix } from '@/common/constants';
import '@/styles/pages/index.less';
import { useEffect, useState } from 'openinula';
import '@/styles/dark.less';
import { IntlProvider } from 'inula-intl';
import { getInitialLocale, DEFAULT_LOCALE, getMessages } from './i18n';
import enUS from 'antd/locale/en_US';
import zhCN from 'antd/locale/zh_CN';

export default function App() {
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [locale, setLocale] = useState(getInitialLocale() || DEFAULT_LOCALE);

  useEffect(() => {
    const handleMessage = (e) => {
      if (e.data?.type === 'localeChange' && e.data?.locale) {
        setLocale(e.data.locale);
      }
    };
    const handleCustom = (e) => {
      if (e.detail?.locale) {
        setLocale(e.detail.locale);
      }
    };
    window.addEventListener('message', handleMessage);
    window.addEventListener('localeChange', handleCustom);
    return () => {
      window.removeEventListener('message', handleMessage);
      window.removeEventListener('localeChange', handleCustom);
    };
  }, []);

  return (
    <>
      <div className="content">
        <ConfigProvider
          prefixCls={containerStylePrefix}
          theme={{ algorithm: currentTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}
          locale={locale === 'en-US' ? enUS : zhCN}
        >
          <BrowserRouter basename={`${window.__OPENFUYAO__ ? extensionConfig.menu.entrypoint : ''}/cluster`}>
            <IntlProvider key={locale} locale={locale} messages={getMessages(locale)}>
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
                      <Route key={item.path} path={item.path} component={item.component} exact />
                    ))}
                    <Redirect to="/cluster" />
                  </Switch>
                  <RecieveCom />
                </div>
              </div>
            </IntlProvider>
          </BrowserRouter>
        </ConfigProvider>
      </div>
    </>
  );
}

const RecieveCom = () => {
  const history = useHistory();

  const handlePassMessage = (event) => {
    if (event.data.type === 'relayRouter') {
      if (event.data.isHome) {
        history.push('/clusterManage');
      }
    }
  };
  useEffect(() => {
    window.addEventListener('message', handlePassMessage);
    return () => window.removeEventListener('message', handlePassMessage);
  }, []);
  return null;
};
