/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:

 *       http://license.coscl.org.cn/MulanPSL2

 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import Inula, { useEffect, useLayoutEffect, useState, useStore, useCallback } from 'openinula';
import { BrowserRouter } from 'inula-router';
import RootRouters from '@/routes/index';
import NavBar from '@/components/NavBar';
import { ConfigProvider, theme, Flex, Spin, message } from 'antd';
import { containerStylePrefix } from '@/constant.js';
import { getUserName } from '@/api/loginApi';
import { getUserAuth } from '@/api/loginApi';
import { ResponseCode } from '@/common/constants';
import { useIntl } from 'inula-intl';
import { checkSessionStorage } from '@/tools/utils';
import NoPermissions from '@/components/NoPermissions';
import { getClustersList } from '@/api/clusterApi';
import useWebSocket from '@/hooks/useWebSocket';
import consolePluginStore from '@/store/consolePluginStore';
import userStoreAll from '@/store/userStore';
import themeStoreAll from '@/store/themeStore';
import intervalStoreAll from '@/store/intervalStore';
import localeStore from '@/store/localeStore';
import { IntlProvider } from 'inula-intl';
import { getMessages } from '@/i18n';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';

function Content() {
  const themeStore = useStore('theme');

  useEffect(() => {
    const handleChangePop = (e) => {
      const url = e.target.location.href;
      setTimeout(() => {
        window.history.replaceState({}, '', url);
      });
    };
    window.addEventListener('popstate', handleChangePop);
    return () => window.removeEventListener('popstate', handleChangePop);
  }, []);

  return (
    <div className={`content ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`}>
      <RootRouters />
    </div>
  );
}
function App() {
  const intl = useIntl();
  const themeStore = useStore('theme');
  const userStore = useStore('user');
  const localeState = useStore('locale');
  const [ifshowpermisson, setIfshowpermisson] = useState(true);
  const [showLoading, setShowLoading] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [ws, wsData] = useWebSocket('/ws/auth/login-status');

  const getCurrentUserInfo = useCallback(async () => {
    const res = await getUserName();
    if (res.data.code === ResponseCode.OK) {
      userStore.$a.setUserInfo({ name: res.data.data, avatar: '' });
      ifShowPermission(res.data.data);
    }
  }, [userStore]);

  const getOnlyCurrentUserInfo = useCallback(async () => {
    await getUserName();
  }, []);

  const ifShowPermission = async (data) => {
    try {
      const res = await getUserAuth(data);
      if (res.status === ResponseCode.OK) {
        await getClustersList()
          .then((clusterRes) => clusterRes.data)
          .then(async (clusterRes) => {
            setShowLoading(false);
            if (Object.prototype.hasOwnProperty.call(clusterRes, 'info')) {
              if (!checkSessionStorage('cluster')) {
                sessionStorage.setItem(
                  'cluster',
                  Object.keys(clusterRes.info).filter((item) => item === 'host').length > 0
                    ? 'host'
                    : Object.keys(clusterRes.info)[0],
                );
                setIfshowpermisson(false);
              } else {
                setIfshowpermisson(false);
              }
            }
          });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        setShowLoading(false);
        setIfshowpermisson(true);
      } else {
        setShowLoading(true);
        messageApi.error(intl.formatMessage({ id: 'auth.clusterFetchFailed' }));
      }
    }
  };
  useEffect(() => {
    // 获取用户信息
    getCurrentUserInfo();
  }, [getCurrentUserInfo]);

  useEffect(() => {
    const useWsData = JSON.parse(JSON.parse(wsData));
    if (useWsData.loginStatus === 'false') {
      // 调用接口
      getOnlyCurrentUserInfo();
      ws.close();
    }
  }, [wsData]);

  useEffect(() => {
    document.body.style.backgroundColor = themeStore.$s.theme === 'light' ? '#f7f7f7' : '#171a1f';
  }, [themeStore.$s.theme]);
  return (
    <IntlProvider
      key={localeState.$s.locale}
      locale={localeState.$s.locale}
      messages={getMessages(localeState.$s.locale)}
    >
      <div id={`root_container`} className={`container ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`}>
        <ConfigProvider
          prefixCls={containerStylePrefix}
          theme={{ algorithm: themeStore.$s.theme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}
          locale={localeState.$s.locale === 'en-US' ? enUS : zhCN}
        >
          <BrowserRouter>
            <NavBar />
            {!showLoading && (ifshowpermisson ? <NoPermissions /> : <Content />)}
            {showLoading && (
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  paddingTop: '20%',
                }}
              >
                <Flex align="center" gap="middle">
                  <Spin size="large" />
                </Flex>
              </div>
            )}
          </BrowserRouter>
        </ConfigProvider>
      </div>
    </IntlProvider>
  );
}

export default App;
