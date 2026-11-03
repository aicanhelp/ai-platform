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

import { useEffect, useState, useStore, useCallback } from 'openinula';
import { BrowserRouter } from 'inula-router';
import RootRouters from '@/routes/index';
import NavBar from '@/components/NavBar';
import { ConfigProvider, theme, Flex, Spin, message } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import { containerStylePrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';
import { getUserName } from '@/api/loginApi';
import { getServiceTimeZoneInterface } from '@/api/containerApi';
import { getUserAuth } from '@/api/loginApi';
import { ResponseCode } from '@/common/constants';
import { NamespaceContext } from '@/namespaceContext';
import infoStore from '@/store/infoStore';
import { solveCompareServerAndClientTime, checkSessionStorage } from '@/tools/utils';
import { getClustersList } from '@/api/clusterApi';
import TerminalBox from '@/components/webTerminal/TerminalBox';
import useWebSocket from '@/hooks/useWebSocket';
import ToastMsg from './components/ToastMsg';

function Content({ permitted = false }) {
  const themeStore = useStore('theme');

  useEffect(() => {
    const handleChangePop = e => {
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
      <RootRouters permitted={permitted} />
    </div >
  );
}
function App() {
  const themeStore = useStore('theme');
  const terminalStore = useStore('terminal');
  const infoStoreExample = infoStore();
  const userStore = useStore('user');
  const namespaceStore = useStore('namespace');
  const [namespace, setNamespace] = useState(''); // 初始化命名空间
  const [ifshowpermisson, setIfshowpermisson] = useState(true);
  const [showLoading, setShowLoading] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [ws, wsData] = useWebSocket('/ws/auth/login-status');
  const { t, i18n } = useTranslation('common');

  const getCurrentUserInfo = useCallback(async () => {
    try {
      const res = await getUserName();
      if (res.data.code === ResponseCode.OK) {
        userStore.$a.setUserInfo({ name: res.data.data, avatar: '' });
        ifShowPermission(res.data.data);
      }
    } catch {
      setShowLoading(false);
    }
  }, [userStore]);

  const getOnlyCurrentUserInfo = useCallback(async () => {
    await getUserName();
  }, []);

  const ifShowPermission = async (data) => {
    try {
      const res = await getUserAuth(data);
      if (res.status === ResponseCode.OK) {
        await getClustersList().
          then(clusterRes => clusterRes.data).
          then(async (clusterRes) => {
            setShowLoading(false);
            if (Object.prototype.hasOwnProperty.call(clusterRes, 'info')) {
              if (!checkSessionStorage('cluster')) {
                sessionStorage.setItem('cluster', Object.keys(clusterRes.info).filter(item => item === 'host').length > 0
                  ? 'host' : Object.keys(clusterRes.info)[0]);
                setIfshowpermisson(false);
              } else {
                setIfshowpermisson(false);
              }
            }
          });
      }
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        setShowLoading(false);
        setIfshowpermisson(true);
      } else {
        setShowLoading(false);
        messageApi.error(t('error.clusterFetchFailed'));
      }
    }
  };
  useEffect(() => {
    // 获取用户信息
    getCurrentUserInfo();
  }, [getCurrentUserInfo]);
  const getServiceTimeZone = useCallback(async () => {
    const offset = (0 - (new Date().getTimezoneOffset() / 60));
    const client = `UTC${offset < 0 ? offset : `+${offset}`}`; // 获取客户端时区
    try {
      const res = await getServiceTimeZoneInterface();
      if (res.status === ResponseCode.OK) {
        const offsetCompareLocal = solveCompareServerAndClientTime(res.data.data.offset, client);
        infoStoreExample.$a.setInfoTimeZone({ server: res.data.data.offset, client });
        infoStoreExample.$a.setOffsetCompareLocal(offsetCompareLocal);
      }
    } catch {
      infoStoreExample.$a.setOffsetCompareLocal(0);
    }
  }, []);

  useEffect(() => {
    getServiceTimeZone();
  }, [getServiceTimeZone]); // 获取服务器时区

  useEffect(() => {
    setNamespace(namespaceStore.$s.name);
  }, [namespaceStore.$s.name]);

  useEffect(() => {
    if (!wsData) { return; } // 未收到消息时（初始 null）不解析
    const useWsData = JSON.parse(JSON.parse(wsData));
    if (useWsData.loginStatus === 'false') {
      // 调用接口
      getOnlyCurrentUserInfo();
      ws?.close();
    }
  }, [wsData]);

  useEffect(() => {
    document.body.style.backgroundColor = themeStore.$s.theme === 'light' ? '#f7f7f7' : '#171a1f';
  }, [themeStore.$s.theme]);
  return (
    <div id="root_container" className={`container ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''} ${terminalStore.$s.isOpen ? `${terminalStore.$s.screenSize}_terminal_container` : ''}`}>
      <ToastMsg contextHolder={contextHolder}/>
      <NamespaceContext.Provider value={namespace}>
        <ConfigProvider
          prefixCls={containerStylePrefix}
          locale={i18n.language === 'zh-CN' ? zhCN : enUS}
          theme={{ algorithm: themeStore.$s.theme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}
        >
          <BrowserRouter>
            <NavBar />
            {!showLoading && <Content permitted={!ifshowpermisson} />}
            {showLoading && <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', paddingTop: '20%' }}>
              <Flex align="center" gap="middle">
                <Spin size="large" />
              </Flex>
            </div>}
            {!!terminalStore.$s.isOpen && <TerminalBox />}
          </BrowserRouter>
        </ConfigProvider>
      </NamespaceContext.Provider>
    </div>
  );
}

export default App;
