/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Link, useHistory } from 'inula-router';
import { useEffect, useState, useStore } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/components/navbar.less';
import { containerRouterPrefix, usermanageRouterPrefix } from '@/constant.js';
import {
  QuestionCircleOutlined,
  UserOutlined,
  MoonOutlined,
  SunOutlined,
  GlobalOutlined,
  HomeFilled,
} from '@ant-design/icons';
import { Avatar, Popover, Space, Button, message, Switch, Dropdown } from 'antd';
import { ResponseCode, docAddress } from '@/common/constants';
import { logout } from '@/api/loginApi';
import { firstAlphabetUp } from '@/tools/utils';
import copilot from '@/assets/images/navbarCopilot.png';
import clusterIcon from '@/assets/images/topIcon/clusterIcon.png';
import userIcon from '@/assets/images/topIcon/userIcon.png';
import openFuyaoLogo from '@/assets/images/openFuyaoLogo.png';
import localeStore from '@/store/localeStore';
import { SUPPORTED_LOCALES } from '@/i18n';
import { getConsolePluginDisplayName } from '@/i18n/utils';
import publicLink from '@/common/publicLink.json';

function OptionMenu({ onLogout }) {
  const intl = useIntl();
  const [themeCheck, setThemeCheck] = useState(localStorage.theme || 'light');
  const themeStore = useStore('theme');

  const handleThemeCheck = (checked) => {
    const nowTheme = checked ? 'dark' : 'light';
    setThemeCheck(nowTheme);
    themeStore.$a.setTheme({ theme: nowTheme });
  };

  return (
    <Space style={{ flexDirection: 'column' }}>
      <Link to="/setting/userinfo">
        <Button type="link">{intl.formatMessage({ id: 'nav.userSettings' })}</Button>
      </Link>
      <Button type="link" onClick={onLogout}>
        {intl.formatMessage({ id: 'nav.logout' })}
      </Button>
    </Space>
  );
}

export default function NavBar() {
  const history = useHistory();
  const intl = useIntl();
  const userStore = useStore('user');
  const localeState = useStore('locale');
  const [key, setKey] = useState('containerPlatform');
  const [messageApi, contextHolder] = message.useMessage();
  // 挂载扩展组件
  const consolePluginStore = useStore('consolePlugins');
  const themeStore = useStore('theme');
  const [themeCheck, setThemeCheck] = useState(themeStore.$s.theme); // 选中进入深色
  const [navbarItems, setNavbarItems] = useState([]);

  const localeMenuItems = Object.entries(SUPPORTED_LOCALES).map(([key, val]) => ({
    key,
    label: val.label,
  }));
  const handleLocaleChange = ({ key: newLocale }) => {
    localeState.$a.setLocale({ locale: newLocale });
  };

  useEffect(() => {
    const handleLocationChange = () => {
      const [prefix, firstName] = window.location.pathname.split('/');
      let consolePluginLocation = [];
      consolePluginStore.$s.consolePlugins.map(({ pluginName }) => {
        let obj = {};
        consolePluginLocation.push((obj[pluginName] = pluginName));
      });
      const locationMap = {
        containerPlatform: 'containerPlatform',
        applicationMarket: 'applicationMarket',
        ...consolePluginLocation,
      };
      let temporaryKey = '';
      firstName.split('_').map((item, index) => {
        index ? (temporaryKey += firstAlphabetUp(item)) : (temporaryKey += item);
      });
      let baseNavItems = [];
      consolePluginStore.$s.consolePlugins.map((item) => {
        baseNavItems.push({
          ...item,
          name: item.pluginName,
          path: item.pluginName,
          displayName: getConsolePluginDisplayName(item, intl.locale),
        });
      });
      baseNavItems.push({
        enabled: true,
        entrypoint: '/',
        name: 'userManagement',
        displayName: intl.formatMessage({ id: 'nav.userManagement' }),
        path: `${usermanageRouterPrefix}/user/usermanage`,
      });
      const otherItems = baseNavItems.filter((item) => item.name !== 'ai-copilot');
      const aiItems = baseNavItems.filter((item) => item.name === 'ai-copilot');
      setNavbarItems([...otherItems, ...aiItems]);
      temporaryKey = locationMap[temporaryKey];
      setKey(temporaryKey);
    };

    const observer = new MutationObserver(() => {
      handleLocationChange();
    });

    observer.observe(document, { subtree: true, childList: true });

    return () => {
      observer.disconnect();
    };
  }, [consolePluginStore.$s.consolePlugins]);

  // 退出登录
  const handleLogout = async () => {
    logout()
      .then((res) => {
        if (res.status === ResponseCode.NoContent) {
          messageApi.success(intl.formatMessage({ id: 'auth.logoutSuccess' }));
          sessionStorage.removeItem('cluster');
          localStorage.clear();
          setTimeout(() => {
            window.location.href = '/rest/auth/login';
          }, 1000);
        }
      })
      .catch((err) => {
        messageApi.error(intl.formatMessage({ id: 'auth.logoutFailed' }));
      });
  };

  const goDoc = () => {
    const fallback = docAddress.find((item) => item.path === '/');
    const matched = docAddress.find((item) => item.path !== '/' && window.location.href.includes(item.path));
    window.open(matched?.address ?? fallback?.address, '_blank');
  };
  // 控制黑夜与白天模式
  const handleThemeCheck = (checked) => {
    const nowTheme = checked === 'dark' ? 'dark' : 'light';
    setThemeCheck(nowTheme);
    themeStore.$a.setTheme({ theme: nowTheme });
  };

  // logo赋值
  const getLogo = (data) => {
    switch (data.name) {
      case 'userManagement':
        return <img className="top_icon_size" src={userIcon} style={{ marginRight: '6px' }} />;
      case 'containerPlatform':
        return <img className="top_icon_size" src={openFuyaoLogo} style={{ marginRight: '6px' }} />;
      case 'multicluster':
        return <img className="top_icon_size" src={clusterIcon} style={{ marginRight: '6px' }} />;
      case 'ai-copilot':
        return <img className="top_icon_size" src={copilot} style={{ marginRight: '6px' }} />;
      default:
        return <UserOutlined />;
    }
  };

  return (
    <div className={`nav`}>
      <div className="flex_top">
        <p className="homeLogo">openFuyao</p>
        <div className="splitLine_top"></div>
        {(() => {
          if (window.location.pathname.startsWith('/container_platform')) {
            return (
              <HomeFilled
                style={{ color: '#fff', cursor: 'pointer' }}
                onClick={() => {
                  window.postMessage({ type: 'relayRouter', isHome: true });
                }}
              />
            );
          } else if (window.location.pathname.startsWith('/user_manage')) {
            return (
              <Link to="/container_platform/cluster">
                <HomeFilled style={{ color: '#fff', cursor: 'pointer' }} />
              </Link>
            );
          } else if (window.location.pathname.startsWith('/setting/userinfo')) {
            return (
              <Link to="/container_platform/cluster">
                <HomeFilled style={{ color: '#fff', cursor: 'pointer' }} />
              </Link>
            );
          } else {
            return null;
          }
        })()}
      </div>
      <div className="nav_tools">
        <div className="navList">
          {navbarItems
            .filter((item) => item.entrypoint === '/' && item.enabled && item.name !== 'multicluster')
            .map((item) => (
              <Link to={`/${item.path}`} className={key === item.name ? 'active' : ''}>
                {getLogo(item)}
                {item.displayName}
              </Link>
            ))}
        </div>
        <div
          style={{
            background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff',
            color: themeStore.$s.theme === 'dark' ? '#fff' : '#333',
          }}
        >
          {contextHolder}
        </div>
        <Dropdown
          menu={{
            items: localeMenuItems,
            selectedKeys: [localeState.$s.locale],
            onClick: handleLocaleChange,
          }}
        >
          <GlobalOutlined className="tool_icon" />
        </Dropdown>
        <Popover
          placement="bottom"
          content={
            <Button type="link" onClick={goDoc}>
              {intl.formatMessage({ id: 'nav.helpDocs' })}
            </Button>
          }
        >
          <div onClick={goDoc}>
            <QuestionCircleOutlined className="tool_icon" />
          </div>
        </Popover>
        <div className="tool_theme">
          {themeCheck === 'light' ? (
            <MoonOutlined className="tool_icon" onClick={() => handleThemeCheck('dark')} />
          ) : (
            <SunOutlined className="tool_icon" onClick={() => handleThemeCheck('light')} />
          )}
        </div>
        <Popover
          placement="bottom"
          content={<OptionMenu onLogout={handleLogout} />}
          getPopupContainer={() => document.getElementById('container')}
        >
          <div className="user_info">
            <Avatar icon={<UserOutlined />} className="avatar" />
            <p>{userStore.$s.user.name}</p>
          </div>
        </Popover>
      </div>
    </div>
  );
}
