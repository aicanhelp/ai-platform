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
import { useTranslation } from 'react-i18next';
import '@/styles/components/navbar.less';
import { containerRouterPrefix, usermanageRouterPrefix } from '@/constant.js';
import {
  QuestionCircleOutlined,
  UserOutlined,
  MoonOutlined,
  SunOutlined,
} from '@ant-design/icons';
import LanguageSwitcher from '@/components/LanguageSwitcher';
import {
  Avatar,
  Popover,
  Space,
  Button,
  message,
  Tooltip,
} from 'antd';
import { ResponseCode, docAddress } from '@/common/constants';
import { logout } from '@/api/loginApi';
import { firstAlphabetUp } from '@/tools/utils';
import { getConsolePluginDisplayName } from '@/i18n/utils';
import copilot from '@/assets/images/navbarCopilot.png';
import clusterIcon from '@/assets/images/topIcon/clusterIcon.png';
import userIcon from '@/assets/images/topIcon/userIcon.png';
import openFuyaoLogo from '@/assets/images/openFuyaoLogo.png';
import homeLogo from '@/assets/images/home/homeLogo.png';
import { HomeFilled } from '@ant-design/icons';
function OptionMenu({ onLogout }) {
  const { t } = useTranslation('nav');

  return (
    <Space style={{ flexDirection: 'column' }}>
      <Link to='/setting/userinfo'>
        <Button type='link'>{t('userMenu.settings')}</Button>
      </Link>
      <Button type='link' onClick={onLogout}>
        {t('userMenu.logout')}
      </Button>
    </Space>
  );
}

export default function NavBar() {
  const history = useHistory();
  const { t, i18n } = useTranslation();
  const terminalStore = useStore('terminal');
  const userStore = useStore('user');
  const [key, setKey] = useState('containerPlatform');
  const [messageApi, contextHolder] = message.useMessage();
  // 挂载扩展组件
  const consolePluginStore = useStore('consolePlugins');
  const themeStore = useStore('theme');
  const [themeCheck, setThemeCheck] = useState(themeStore.$s.theme); // 选中进入深色
  const [navbarItems, setNavbarItems] = useState([]);

  useEffect(() => {
    const handleLocationChange = () => {
      const firstName = window.location.pathname.split('/')[1];
      let consolePluginLocation = [];
      consolePluginStore.$s.consolePlugins.forEach(({ pluginName }) => {
        consolePluginLocation.push(pluginName);
      });
      const locationMap = {
        containerPlatform: 'containerPlatform',
        applicationMarket: 'applicationMarket',
        ...consolePluginLocation,
      };
      let temporaryKey = '';
      firstName.split('_').map((item, index) => {
        index ? temporaryKey += firstAlphabetUp(item) : temporaryKey += item;
      });
      let baseNavItems = [];
      consolePluginStore.$s.consolePlugins.map(item => {
        baseNavItems.push({
          ...item,
          name: item.pluginName,
          path: item.pluginName,
          displayName: getConsolePluginDisplayName(item, i18n.language),
        });
      });
      baseNavItems.push({
        enabled: true,
        entrypoint: '/',
        name: 'userManagement',
        displayName: t('platform.userManagement', { ns: 'nav' }),
        path: `${usermanageRouterPrefix}/user`,
      });
      const otherItems = baseNavItems.filter(item => item.name !== 'ai-copilot');
      const aiItems = baseNavItems.filter(item => item.name === 'ai-copilot');
      setNavbarItems([...otherItems, ...aiItems]);
      temporaryKey = locationMap[temporaryKey];
      setKey(temporaryKey);
    };

    handleLocationChange();
    const unListen = history.listen(() => {
      handleLocationChange();
    });

    return () => {
      unListen();
    };
  }, [consolePluginStore.$s.consolePlugins, t, i18n.language, history]);

  // 退出登录
  const handleLogout = async () => {
    logout()
      .then(res => {
        if (res.status === ResponseCode.NoContent) {
          messageApi.success(t('user.logoutSuccess', { ns: 'common' }));
          sessionStorage.removeItem('cluster');
          localStorage.clear();
          setTimeout(() => {
            window.location.href = '/rest/auth/login';
          }, 1000);
        }
      })
      .catch(() => {
        messageApi.error(t('user.logoutFailed', { ns: 'common' }));
      });
  };

  const goDoc = () => {
    const fallback = docAddress.find(item => item.path === '/');
    const matched = docAddress.find(
      item => item.path !== '/' && window.location.href.includes(item.path),
    );
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
        return <img className='top_icon_size' src={userIcon} style={{ marginRight: '6px' }} />;
      case 'containerPlatform':
        return <img className='top_icon_size' src={openFuyaoLogo} style={{ marginRight: '6px' }} />;
      case 'multicluster':
        return <img className='top_icon_size' src={clusterIcon} style={{ marginRight: '6px' }} />;
      case 'ai-copilot':
        return <img className='top_icon_size' src={copilot} style={{ marginRight: '6px' }} />;
      default:
        return <UserOutlined />;
    }
  };

  const handleCloseTerminal = () => {
    terminalStore.$a.setTerminalOpen(false);
  };

  return (
    <div className="nav">
      <div className='flex_top'>
        <Link to='/home' onClick={handleCloseTerminal}>
          <img src={homeLogo} className='homeLogo' alt='home' />
        </Link>
        <div className='splitLine_top'></div>
        <HomeFilled style={{ color: '#fff', cursor: 'pointer' }} onClick={() => history.push(`/${containerRouterPrefix}`)} />
      </div>
      <div className='nav_tools'>
        <div className='navList'>
          {
            navbarItems.filter(item => item.entrypoint === '/' && item.enabled).map((item) => (
              <Link key={item.name} to={`/${item.path}`} className={key === item.name ? 'active' : ''}>
                {getLogo(item)}
                {item.displayName}
              </Link>
            ))
          }
        </div>
        <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
          {contextHolder}
        </div>
        <Tooltip title={t('userMenu.help', { ns: 'nav' })} placement="bottom">
          <QuestionCircleOutlined className='tool_icon' onClick={goDoc} />
        </Tooltip>
        <LanguageSwitcher />
        <Tooltip title={t('theme.toggle', { ns: 'common' })} placement="bottom">
          <div className='tool_theme'>
            {themeCheck === 'light' ? <MoonOutlined className='tool_icon' onClick={() => handleThemeCheck('dark')} /> :
              <SunOutlined className='tool_icon' onClick={() => handleThemeCheck('light')} />}
          </div>
        </Tooltip>
        <Popover
          placement='bottom'
          content={<OptionMenu onLogout={handleLogout} />}
          getPopupContainer={() => document.getElementById('root_container') || document.body}
        >
          <div className='user_info'>
            <Avatar icon={<UserOutlined />} className='avatar' />
            <p>{userStore.$s.user.name}</p>
          </div>
        </Popover>
      </div>
    </div >
  );
}
