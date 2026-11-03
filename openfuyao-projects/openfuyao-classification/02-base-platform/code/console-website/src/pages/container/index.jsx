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

import '@/styles/pages/containerIndex.less';
import '@/styles/components/namespaceBar.less';
import MenuBar from '@/components/Menu.jsx';
import ContainerRouters from '@/routes/containerRouters.jsx';
import NamespaceBar from '@/components/NamespaceBar';
import { useState, useEffect, useStore, Fragment } from 'openinula';
import { useTranslation } from 'react-i18next';
import clusterIcon from '@/assets/images/cluster_icon.png';
import { getClustersList } from '@/api/clusterApi';
import { ResponseCode } from '@/common/constants';
import { Link, useHistory } from 'inula-router';
import '@/styles/components/navbar.less';
import { containerRouterPrefix } from '@/constant.js';
import alertNoData from '@/assets/images/navbarAlert.png';
import { filterAlertStatus } from '@/utils/common';
import dayjs from 'dayjs';
import { getAlertOptions } from '@/api/containerAlertApi';
import fastDeploy from '@/assets/images/fastDeploy.png';
import { AppstoreAddOutlined, AlertOutlined, CodeOutlined } from '@ant-design/icons';
import { Button, Select, Checkbox, Timeline, Modal, Tooltip, message } from 'antd';
import NoPermissionsModal from '@/components/NoPermissionsModal';
import TerminalNewModal from '@/components/webTerminal/TerminalNewModal';
import { checkTerminalAccess } from '@/api/terminalApi';
import ToastMsg from '@/components/ToastMsg';

function AlertBox({ onJump, onAlertCount }) {
  const { t } = useTranslation('nav');
  const [critical, setCritical] = useState(0);
  const [warning, setWarning] = useState(0);
  const [info, setInfo] = useState(0);
  const [alertList, setAlertList] = useState([]);
  const [alertTimeLineItems, setAlertTimeLineItems] = useState([]);
  const history = useHistory();
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const toAlertPage = () => {
    history.push(`/${containerRouterPrefix}/alert/alertIndex`);
    onJump();
  };

  useEffect(() => {
    getAlertLists();
  }, []);

  const getAlertLists = async () => {
    let timeLine = [];
    const res = await getAlertOptions().catch((e) => {
      messageApi.error(t('messages.fetchDataError', { ns: 'common', message: e.response?.data?.msg }));
      return null;
    });
    if (!res) { return; }
    let _alertList = [];
    if (res.status === ResponseCode.OK) {
      setAlertList(res.data);
      let _filterData = res.data.filter(
        (item) => item.status.state === 'active',
      );
      onAlertCount?.(_filterData.length);
      _filterData = _filterData.sort((a, b) => dayjs(b.startsAt).unix() - dayjs(a.startsAt).unix());
      _filterData.slice(0, 5).map((item) => {
        let _resourceType = '';
        if (item.labels.prometheus) {
          _resourceType = 'prometheus';
        }
        if (item.labels.loki) {
          _resourceType = 'loki';
        }
        let _color;
        let _style;
        if (item.labels.severity === 'warning') {
          _color = '#f4840c';
          _style = {
            color: '#f4840c',
            background: '#fffaf3ff',
            border: '1px solid #f4840c',
          };
        } else if (item.labels.severity === 'critical') {
          _color = '#f5595b';
          _style = {
            color: '#f5595b',
            background: '#fff2f0ff',
            border: '1px solid #f5595b',
          };
        } else {
          _color = '#4b8bea';
          _style = {
            color: '#4b8bea',
            background: '#f0f8ff',
            border: '1px solid #4b8bea',
          };
        }

        _alertList.push({
          name: item.labels.alertname ? item.labels.alertname : '--',
          level: item.labels.severity ? item.labels.severity : '--',
          time: item.startsAt
            ? dayjs(item.startsAt).format('YYYY-MM-DD HH:mm:ss')
            : '--',
          resource: _resourceType,
          color: _color,
          style: _style,
          desc: item.annotations.description ? item.annotations.description : '--',
          summary: item.annotations.summary ? item.annotations.summary : '--',
          labels: item.labels,
        });
      });
      setCritical(_filterData.filter((item) => item.labels.severity === 'critical').length);
      setWarning(_filterData.filter((item) => item.labels.severity === 'warning').length);
      setInfo(_filterData.filter((item) => item.labels.severity === 'info').length);
    }
    _alertList.map((item) => {
      timeLine.push({
        color: item.color,
        children: (
          <Link
            className='alert_hover'
            to={{ pathname: `/${containerRouterPrefix}/alert/alertIndex/detail/${item.name}`, state: { currentdetail: item } }}
            onClick={toAlertPage}
          >
            <div className='alert_timeLine_box'>
              <p className='alert_timeLine_box_level' style={item.style}>
                {filterAlertStatus(item.level)}
              </p>
              <div className='alert_timeLine_box_content'>
                <p className='alert_timeLine_box_content_title'>{t('alert.alertName')}：</p>
                <p className='alert_timeLine_box_content_value'>{item.name}</p>
              </div>
              <div className='alert_timeLine_box_content'>
                <p className='alert_timeLine_box_content_title'>{t('alert.alertSource')}：</p>
                <p className='alert_timeLine_box_content_value'>
                  {item.resource}
                </p>
              </div>
              <div className='alert_timeLine_box_content'>
                <p className='alert_timeLine_box_content_title'>{t('alert.triggerTime')}：</p>
                <p className='alert_timeLine_box_content_value'>{item.time}</p>
              </div>
            </div>
          </Link>
        ),
      });
    });
    let _timeline = [
      {
        color: '#89939b',
        children: (
          <p style={{ fontSize: '12px', color: '#89939b' }}>
            {t('alert.more')}
          </p>
        ),
      },
    ];
    setAlertTimeLineItems([...timeLine.concat(_timeline)]);
  };

  return (
    <div className='alert-notification-content'>
      <ToastMsg contextHolder={contextHolder} />
      <div className='alert-notification-content-group'>
        <div className='alert-notification-content-single'>
          <p className='alert-notification-content-single-title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('alert.critical')}：</p>
          <p className='single-count1'>{critical}</p>
        </div>
        <div className='alert-notification-content-single'>
          <p className='alert-notification-content-single-title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('alert.warning')}：</p>
          <p className='single-count2'>{warning}</p>
        </div>
        <div className='alert-notification-content-single'>
          <p className='alert-notification-content-single-title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('alert.info')}：</p>
          <p className='single-count3'>{info}</p>
        </div>
      </div>
      <img src={alertNoData} className='alertImg' alt="" style={{ display: alertList.length === 0 ? 'block' : 'none' }} />
      <p className='alertTip' style={{ display: alertList.length === 0 ? 'block' : 'none' }}>{t('alert.noData')}</p>
      <Timeline items={alertTimeLineItems} style={{ display: alertList.length > 0 ? 'block' : 'none' }} />
      <div className='alert-notification-content-bottom'>
        <Button
          type='link'
          className='alert-notification-content-bottom-button'
          onClick={toAlertPage}
        >
          {t('alert.detail')}
        </Button>
      </div>
    </div>
  );
}

function DeployBox({ onJump }) {
  const { t } = useTranslation('nav');
  const [rememberChecked, setRememberChecked] = useState(localStorage.getItem('rememberChecked') === 'true');
  const history = useHistory();

  const handleRemember = (e) => {
    setRememberChecked(e.target.checked);
    localStorage.setItem('rememberChecked', e.target.checked);
  };

  const jumpToDeploy = () => {
    history.push(`/${containerRouterPrefix}/appMarket/oneClickDeploy`);
    onJump();
  };
  return (
    <div className='deploy_container'>
      <div style={{ display: 'flex', gap: '22px' }}>
        <div>
          <img src={fastDeploy} style={{ width: '128px', height: '138px' }} />
        </div>
        <div className='deploy_content'>
          <div style={{ fontSize: '20px', fontWeight: 'bold' }}>
            {t('tools.yamlDeploy')}
          </div>
          <div style={{ color: '#89939b' }}>
            {t('tools.yamlDeployDesc')}
          </div>
          <div style={{ color: '#89939b' }}>
            <Checkbox onChange={handleRemember} checked={rememberChecked}>
              {t('deploy.noReminder')}
            </Checkbox>
          </div>
        </div>
      </div>
      <div className='btn_deploy'>
        <Button
          className='primary_btn'
          onClick={jumpToDeploy}
        >
          {t('tools.goToDeploy')}
        </Button>
      </div>
    </div>
  );
}
export default function ContainerPlatFormIndex() {
  const { t } = useTranslation('nav');
  const [openNotification, setOpenNotification] = useState(false);
  const [openDeploy, setOpenDeploy] = useState(false);
  const themeStore = useStore('theme');
  const terminalStore = useStore('terminal');
  const userStore = useStore('user');
  const collapsedStore = useStore('collapsed');
  const [collapsed, setCollapsed] = useState(false);
  const [noPermissionsModalOpen, setNoPermissionsModalOpen] = useState(false);
  const [newTerminalModal, setNewTerminalModal] = useState(false);
  const [isShow, setIsShow] = useState(false);
  const [clusterOption, setClusterOption] = useState();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [clusterName, setClusterame] = useState(sessionStorage.getItem('cluster') || '');
  const [alertNum, setAlertNum] = useState(0);
  const handleCollapseTrigger = (status) => {
    setCollapsed(status);
    localStorage.setItem('collapsed', status);
    collapsedStore.$a.setCollapsed(status);
  };

  const namespacedLinkKeys = ['workload', 'network', 'configuration', 'userManage'];
  const namespacedChildKeys = ['limitRange', 'resourceQuota', 'pvc'];
  const handleChangeCluser = (value) => {
    setClusterame(value);
    sessionStorage.setItem('cluster', value);
    window.location.reload();
  };
  const filterOption = (input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase());
  const getClusters = async () => {
    const res = await getClustersList();
    if (res.status === ResponseCode.OK) {
      let option = [];
      if (Object.prototype.hasOwnProperty.call(res.data, 'info')) {
        Object.keys(res.data.info).map(item => {
          option.push({
            label: item,
            value: item,
          });
        });
      }
      setClusterOption(option);
    }
  };
  // 通知Popover关闭
  const closeNotification = () => {
    setOpenNotification(false);
  };
  const handleOpenChange = (newOpen) => {
    setOpenNotification(newOpen);
  };
  const handleCancel = () => {
    setOpenNotification(false);
  };

  const goToDeploy = () => {
    setOpenDeploy(false);
  };

  const handleDeploy = () => {
    if (localStorage.getItem('rememberChecked') === 'true') {
      history.push(
        `/${containerRouterPrefix}/appMarket/oneClickDeploy`,
      );
    } else {
      setOpenDeploy(true);
    }
  };

  const handleCloseDeploy = () => {
    setOpenDeploy(false);
  };

  useEffect(() => {
    getClusters();
  }, []);
  useEffect(() => {
    const handleLocationChange = () => {
      const [, , linkKey, childKey] = window.location.pathname.split('/');
      setIsShow(namespacedLinkKeys.includes(linkKey) ||
        namespacedChildKeys.includes(childKey));
    };
    handleLocationChange();
    const unlisten = history.listen(handleLocationChange);
    return () => {
      unlisten();
    };
  }, [history]);

  const handleClosePermissionModal = () => {
    setNoPermissionsModalOpen(false);
  };

  const handleCancelNewTerminal = () => {
    setNewTerminalModal(false);
  };

  const handleOkNewTerminal = () => {
    terminalStore.$a.setTerminalInit({ type: 'cluster', name: sessionStorage.getItem('cluster') });
    terminalStore.$a.setTerminalScreenSize('middle');
    terminalStore.$a.setTerminalOpen(false);
    setTimeout(() => {
      terminalStore.$a.setTerminalOpen(true);
    });
    setNewTerminalModal(false);
  };

  const checkPermissions = async () => {
    try {
      const permissionsRes = await checkTerminalAccess(`/user/${userStore.$s.user.name}/terminal`);
      if (permissionsRes.data.Code === ResponseCode.OK) {
        if (terminalStore.$s.isOpen) {
          setNewTerminalModal(true);
        } else {
          terminalStore.$a.setTerminalInit({ type: 'cluster', name: sessionStorage.getItem('cluster') });
          terminalStore.$a.setTerminalScreenSize('middle');
          terminalStore.$a.setTerminalOpen(true);
        }
      }
      if (permissionsRes.data.Code === ResponseCode.Forbidden) {
        setNoPermissionsModalOpen(true);
      }
      if (permissionsRes.data.Code === ResponseCode.InternalServerError) {
        messageApi.error(t('error.serverError', { ns: 'common' }));
      }
    } catch (e) {
      if (e.response?.data?.Code === ResponseCode.Forbidden) {
        setNoPermissionsModalOpen(true);
      } else if (e.response?.data?.Code === ResponseCode.InternalServerError) {
        messageApi.error(t('error.serverError', { ns: 'common' }));
      } else {
        messageApi.error(t('resourceDetail.terminalCheckFailed', { ns: 'common' }));
      }
    }
  };

  useEffect(() => {
    const collapsedStroage = localStorage.getItem('collapsed');
    if (collapsedStroage === 'true') {
      setCollapsed(true);
    }
  }, []);

  return <Fragment>
    <div className={`container_platform_box ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`}>
      <ToastMsg contextHolder={contextHolder} />
      <div className='container_platform_box_cluster_top'>
        <div className='container_platform_box_cluster_namespace'>
          <div className='clusterbar namespace_bar'>
            <img className='clusterbar_img' src={clusterIcon} alt="" />
            <p className='label'>{t('cluster.label')}：</p>
            <Select
              value={clusterName}
              showSearch
              filterOption={filterOption}
              popupMatchSelectWidth={false}
              popupClassName='namespace_popup_options'
              options={clusterOption}
              onChange={handleChangeCluser}
              getPopupContainer={() => document.getElementById('root_container') || document.body}
            />
          </div>
          {isShow && <NamespaceBar />}
        </div>
        <div className='nav_tools'>
          <Tooltip title={t('tools.kubectl')} placement="bottom">
            <CodeOutlined className='kubetcl_console_btn' style={{ color: themeStore.$s.theme === 'dark' ? '#fff' : '#333', fontSize: 20 }} onClick={checkPermissions} />
          </Tooltip>
          <Tooltip title={t('tools.deploy')} placement="bottom">
            <AppstoreAddOutlined style={{ color: themeStore.$s.theme === 'dark' ? '#fff' : '#333', fontSize: 20 }} className='tool_icon appStore' rotate={180} onClick={handleDeploy} />
          </Tooltip>
          <Modal open={openDeploy} onCancel={handleCloseDeploy} className='deployTopModal' footer={[]} mask={false}>
            <DeployBox onJump={goToDeploy} />
          </Modal>
          <Tooltip title={t('tools.alert')} placement="bottom">
            <AlertOutlined className='tool_icon nav_tools_alert' style={{ color: themeStore.$s.theme === 'dark' ? '#fff' : '#333', fontSize: 20 }} onClick={handleOpenChange} />
          </Tooltip>
          <p className='alertNum' style={{ color: 'red', fontSize: 12, display: alertNum > 0 ? 'block' : 'none' }}>{alertNum > 999 ? 999 : alertNum}</p>
          <p className='alertNumPlus' style={{ color: 'red', fontSize: 12, display: alertNum > 999 ? 'block' : 'none' }}>+</p>
          <Modal title={t('alert.title')} getContainer={false} forceRender open={openNotification} onCancel={handleCancel} className='alertTopModal' footer={[]} mask={false}>
            <AlertBox onJump={closeNotification} onAlertCount={setAlertNum} />
          </Modal>
        </div>
      </div>
      <div className='container_platform_box_content'>
        <div className={collapsed ? 'container_nav_bar collapse_self_bar' : 'container_nav_bar'}>
          <MenuBar collapsed={collapsed} handleCollapseChange={handleCollapseTrigger} />
        </div>
        <div className={collapsed ? 'container_content collapse_self_content' : 'container_content'}>
          {/* 后续根据路由取值隐藏该命名空间 */}

          <ContainerRouters />
        </div>
      </div>
      <NoPermissionsModal open={noPermissionsModalOpen} handleOpen={handleClosePermissionModal} />
      <TerminalNewModal open={newTerminalModal} handleOk={handleOkNewTerminal} cancelFn={handleCancelNewTerminal} />
    </div>
  </Fragment>;
}