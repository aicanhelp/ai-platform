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

import { Fragment, useEffect, useState, useStore, useMemo, useRef } from 'openinula';
import { ResponseCode } from '@/common/constants';
import Dayjs from 'dayjs';
import { EditOutlined, InfoCircleFilled } from '@ant-design/icons';
import { message, Switch, Button } from 'antd';
import { getHelmManageStateKey, getHelmManageStateIcon } from '@/utils/common';
import { useTranslation } from 'react-i18next';
import { getExtendPageList, updateExtendPageItemState } from '@/api/containerApi';
import '@/styles/pages/extend.less';
import NoExtendManage from '@/assets/images/noExtendManage.png';
import ToastMsg from '@/components/ToastMsg';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function ExtendDetailInfo({
  extendName,
  extendDetailDataProps,
  isViewMode,
  handleEditFn,
  handleOkRefreshFn,
  handlePluginShow,
}) {
  const { t } = useTranslation('application');
  const [saveLoading, setSaveLoading] = useButtonLoading();

  const arrRef = useRef([]);
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  const [messageApi, contextHolder] = message.useMessage();

  const [pluginsList, setPluginsList] = useState([]);

  const themeStore = useStore('theme');

  const getExtendDetailExtendPageList = async () => {
    arrRef.current = []; // 每次加载重置，丢弃上次会话残留
    let resourcesList = extendDetailDataProps?.resources;
    let pluginName = [];
    if (resourcesList) {
      resourcesList.forEach(item => {
        if (item.kind === 'ConsolePlugin') {
          pluginName.push(item.name);
        }
      });
    }
    try {
      const results = await Promise.allSettled(pluginName.map(name => getExtendPageList(name)));
      const _pluginsArr = [];
      let failedCount = 0;
      results.forEach(result => {
        if (result.status === 'fulfilled' && result.value.status === ResponseCode.OK) {
          const d = result.value.data.data;
          _pluginsArr.push({
            name: d.pluginName,
            displayName: d.displayName ?? '',
            pluginName: d.pluginName ?? '',
            entrypoint: d.entrypoint ?? '',
            enabled: d.enabled,
          });
          arrRef.current.push({ name: d.pluginName, value: d.enabled });
        } else {
          failedCount++;
        }
      });
      if (failedCount) {
        messageApi.warning(t('plugin.fetchPartialFailed', { failed: failedCount }));
      }
      if (!mountedRef.current) { return; }
      setPluginsList(_pluginsArr);
      handlePluginShow(_pluginsArr.length > 0);
    } catch {
      if (!mountedRef.current) { return; }
      setPluginsList([]);
      handlePluginShow(false);
    }
  };

  const onChangeSwitch = (checked, record) => {
    setPluginsList(prev => prev.map(item =>
      (item.name === record.name ? { ...item, enabled: checked } : item),
    ));
    arrRef.current.forEach(item => {
      if (item.name === record.name) {
        item.value = checked;
      }
    });
  };

  const handSave = async () => {
    setSaveLoading(true);
    let successCount = 0;
    let failedCount = 0;
    try {
      for (let i = 0; i < pluginsList.length; i++) {
        const indexList = arrRef.current.filter(item => item.name === pluginsList[i].name);
        if (indexList.length) {
          try {
            const res = await updateExtendPageItemState(
              indexList[0].name,
              { pluginName: indexList[0].name, enabled: indexList[0].value },
            );
            if (res.status === ResponseCode.OK) {
              successCount++;
            } else {
              failedCount++;
            }
          } catch {
            failedCount++;
          }
        }
      }
      if (successCount === pluginsList.length && successCount > 0) {
        handleOkRefreshFn();
        handleEditFn(true);
      } else if (failedCount > 0) {
        messageApi.error(t('plugin.updatePartialFailed', { failed: failedCount, total: pluginsList.length }));
      }
    } finally {
      setSaveLoading(false);
    }
  };

  const handCancel = () => {
    handleEditFn(true);
    getExtendDetailExtendPageList();
  };

  useEffect(() => {
    getExtendDetailExtendPageList();
  }, []);

  // Get status key and translated label
  const statusKey = getHelmManageStateKey(extendDetailDataProps.info?.status);
  const statusLabel = t(`status.${statusKey}`);

  const baseInfoItems = useMemo(() => [
    { key: t('detail.appName'), value: extendName },
    {
      key: t('detail.status'),
      value: <span>{getHelmManageStateIcon(statusKey, 'extend_state_')}{statusLabel}</span>,
    },
    { key: t('detail.createTime'), value: Dayjs(extendDetailDataProps.info?.firstDeployed).format('YYYY-MM-DD HH:mm') },
    { key: t('detail.updateTime'), value: Dayjs(extendDetailDataProps.info?.lastDeployed).format('YYYY-MM-DD HH:mm') },
    { key: t('detail.namespace'), value: extendDetailDataProps.namespace },
    { key: t('detail.chartName'), value: extendDetailDataProps.chart?.metadata?.name ? extendDetailDataProps.chart.metadata.name : '--' },
    { key: t('detail.chartVersion'), value: extendDetailDataProps.chart?.metadata?.version ? extendDetailDataProps.chart.metadata.version : '--' },
  ], [extendDetailDataProps, t, statusKey, statusLabel]);

  return <Fragment>
    <ToastMsg contextHolder={contextHolder} />
    <div className={'extend_tab_container container_margin_box'} style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <div className="detail_card basic_info_card_extend">
        <div className="basic_info_card_left">
          <h3>{t('detail.basicInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={baseInfoItems} columns={1} />
          </div>
        </div>
        <div className='basic_info_card_right'>
          {extendDetailDataProps.info?.status !== 'failed' ? <div>
            <div className='title' >
              <h3 style={{ marginRight: '10px', display: 'flex', gap: '10px' }}>
                {t('plugin.title')}
                {(isViewMode && pluginsList.length !== 0) && <EditOutlined className='edit' onClick={() => handleEditFn(false)} />}
              </h3>
            </div>
            <div className='hint'>
              <InfoCircleFilled className='hint_icon' />
              <span className='hint_text'>{t('plugin.hint')}</span>
            </div>
            {pluginsList.length !== 0 ? pluginsList.map(item => {
              return <div key={item.name} className='extend_box'>
                <div className='extend_content'>
                  <div className='extend_content_group'>
                    <p className='group_name'>{item.name}</p>
                    <Switch disabled={isViewMode} checked={item.enabled} onChange={e => onChangeSwitch(e, item)} />
                  </div>
                  <div className='extend_content_group extend_plugin_meta'>
                    <p className='group_info extend_plugin_meta_row'>
                      <span className='plugin_meta_label'>{t('plugin.pluginNameLabel')}</span>
                      <span className='plugin_meta_value'>{item.pluginName || '--'}</span>
                    </p>
                    <p className='group_info extend_plugin_meta_row'>
                      <span className='plugin_meta_label'>{t('plugin.displayNameLabel')}</span>
                      <span className='plugin_meta_value'>{item.displayName || '--'}</span>
                    </p>
                    <p className='group_info extend_plugin_meta_row'>
                      <span className='plugin_meta_label'>{t('plugin.entrypointLabel')}</span>
                      <span className='plugin_meta_value'>{item.entrypoint || '--'}</span>
                    </p>
                  </div>
                </div>
              </div>;
            }) :
              <div>
                <div className='no_extend_box'>
                  <div className='no_extend_content'>
                    <img src={NoExtendManage} alt="" />
                    <p className='no_extend_content_info'>{t('plugin.noPlugin')}</p>
                  </div>
                </div>
              </div>
            }
            {!isViewMode && <div className='extend_boxButton'>
              <Button className='cancel_btn' onClick={handCancel} style={{ marginRight: '16px' }}>{t('actions.cancel')}</Button>
              <Button className='primary_btn' onClick={handSave} loading={saveLoading}>{t('actions.save')}</Button>
            </div>}
          </div> :
            <div>
              <div className='title' >
                <h3 className="box_title_h3" style={{ marginRight: '10px' }}>{t('plugin.title')}</h3>
              </div>
              <div className='no_extend_box'>
                <div className='no_extend_content'>
                  <img src={NoExtendManage} alt="" />
                  <p className='no_extend_content_info'>{t('plugin.deployFailed')}</p>
                </div>
              </div>
            </div>
          }
        </div>
      </div>
    </div>
  </Fragment>;
}