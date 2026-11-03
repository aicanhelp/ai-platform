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
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/podDetail.less';
import { useState, useRef, useStore } from 'openinula';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { message, Button } from 'antd';
import { addResourceInstanceYamlData } from '@/api/containerApi';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useHistory, useLocation } from 'inula-router';
import { yamlTojson, exportYamlOutPut, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { serviceMonitorYamlExample } from '@/common/exampleYaml';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function MonitorCreateService() {
  const { state: prefixObj } = useLocation();
  const history = useHistory();
  const { t } = useTranslation(['monitoring', 'common']);
  const [serviceMonitorYaml, setServiceMonitorYaml] = useState(serviceMonitorYamlExample);
  const [saveLoading, setSaveLoading] = useButtonLoading();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const childCodeMirrorRef = useRef(null);

  const handleCopyYaml = () => {
    copy(serviceMonitorYaml);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleChangeYaml = (str) => {
    setServiceMonitorYaml(str);
  };

  const exportYaml = () => {
    exportYamlOutPut('ServiceMonitor', serviceMonitorYaml);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  const handleSaveYaml = async () => {
    setSaveLoading(true);
    try {
      await handleSaveYamlInner();
    } finally {
      setSaveLoading(false);
    }
  };
  const handleSaveYamlInner = async () => {
    let yamlJson;
    try {
      yamlJson = yamlTojson(serviceMonitorYaml);
    } catch (e) {
      messageApi.error(t('resourceYaml.invalidYaml', {
        ns: 'common',
        message: e.message,
        interpolation: { escapeValue: false },
      }));
      return;
    }
    const writtenNamespace = yamlJson?.metadata?.namespace || '';
    try {
      const res = await addResourceInstanceYamlData(
        { ...prefixObj, namespace: writtenNamespace || 'default' },
        yamlJson,
      );
      if (res.status === ResponseCode.Created) {
        showSuccessThen(messageApi, t('messages.createSuccess'), () => {
          history.push(`/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor`);
        });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        messageApi.error(t('resourceYaml.createFailed', {
          ns: 'common',
          message: error.response?.data?.message ?? error.message ?? '',
          interpolation: { escapeValue: false },
        }));
      }
    }
  };

  const handleResetCode = () => {
    setServiceMonitorYaml(serviceMonitorYamlExample);
    childCodeMirrorRef.current.resetCodeEditor(serviceMonitorYamlExample);
  };

  return <div className="child_content">
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), path: `/${containerRouterPrefix}/monitor`, disabled: true },
      { title: t('module.targetList'), path: '/monitorGoalManage' },
      { title: t('serviceMonitor.title'), path: '/serviceMonitor' },
      { title: t('serviceMonitor.create'), path: '/create' },
    ]} />
    <div className="tab_container container_margin_box normal_container_height">
      <div className="yaml_card">
        <div className="yaml_flex_box">
          <h3>{t('serviceMonitor.yaml.title')}</h3>
          <div className="yaml_tools">
            <div className="tool_word_group monitorCreateServiceExport" onClick={exportYaml}>
              <ExportOutlined className="common_antd_icon primary_color" />
              <span>{t('serviceMonitor.yaml.export')}</span>
            </div>
            <div className="tool_word_group monitorCreateServiceCopy" onClick={handleCopyYaml}>
              <CopyOutlined className="common_antd_icon primary_color" />
              <span>{t('serviceMonitor.yaml.copy')}</span>
            </div>
          </div>
        </div>
      </div>
      <div className="yaml_space_box">
        <CodeMirrorEditor yamlData={serviceMonitorYaml} changeYaml={handleChangeYaml} ref={childCodeMirrorRef} />
        <div className="btn_footer">
          <Button className="cancel_btn" onClick={() => history.go(-1)}>{t('serviceMonitor.yaml.cancel')}</Button>
          <Button className="cancel_btn" onClick={handleResetCode}>{t('serviceMonitor.yaml.reset')}</Button>
          <Button className="primary_btn" onClick={handleSaveYaml} loading={saveLoading}>{t('serviceMonitor.yaml.confirm')}</Button>
        </div>
      </div>
    </div>
  </div>;
}
