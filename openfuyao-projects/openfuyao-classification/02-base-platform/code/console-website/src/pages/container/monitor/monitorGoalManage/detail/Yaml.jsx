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
import { useState, useRef, Fragment, useEffect } from 'openinula';
import { useTranslation } from 'react-i18next';
import { ExportOutlined, CopyOutlined, EditOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { message, Button } from 'antd';
import ToastMsg from '@/components/ToastMsg';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { updateResourceInstanceYamlData } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import { yamlTojson, exportYamlOutPut, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { disabledModifyMonitorServiceCr } from '@/common/constants';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function Yaml({ serviceMonitorYamlProps, prefixObjProps, readOnly, handleEditFn, refreshFn }) {
  const { t } = useTranslation('monitoring');
  const [serviceYaml, setServiceYaml] = useState(serviceMonitorYamlProps);
  const [saveLoading, setSaveLoading] = useButtonLoading();
  const [messageApi, contextHolder] = message.useMessage();
  const [hidden, setHidden] = useState(false);
  const [fileName, setFileName] = useState('');
  const childCodeMirrorRef = useRef(null);

  const handleCopyYaml = () => {
    copy(serviceYaml);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleChangeYaml = (yaml) => {
    setServiceYaml(yaml);
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
    let yamlJson = '';
    let errorCount = 0;
    let passNamespace = '';
    let passName = '';
    try {
      yamlJson = yamlTojson(serviceYaml);
      passNamespace = yamlTojson(serviceMonitorYamlProps)?.metadata?.namespace ?? ''; // 默认取之前的命名空间
      passName = yamlTojson(serviceMonitorYamlProps)?.metadata?.name ?? '';
    } catch (e) {
      messageApi.error(`${t('messages.yamlFormatError')}${e.message}`);
      errorCount++;
    }
    if (!errorCount) {
      if (passNamespace) {
        try {
          const res = await updateResourceInstanceYamlData({ ...prefixObjProps, namespace: passNamespace }, passName, yamlJson);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.updateSuccess'), () => {
              refreshFn();
              handleEditFn(true);
            });
          }
        } catch (error) {
          if (error.response.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(`${t('messages.updateFailed')}${error.response.data.message}`);
          }
          // 失败停留当前页，等错误提示展示完再重置yaml，避免toast随Tabs卸载被吞掉
          setTimeout(() => {
            refreshFn();
          }, 3000);
        }
      }
    }
  };

  const exportYaml = () => {
    exportYamlOutPut(`${fileName}`, serviceYaml);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  const handleResetCode = () => {
    setServiceYaml(serviceMonitorYamlProps);
    childCodeMirrorRef.current.resetCodeEditor(serviceMonitorYamlProps);
  };

  const handleCodeCancel = () => {
    handleEditFn(true);
    setServiceYaml(serviceMonitorYamlProps);
  };

  const handleEditJudge = () => {
    try {
      const detailData = yamlTojson(serviceMonitorYamlProps);
      if (detailData?.metadata?.name && !disabledModifyMonitorServiceCr.includes(detailData.metadata.name)) {
        handleEditFn(false);
      }
    } catch (e) {
      messageApi.error(`${t('messages.yamlFormatError')}${e.message}`);
    }
  };

  useEffect(() => {
    try {
      const detailData = yamlTojson(serviceMonitorYamlProps);
      setFileName(detailData?.metadata?.name ?? '');
      if (detailData?.metadata?.name && disabledModifyMonitorServiceCr.includes(detailData.metadata.name)) {
        setHidden(true);
        handleEditFn(true);
      }
    } catch (e) {
      messageApi.error(`${t('messages.yamlFormatError')}${e.message}`);
    }
  }, [serviceMonitorYamlProps]);

  return <div className="tab_container container_margin_box normal_container_height">
    <ToastMsg contextHolder={contextHolder} />
    <div className="yaml_card">
      <div className="yaml_flex_box">
        {hidden ?
          <h3>{t('serviceMonitor.yaml.title')}</h3> :
          <h3>{readOnly ? <Fragment>{t('serviceMonitor.yaml.title')} <EditOutlined style={{ color: '#3f66f5' }} onClick={handleEditJudge} /></Fragment> : t('serviceMonitor.yaml.title')}</h3>
        }
        <div className="yaml_tools">
          <div className="tool_word_group yamlExport" onClick={exportYaml}>
            <ExportOutlined className="common_antd_icon primary_color" />
            <span>{t('serviceMonitor.yaml.export')}</span>
          </div>
          <div className="tool_word_group yamlCopy" onClick={handleCopyYaml}>
            <CopyOutlined className="common_antd_icon primary_color" />
            <span>{t('serviceMonitor.yaml.copy')}</span>
          </div>
        </div>
      </div>
    </div>
    <div className='yaml_space_box'>
      <CodeMirrorEditor className={readOnly ? 'no_btn' : ''} yamlData={serviceYaml} changeYaml={handleChangeYaml} isEdit={!readOnly} ref={childCodeMirrorRef} />
      {!readOnly && <div className="btn_footer">
        <Button className="cancel_btn" onClick={handleCodeCancel}>{t('serviceMonitor.yaml.cancel')}</Button>
        <Button className="cancel_btn" onClick={handleResetCode}>{t('serviceMonitor.yaml.reset')}</Button>
        <Button className="primary_btn" onClick={handleSaveYaml} loading={saveLoading}>{t('serviceMonitor.yaml.confirm')}</Button>
      </div>}
    </div>
  </div>;
}
