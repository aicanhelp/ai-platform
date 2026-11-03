/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useState, useRef, useEffect, Fragment } from 'openinula';
import { useTranslation } from 'react-i18next';
import { ExportOutlined, CopyOutlined, EditOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { message, Button } from 'antd';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { exportYamlOutPut, yamlTojson, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import { ResponseCode } from '@/common/constants';
import ToastMsg from '../ToastMsg';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function ResourceYaml({
  yamlProps,
  readOnly,
  handleEditFn,
  updateFn,
  refreshFn,
  clusterScoped = false,
}) {
  const { t } = useTranslation('common');
  const [workloadYaml, setWorkloadYaml] = useState(yamlProps);
  const [saveLoading, setSaveLoading] = useButtonLoading();
  const [yamlSourceRevision, setYamlSourceRevision] = useState(0);
  const [messageApi, contextHolder] = message.useMessage();
  const [fileName, setFileName] = useState('');
  const childCodeMirrorRef = useRef(null);
  const refreshTimerRef = useRef(null);

  useEffect(() => () => clearTimeout(refreshTimerRef.current), []);

  const handleCopyYaml = () => {
    copy(workloadYaml);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleChangeYaml = (yaml) => {
    setWorkloadYaml(yaml);
  };

  const showUpdateFailed = (error) => {
    messageApi.error(t('resourceYaml.updateFailed', {
      message: error.response?.data?.message ?? error.message ?? '',
      interpolation: { escapeValue: false },
    }));
  };

  const handleSaveYaml = async () => {
    let yamlJson;
    try {
      yamlJson = yamlTojson(workloadYaml);
    } catch (e) {
      messageApi.error(t('resourceYaml.invalidYaml', {
        message: e.message,
        interpolation: { escapeValue: false },
      }));
      return;
    }
    const passName = yamlJson?.metadata?.name ?? '';
    // 命名空间级：YAML 写了就用，没写则请求地址用 default，不改 YAML 正文。
    // 集群级：忽略 YAML 里的 namespace，正文原样提交，地址也不带命名空间。
    const passNamespace = yamlJson?.metadata?.namespace || '';
    setSaveLoading(true);
    try {
      const res = clusterScoped
        ? await updateFn(passName, yamlJson)
        : await updateFn(passNamespace || 'default', passName, yamlJson);
      if (res.status === ResponseCode.OK) {
        await showSuccessThen(messageApi, t('resourceYaml.updateSuccess'), () => {
          refreshFn();
          handleEditFn(true);
        });
      } else {
        showUpdateFailed({ response: res, message: res.data?.message });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        showUpdateFailed(error);
      }
      // 失败停留当前页，等错误提示展示完再重置yaml，避免toast随Tabs卸载被吞掉
      refreshTimerRef.current = setTimeout(() => {
        refreshFn();
      }, 3000);
    } finally {
      setSaveLoading(false);
    }
  };

  const handleResetCode = () => {
    setWorkloadYaml(yamlProps);
    childCodeMirrorRef.current.resetCodeEditor(yamlProps);
  };

  const exportYaml = () => {
    exportYamlOutPut(`${fileName}`, workloadYaml);
    messageApi.success(t('resourceYaml.exportSuccess'));
  };

  const handleCodeCancel = () => {
    handleEditFn(true);
    setWorkloadYaml(yamlProps);
  };

  useEffect(() => {
    setWorkloadYaml(yamlProps);
    setYamlSourceRevision((r) => r + 1);
  }, [yamlProps]);

  useEffect(() => {
    try {
      const detailData = yamlTojson(yamlProps);
      setFileName(detailData.metadata.name);
    } catch {
      setFileName(t('resourceYaml.defaultFileName'));
    }
  }, [yamlProps, t]);

  return <div className="tab_container container_margin_box normal_container_height">
    <ToastMsg contextHolder={contextHolder} />
    <div className="yaml_card">
      <div className="yaml_flex_box">
        <h3>{readOnly ? <Fragment>{t('resourceYaml.title')} <EditOutlined style={{ color: '#3f66f5' }} onClick={() => handleEditFn(false)} /> </Fragment> : t('resourceYaml.title')}</h3>
        <div className="yaml_tools">
          <div className="tool_word_group" onClick={exportYaml}>
            <ExportOutlined className="common_antd_icon primary_color" />
            <span>{t('resourceYaml.export')}</span>
          </div>
          <div className="tool_word_group" onClick={handleCopyYaml}>
            <CopyOutlined className="common_antd_icon primary_color" />
            <span>{t('resourceYaml.copy')}</span>
          </div>
        </div>
      </div>
    </div>
    <div className="yaml_space_box">
      <CodeMirrorEditor className={readOnly ? 'no_btn' : ''} yamlData={workloadYaml} changeYaml={handleChangeYaml} isEdit={!readOnly} yamlSourceRevision={yamlSourceRevision} ref={childCodeMirrorRef} />
      {!readOnly && <div className="btn_footer">
        <Button className="cancel_btn" onClick={handleCodeCancel}>{t('actions.cancel')}</Button>
        <Button className="cancel_btn" onClick={handleResetCode}>{t('actions.reset')}</Button>
        <Button className="primary_btn" onClick={handleSaveYaml} loading={saveLoading}>{t('actions.confirm')}</Button>
      </div>}
    </div>
  </div>;
}