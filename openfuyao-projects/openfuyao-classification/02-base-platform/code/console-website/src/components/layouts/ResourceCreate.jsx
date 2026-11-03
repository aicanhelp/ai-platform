/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { useState, useRef } from 'openinula';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { message, Button } from 'antd';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useHistory } from 'inula-router';
import { useTranslation } from 'react-i18next';
import { yamlTojson, exportYamlOutPut, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import '@/styles/pages/podDetail.less';
import ToastMsg from '@/components/ToastMsg';
import useButtonLoading from '@/hooks/useButtonLoading';

/**
 * 通用 WorkloadCreate 组件
 * @param {string} type 类型
 * @param {string} yamlExample 示例 yaml
 * @param {Function} submitApi 命名空间资源: (namespace, yamlJson) => Promise；clusterScoped 时: (yamlJson) => Promise
 * @param {string} jumpPath 成功创建后跳转地址
 * @param {boolean} clusterScoped 为 true 时地址不带命名空间，仅传入 yaml 调用 submitApi(yamlJson)
 */
export default function ResourceCreate({
  type = '',
  yamlExample = '',
  submitApi,
  jumpPath = '/',
  clusterScoped = false,
}) {
  const history = useHistory();
  const { t } = useTranslation('common');
  const [yamlStr, setYamlStr] = useState(yamlExample);
  const [saveLoading, setSaveLoading] = useButtonLoading();
  const [messageApi, contextHolder] = message.useMessage();
  const childCodeMirrorRef = useRef(null);

  const handleCopyYaml = () => {
    copy(yamlStr);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleChangeYaml = (str) => {
    setYamlStr(str);
  };

  const showCreateFailed = (error) => {
    messageApi.error(t('resourceYaml.createFailed', {
      message: error.response?.data?.message ?? error.message ?? '',
      interpolation: { escapeValue: false },
    }));
  };

  const handleSaveYaml = async () => {
    let yamlJson;
    try {
      yamlJson = yamlTojson(yamlStr);
    } catch (e) {
      messageApi.error(t('resourceYaml.invalidYaml', {
        message: e.message,
        interpolation: { escapeValue: false },
      }));
      return;
    }
    const writtenNamespace = yamlJson?.metadata?.namespace || '';
    setSaveLoading(true);
    try {
      const res = clusterScoped
        ? await submitApi(yamlJson)
        : await submitApi(writtenNamespace || 'default', yamlJson);
      if (res.status === ResponseCode.Created) {
        await showSuccessThen(messageApi, t('resourceYaml.createSuccess'), () => {
          history.push(jumpPath);
        });
      } else {
        showCreateFailed({ response: res, message: res.data?.message });
      }
    } catch (error) {
      if (error.response?.status === ResponseCode.Forbidden) {
        forbiddenMsg(messageApi, error);
      } else {
        showCreateFailed(error);
      }
    } finally {
      setSaveLoading(false);
    }
  };

  const handleResetCode = () => {
    setYamlStr(yamlExample);
    childCodeMirrorRef.current.resetCodeEditor(yamlExample);
  };

  const exportYaml = () => {
    exportYamlOutPut(type, yamlStr);
    messageApi.success(t('resourceYaml.exportSuccess'));
  };

  return <div className="tab_container container_margin_box normal_container_height">
    <ToastMsg contextHolder={contextHolder} />
    <div className="yaml_card">
      <div className="yaml_flex_box">
        <h3>{t('resourceYaml.title')}</h3>
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
      <CodeMirrorEditor
        yamlData={yamlStr}
        changeYaml={handleChangeYaml}
        ref={childCodeMirrorRef}
      />
      <div className="btn_footer">
        <Button className="cancel_btn" onClick={() => history.go(-1)}>{t('actions.cancel')}</Button>
        <Button className="cancel_btn" onClick={handleResetCode}>{t('actions.reset')}</Button>
        <Button className="primary_btn" onClick={handleSaveYaml} loading={saveLoading}>{t('actions.confirm')}</Button>
      </div>
    </div>

  </div>;
}
