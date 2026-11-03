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

import { useCallback, useEffect, useState, useRef } from 'openinula';
import { useTranslation } from 'react-i18next';
import { ExportOutlined, CopyOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { message, Checkbox, Tooltip } from 'antd';
import { getHelmDetailYaml, getHelmDetailYamlOnlyValue } from '@/api/containerApi';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { jsonToYaml, exportYamlOutPut } from '@/tools/utils';
import '@/styles/pages/helm.less';
import ToastMsg from '@/components/ToastMsg';

/**
 * Application YAML Tab Component
 *
 * @param {Object} props
 * @param {string} props.nameProps - Application name
 * @param {string} props.namespaceProps - Namespace
 */
export default function ApplicationYamlTab({ nameProps, namespaceProps }) {
  const { t } = useTranslation('application');
  const [helmYaml, setHelmYaml] = useState('');
  const [loading, setLoading] = useState(true);
  const [messageApi, contextHolder] = message.useMessage();
  const [isOnlyValue, setIsOnlyValue] = useState(false);
  const childCodeMirrorRef = useRef(null);

  const getHelmYaml = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getHelmDetailYaml(namespaceProps, nameProps);
      if (res.status === ResponseCode.OK) {
        setHelmYaml(res.data.data);
      }
    } catch {
      messageApi.error(t('messages.fetchYamlFailed'));
    } finally {
      setLoading(false);
    }
  }, [t, messageApi, nameProps, namespaceProps]);

  const handleCopyYaml = () => {
    copy(helmYaml);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleExportYaml = () => {
    if (!isOnlyValue) {
      exportYamlOutPut(nameProps, helmYaml);
    } else {
      exportYamlOutPut('Values', helmYaml);
    }
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  const onChangeYaml = useCallback(async (e) => {
    if (e.target.checked) {
      setIsOnlyValue(true);
      setLoading(true);
      try {
        const res = await getHelmDetailYamlOnlyValue(namespaceProps, nameProps);
        if (res.status === ResponseCode.OK) {
          setHelmYaml(jsonToYaml(JSON.stringify(res.data.data)));
        }
      } catch {
        messageApi.error(t('messages.fetchYamlFailed'));
      } finally {
        setLoading(false);
      }
    } else {
      setIsOnlyValue(false);
      getHelmYaml();
    }
  }, [t, getHelmYaml]);

  useEffect(() => {
    getHelmYaml();
  }, []);

  return (
    <div className="helm_tab_container container_margin_box tooltip_container_height">
      <ToastMsg contextHolder={contextHolder} />
      <div className="yaml_card">
        <div className="yaml_flex_box">
          <h3 className="box_title_h3" style={{ marginRight: '10px' }}>YAML</h3>
          <div className="yaml_tools">
            <div className="tool_word_group managementDetailExport" onClick={handleExportYaml}>
              <ExportOutlined className="common_antd_icon primary_color" />
              <span>{t('resourceYaml.export', { ns: 'common' })}</span>
            </div>
            <div className="tool_word_group managementDetailYamlCopy" onClick={handleCopyYaml}>
              <CopyOutlined className="common_antd_icon primary_color" />
              <span>{t('resourceYaml.copy', { ns: 'common' })}</span>
            </div>
          </div>
        </div>
        <div className="yaml_checkbox_row">
          <Checkbox onChange={onChangeYaml} disabled={loading}>{t('yaml.onlyValues')}</Checkbox>
          <Tooltip placement="topLeft" title={t('yaml.prompt')}>
            <QuestionCircleOutlined />
          </Tooltip>
        </div>
      </div>
      <div className="yaml_space_box">
        {!loading && <CodeMirrorEditor yamlData={helmYaml} isEdit={false} ref={childCodeMirrorRef} />}
      </div>
    </div>
  );
}