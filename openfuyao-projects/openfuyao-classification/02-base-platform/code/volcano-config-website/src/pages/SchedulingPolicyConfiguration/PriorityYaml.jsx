/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useState, createRef, Fragment, useEffect, useRef } from 'openinula';
import { ExportOutlined, CopyOutlined, SearchOutlined, EditOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { message, Button } from 'antd';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { ResponseCode } from '@/common/constants';
import { yamlTojson, exportYamlOutPut, jsonToYaml } from '@/tools/utils';
import { updateSpecifiedPriorityClasses } from '@/api/schedulingApi';
import { getSpecifiedPriorityClasses } from '@/api/schedulingApi';
import { useHistory } from 'inula-router';
import { useIntl } from 'inula-intl';

export default function PriorityYaml({ priorityName, details, isEdit, readOnly, handleEditFn, refreshFn }) {
  const intl = useIntl();
  const history = useHistory();
  const [serviceAccountYaml, setServiceAccountYaml] = useState(details);
  const [messageApi, contextHolder] = message.useMessage();
  const yamlRef = useRef(null);

  const handleCopyYaml = () => {
    copy(serviceAccountYaml);
    messageApi.success(intl.formatMessage({ id: 'yaml.copySuccess' }));
  };

  const exportYaml = () => {
    exportYamlOutPut(`${priorityName}`, serviceAccountYaml);
    messageApi.success(intl.formatMessage({ id: 'yaml.exportSuccess' }));
  };

  const handleChangeYaml = (yaml) => {
    setServiceAccountYaml(yaml);
  };

  const handleSaveYaml = async () => {
    let yamlJson = '';
    let errorCount = 0;
    let passName = '';
    try {
      yamlJson = yamlTojson(serviceAccountYaml);
      passName = yamlJson.metadata.name;
    } catch (e) {
      messageApi.error(`${intl.formatMessage({ id: 'yaml.invalidFormat' })}${e.message}`);
      errorCount++;
    }
    if (!errorCount) {
      try {
        const res = await updateSpecifiedPriorityClasses(passName, yamlTojson(serviceAccountYaml));
        if (res.status === ResponseCode.OK) {
          messageApi.success(intl.formatMessage({ id: 'yaml.modifySuccess' }));
          setTimeout(() => {
            history.push('/SchedulingPolicyConfiguration/PriorityClass');
          }, 1000);
        }
      } catch (error) {
        if (error.response.status === 403) {
          messageApi.error(intl.formatMessage({ id: 'common.permissionDenied' }));
        } else {
          messageApi.error(`${intl.formatMessage({ id: 'yaml.modifyFailed' })}${error.response.data.message}`);
        }
      }
    }
  };

  const handleResetCode = () => {
    setServiceAccountYaml(details);
    yamlRef.current.resetCodeEditor(details);
  };

  const handleCodeCancel = () => {
    setServiceAccountYaml(details);
    handleEditFn(true);
  };

  const getSpecifiedPriorityClassesInfo = async (name) => {
    const res = await getSpecifiedPriorityClasses(name);
    setServiceAccountYaml(jsonToYaml(JSON.stringify(res.data)));
  };

  useEffect(() => {
    getSpecifiedPriorityClassesInfo(priorityName);
  }, [priorityName]);

  return (
    <div className="tab_container container_margin_box normal_container_height" style={{ margin: '20px 32px' }}>
      {contextHolder}
      <div className="yaml_card">
        <div className="yaml_flex_box">
          <h3>
            {readOnly ? (
              <Fragment>
                YAML <EditOutlined style={{ color: '#3f66f5' }} onClick={() => handleEditFn(false)} />{' '}
              </Fragment>
            ) : (
              'YAML'
            )}
          </h3>
          <div className="yaml_tools PriorityYaml">
            <div className="tool_word_group" onClick={exportYaml}>
              <ExportOutlined className="common_antd_icon primary_color" />
              <span>{intl.formatMessage({ id: 'common.export' })}</span>
            </div>
            <div className="tool_word_group PriorityYaml" onClick={handleCopyYaml}>
              <CopyOutlined className="common_antd_icon primary_color" />
              <span>{intl.formatMessage({ id: 'common.copy' })}</span>
            </div>
          </div>
        </div>
      </div>
      <div className="yaml_space_box PriorityYaml">
        <CodeMirrorEditor
          className={isEdit ? 'no_btn' : ''}
          yamlData={serviceAccountYaml}
          changeYaml={handleChangeYaml}
          isEdit={!readOnly}
          ref={yamlRef}
        />
        {!readOnly && (
          <div className="btn_footer">
            <Button className="cancel_btn" onClick={handleCodeCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={handleResetCode}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={handleSaveYaml}>
              {intl.formatMessage({ id: 'common.confirm' })}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
