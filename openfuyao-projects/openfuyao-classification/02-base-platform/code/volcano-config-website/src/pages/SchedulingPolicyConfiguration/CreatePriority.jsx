/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/createPriority.less';
import { useState, createRef } from 'openinula';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { message, Button } from 'antd';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useHistory } from 'inula-router';
import { yamlTojson, exportYamlOutPut } from '@/tools/utils';
import { priorityClassesExample } from '@/common/exampleYaml';
import { createSpecifiedPriorityClasses } from '@/api/schedulingApi';
import { useIntl } from 'inula-intl';

export default function CreatePriority() {
  const intl = useIntl();
  const history = useHistory();

  const [serviceAccountYaml, setServiceAccountYaml] = useState(priorityClassesExample);

  const [messageApi, contextHolder] = message.useMessage();

  const childCodeMirrorRef = createRef(null);

  const handleCopyYaml = () => {
    copy(serviceAccountYaml);
    messageApi.success('复制成功！');
  };

  const handleChangeYaml = (str) => {
    setServiceAccountYaml(str);
  };

  const handleSaveYaml = async () => {
    // YAML转json
    let errorCount = 0;
    let yamlJson = '';
    try {
      yamlJson = yamlTojson(serviceAccountYaml);
    } catch (e) {
      messageApi.error(`YAML格式不规范!${e.message}`);
      errorCount++;
    }
    if (!errorCount) {
      try {
        const res = await createSpecifiedPriorityClasses(yamlTojson(serviceAccountYaml));
        if (res.status === ResponseCode.Created) {
          messageApi.success('创建成功');
          setTimeout(() => {
            history.push(`/SchedulingPolicyConfiguration/priorityClass`);
          }, 2000);
        }
      } catch (error) {
        if (error.response.status === 403) {
          message.error('创建失败，当前用户没有操作权限，请联系管理员添加权限!');
        } else {
          message.error(`创建失败！${error.response.data.message}`);
        }
      }
    }
  };
  const handleResetCode = () => {
    setServiceAccountYaml(priorityClassesExample);
    childCodeMirrorRef.current.resetCodeEditor(priorityClassesExample);
  };

  const handleCancle = () => {
    history.go(-1);
  };

  const exportYaml = () => {
    exportYamlOutPut('PriorityClass', serviceAccountYaml);
    messageApi.success('导出成功');
  };

  return (
    <div className="child_content withBread_content">
      {contextHolder}

      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'scheduling.schedulingPolicyConfig' }), disabled: true },
          { title: 'PriorityClass', path: '/SchedulingPolicyConfiguration/priorityClass' },
          { title: intl.formatMessage({ id: 'common.details' }), disabled: true },
        ]}
      />
      <div className="tab_container container_margin_box normal_container_height">
        <div className="yaml_card">
          <div className="yaml_flex_box">
            <h3>YAML</h3>
            <div className="yaml_tools CreatePriority">
              <div className="tool_word_group" onClick={exportYaml}>
                <ExportOutlined className="common_antd_icon primary_color" />
                <span>{intl.formatMessage({ id: 'common.export' })}</span>
              </div>

              <div className="tool_word_group CreatePriority" onClick={handleCopyYaml}>
                <CopyOutlined className="common_antd_icon primary_color" />
                <span>{intl.formatMessage({ id: 'common.copy' })}</span>
              </div>
            </div>
          </div>
        </div>
        <div className="yaml_space_box CreatePriority">
          <CodeMirrorEditor yamlData={serviceAccountYaml} changeYaml={handleChangeYaml} ref={childCodeMirrorRef} />
          <div className="btn_footer">
            <Button className="cancel_btn" onClick={handleCancle}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className="cancel_btn" onClick={handleResetCode}>
              {intl.formatMessage({ id: 'common.reset' })}
            </Button>
            <Button className="primary_btn" onClick={handleSaveYaml}>
              {intl.formatMessage({ id: 'common.confirm' })}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
