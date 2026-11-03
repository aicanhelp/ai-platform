/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button, message } from 'antd';
import { CopyOutlined, ExportOutlined } from '@ant-design/icons';
import '@/styles/components/addYamlModal.less';
import { Fragment, createRef } from 'openinula';
import useLocalStorage from '@/hooks/useLocalStorage';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useState } from 'openinula';
import { exportYamlOutPut } from '@/tools/utils';
import copy from 'copy-to-clipboard';
import { useIntl } from 'inula-intl';
/**
 *
 * @param title 对话框标题
 * @param open 控制对话框打开
 * @param cancelFn 取消函数回调
 * @param content 对话框内容 数组！
 * @param confirmFn 确认回调
 * @param confirmText 确认按钮文本 默认可不填
 * @param showCheck 是否展示检查框(默认不展示)
 * @param isCheck 是否选中(默认不展示)
 * @param checkFn 选中回调(默认不展示)
 * @returns
 */
export default function AddYamlModal({ open, parentYaml, cancelFn, getYaml }) {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const [paramsYamlData, setParamsYamlData] = useState(parentYaml);
  const childCodeMirrorRef = createRef(null);
  const handleChangeAddModalYaml = (yaml) => {
    setParamsYamlData(yaml);
  };
  const exportAddModalYaml = () => {
    exportYamlOutPut('Values', paramsYamlData);
    messageApi.success(intl.formatMessage({ id: 'common.exportSuccess' }));
  };
  const handleCopyAddModalYaml = () => {
    copy(paramsYamlData);
    messageApi.success(intl.formatMessage({ id: 'common.copySuccess' }));
  };
  const submitAddModalYaml = () => {
    getYaml(paramsYamlData);
  };
  return (
    <Modal
      getContainer={false}
      width={1200}
      height={720}
      closable={false}
      className={`modal_flex_addyaml ${currentTheme === 'dark' ? 'dark_box' : ''}`}
      open={open}
      title={''}
      onCancel={cancelFn}
      destroyOnClose
      footer={[
        <Fragment>
          <Button onClick={cancelFn} className="cancel_btn">
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Button className="primary_btn" onClick={submitAddModalYaml}>
            {intl.formatMessage({ id: 'common.ok' })}
          </Button>
        </Fragment>,
      ]}
    >
      {contextHolder}
      <div className="addYamlContent">
        <p className="addYamlContentTitle">YAML</p>
        <div className="yaml_tools defaultAdd">
          <div className="tool_word_group deployExport" onClick={exportAddModalYaml}>
            <ExportOutlined className="common_antd_icon primary_color defaultAdd" />
            <span>{intl.formatMessage({ id: 'common.export' })}</span>
          </div>
          <div className="tool_word_group deployCopy" onClick={handleCopyAddModalYaml}>
            <CopyOutlined className="common_antd_icon primary_color defaultAdd" />
            <span>{intl.formatMessage({ id: 'common.copy' })}</span>
          </div>
        </div>
      </div>
      <CodeMirrorEditor
        className="addYamlCode"
        yamlData={paramsYamlData}
        changeYaml={handleChangeAddModalYaml}
        ref={childCodeMirrorRef}
      />
    </Modal>
  );
}
