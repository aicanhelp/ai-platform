import { Button, message } from 'antd';
import { CopyOutlined, ExportOutlined } from '@ant-design/icons';
import { useHistory } from 'inula-router';
import '@/styles/components/addYaml.less';
import { createRef } from 'openinula';
import useLocalStorage from '@/hooks/useLocalStorage';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useState } from 'openinula';
import { exportYamlOutPut } from '@/tools/utils';
import copy from 'copy-to-clipboard';
import { useIntl } from 'inula-intl';
/**
 *
 * @param cancelFn 取消函数回调
 * @param parentYaml 接收yaml值
 * @param getYaml 回传yaml值
 * @param isShowBtn 是否显示底部按钮
 * @param ifEditCode 是否可编辑代码框
 * @param path 返回上页面路径
 * @param yamlName 导出文件名
 * @returns
 */

export default function AddYaml({ parentYaml, getYaml, isShowBtn, ifEditCode, path, yamlName }) {
  const intl = useIntl();
  const childCodeMirrorRef = createRef(null);
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [paramsYamlData, setParamsYamlData] = useState(parentYaml);
  const handleChangeAddYaml = (yaml) => {
    setParamsYamlData(yaml);
  };
  const exportAddYaml = () => {
    exportYamlOutPut(yamlName, paramsYamlData);
    messageApi.success(intl.formatMessage({ id: 'common.exportSuccess' }));
  };
  const handleCopyAddYaml = () => {
    copy(paramsYamlData);
    messageApi.success(intl.formatMessage({ id: 'common.copySuccess' }));
  };
  const submitAddYaml = () => {
    const isStart = false;
    getYaml(isStart, paramsYamlData);
  };
  const submitYamlStasrt = () => {
    const isStart = true;
    getYaml(isStart, paramsYamlData);
  };
  const cancelFn = () => {
    setParamsYamlData('');
    history.push(path);
  };

  // Extract the CodeMirrorEditor component to avoid nested ternary
  const renderCodeMirror = () => {
    if (isShowBtn) {
      return (
        <div className="code_box">
          <CodeMirrorEditor
            isEdit={ifEditCode}
            yamlData={paramsYamlData}
            changeYaml={handleChangeAddYaml}
            ref={childCodeMirrorRef}
          />
        </div>
      );
    }

    if (paramsYamlData) {
      return (
        <div className="code_box">
          <CodeMirrorEditor
            isEdit={ifEditCode}
            yamlData={paramsYamlData}
            changeYaml={handleChangeAddYaml}
            ref={childCodeMirrorRef}
          />
        </div>
      );
    }

    return null;
  };

  return (
    <div className="addYaml">
      {contextHolder}
      <div className="addYamlContent" style={{ padding: isShowBtn ? '32px' : '0 32px 32px' }}>
        <div className="addYamlTop">
          <p className="addYamlTopTitle">YAML</p>
          <div className="yaml_tools defaultAddModal">
            <div className="tool_word_group deployExport" onClick={exportAddYaml}>
              <ExportOutlined className="common_antd_icon primary_color defaultAddModal" />
              <span>{intl.formatMessage({ id: 'common.export' })}</span>
            </div>
            <div className="tool_word_group deployCopy" onClick={handleCopyAddYaml}>
              <CopyOutlined className="common_antd_icon primary_color defaultAddModal" />
              <span>{intl.formatMessage({ id: 'common.copy' })}</span>
            </div>
          </div>
        </div>
        {renderCodeMirror()}
      </div>
      {isShowBtn ? (
        <div className="addYamlBtns">
          <Button className="cancel_btn" onClick={cancelFn}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Button className="primary_btn ml" onClick={submitAddYaml}>
            {intl.formatMessage({ id: 'common.create' })}
          </Button>
          <Button className="primary_btn ml" onClick={submitYamlStasrt}>
            {intl.formatMessage({ id: 'ray.cluster.createAndStart' })}
          </Button>
        </div>
      ) : (
        ''
      )}
    </div>
  );
}
