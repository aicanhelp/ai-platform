/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button, message, Upload } from 'antd';
import { DownloadOutlined, UploadOutlined } from '@ant-design/icons';
import '@/styles/components/uploadConfig.less';
import { Fragment } from 'openinula';
import useLocalStorage from '@/hooks/useLocalStorage';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useState, createRef } from 'openinula';
import fileimg from '@/assets/images/file.png';
import { jsonToYaml, yamlTojson, exportYamlOutPut } from '@/tools/utils';
import { useEffect } from 'react';
import { clusterSample, jobSample, serviceSample } from '@/common/constants';
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
 * @param size 上传文件大小(必传)
 * @param type 上传文件类型(必传)
 * @param downloadType 下载文件类型(必传)
 * @returns
 */
export default function UploadConfig({
  open,
  title,
  cancelText,
  downloadType,
  downloadText,
  confirmText,
  cancelFn,
  uploadText,
  uploadtip,
  size,
  type,
  propsServiceUpload,
}) {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [fileList, setFileList] = useState([]);
  const childCodeMirrorRef = createRef(null);
  const [messageApi, contextHolder] = message.useMessage();
  const [isClickUpload, setIsClickUpload] = useState(true);
  const [isShowCode, setIsShowCode] = useState(false);
  const [paramsYamlData, setParamsYamlData] = useState();
  const props = {
    name: 'file',
    onRemove: (file) => {
      const index = fileList.indexOf(file);
      const newFileList = fileList.slice();
      newFileList.splice(index, 1);
      setFileList(newFileList);
      setIsClickUpload(true);
      setIsShowCode(false);
    },
    beforeUpload: (file) => {
      let uploadfile = [];
      const isType = file.name.split('.').pop() === type;
      const isLtSize = file.size / 1024 / 1024 <= size;
      if (!isType || !isLtSize) {
        messageApi.error(uploadtip);
      } else {
        setIsClickUpload(false);
      }
      if (fileList.length === 0) {
        setFileList([...fileList, file]);
      } else {
        setFileList([]);
        setFileList([...uploadfile, file]);
      }
      return false;
    },
    progress: {
      strokeColor: {
        '0%': '#108ee9',
        '100%': '#87d068',
      },
      strokeWidth: 3,
      format: (percent) => percent && `${parseFloat(percent.toFixed(2))}%`,
    },
  };
  const handleUploadChange = ({ fileList }) => {
    const file = fileList[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        setIsShowCode(false);
        const content = e.target.result;
        const parsedData = yamlTojson(content);
        setParamsYamlData(jsonToYaml(JSON.stringify(parsedData)));
        setIsShowCode(true);
        messageApi.success(intl.formatMessage({ id: 'upload.yamlParseSuccess' }));
      } catch (error) {
        messageApi.error(intl.formatMessage({ id: 'upload.yamlParseFail' }));
      }
    };
    reader.readAsText(file.originFileObj);
  };
  const exportAddYaml = () => {
    let yaml = '';
    if (downloadType === 'raycluster') {
      yaml = clusterSample;
    } else if (downloadType === 'rayjob') {
      yaml = jobSample;
    } else {
      yaml = serviceSample;
    }
    exportYamlOutPut(`${downloadType}-sample`, yaml);
    messageApi.success(intl.formatMessage({ id: 'upload.downloadSuccess' }));
  };
  useEffect(() => {
    if (!open) {
      setParamsYamlData('');
      setIsShowCode(false);
    }
  }, [open]);
  return (
    <Modal
      width={800}
      getContainer={false}
      className={`modal_flex_uploadConfig ${currentTheme === 'dark' ? 'dark_box' : ''}`}
      open={open}
      title={title}
      onCancel={cancelFn}
      destroyOnClose
      footer={[
        <Fragment>
          <Button className="cancel_btn" onClick={cancelFn}>
            {cancelText}
          </Button>
          <Button className="primary_btn" disabled={isClickUpload} onClick={() => propsServiceUpload(paramsYamlData)}>
            {confirmText}
          </Button>
        </Fragment>,
      ]}
    >
      {contextHolder}
      <div className="downloadContent">
        <DownloadOutlined className="textColor" />
        <p className="textColor downloadText" onClick={exportAddYaml}>
          {downloadText}
        </p>
      </div>
      <div className="uploadContent">
        <img src={fileimg} alt="" className="uploadContentImg" />
        <div className="uploadContentSelect">
          <Upload {...props} maxCount={1} onChange={handleUploadChange}>
            <Button icon={<UploadOutlined />}>{uploadText}</Button>
          </Upload>
          <p className="uploadContentSelectTip">{uploadtip}</p>
        </div>
      </div>
      {isShowCode ? (
        <CodeMirrorEditor
          className="uploadServiceYaml"
          yamlData={paramsYamlData}
          ref={childCodeMirrorRef}
          isEdit={false}
        />
      ) : (
        <p className="pleaserSelect">{intl.formatMessage({ id: 'upload.selectYamlFile' })}</p>
      )}
    </Modal>
  );
}
