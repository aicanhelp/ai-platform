/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Upload, Button, message, Modal } from 'antd';
import { ResponseCode } from '@/common/constants';
import '@/styles/applicationMarket/index.less';
import { useState, useStore } from 'openinula';
import { CloudUploadOutlined } from '@ant-design/icons';
import uploadPackage from '@/assets/images/uploadPackage.png';
import axios from 'axios';
import { checkSessionStorage } from '@/tools/utils';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function CreateVersion({ openAdd, onCancelFn, onRefresh }) {
  const [uploading, setUploading] = useButtonLoading();
  const [fileListVersion, setFileListVersion] = useState([]);
  const [errorInfo, setErrorInfo] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const { t } = useTranslation(['appMarket', 'common']);
  const props = {
    onRemove: () => {
      setErrorInfo('');
      setFileListVersion([]); // 清空文件列表
    },
    beforeUpload: (file) => {
      setErrorInfo('');
      const isLt10M = file.size / 1024 / 1024 < 2;
      if (!isLt10M) {
        messageApi.error(t('modal.fileSizeError'));
        return false;
      }
      const isGzipFile = file.name.endsWith('.tgz') || file.name.endsWith('.gz');
      if (isGzipFile) {
        setFileListVersion([file]); // 替换文件列表，确保只有一个文件
        return false; // 创建版本取消默认上传行为
      } else {
        messageApi.error(t('modal.fileFormatError'));
        return false;
      }
    },
    fileList: fileListVersion,
  };

  const handleUpload = async () => {
    const versionFormData = new FormData();
    fileListVersion.forEach((file) => {
      versionFormData.append('chart', file);
    });
    setUploading(true);
    let clusterName = checkSessionStorage('cluster') ? sessionStorage.getItem('cluster') : 'host';
    try {
      await axios({
        method: 'post',
        url: `/clusters/${clusterName}/rest/marketplace/v1beta1/helm-charts`,
        data: versionFormData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setFileListVersion([]);
      messageApi.success(t('modal.fileUploadSuccess'));
      onRefresh();
    } catch (error) {
      setErrorInfo(error?.response?.data?.msg);
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(error?.response?.data?.msg || t('modal.fileUploadFailed'));
      }
    } finally {
      setUploading(false);
    }
  };

  const handleCancel = () => {
    setErrorInfo('');
    setFileListVersion([]); // 清空文件列表
    onCancelFn();
  };
  return (
    <Modal
      onCancel={handleCancel}
      width={720}
      open={openAdd}
      title={t('modal.addVersion')}
      footer={[
        <Button
          className='cancel_btn'
          onClick={handleCancel}
        >
          {t('modal.cancel')}
        </Button>,
        <Button
          className='primary_btn'
          loading={uploading}
          disabled={!fileListVersion.length}
          onClick={handleUpload}
        >
          {t('actions.confirm', { ns: 'common' })}
        </Button>,
      ]}
    >
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className='upload_container versionUpload'>
        <div>
          <img src={uploadPackage} />
        </div>
        <div className='versionUpload'>
          <Upload {...props}>
            <Button icon={<CloudUploadOutlined />} type='text' style={{ padding: '0 8px', borderRadius: 6 }}>
              {t('modal.uploadFile')}
            </Button>
            <div className='versionUpload' style={{ color: '#89939b' }}>
              {t('modal.fileLimit')}
            </div>
          </Upload>
        </div>
      </div>
      <div className='versionUpload' style={{ color: '#e7434a' }}>{errorInfo}</div>
    </Modal>
  );
}
