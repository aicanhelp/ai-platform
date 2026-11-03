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
import '@/styles/applicationMarket/index.less';
import { useState, useStore } from 'openinula';
import { CloudUploadOutlined } from '@ant-design/icons';
import uploadPackage from '@/assets/images/uploadPackage.png';
import axios from 'axios';
import { getLimitPackageNumber } from '@/api/applicationMarketApi';
import { ResponseCode } from '@/common/constants';
import { checkSessionStorage } from '@/tools/utils';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function CreateApplicationPackage({ openAdd, onCancelFn }) {
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const { t } = useTranslation(['appMarket', 'common']);
  const [uploading, setUploading, uploadPending] = useButtonLoading();
  const [fileList, setFileList] = useState([]);
  const [errorInfo, setErrorInfo] = useState('');

  const props = {
    onRemove: () => {
      setErrorInfo('');
      setFileList([]); // 清空文件列表
    },
    beforeUpload: async (file) => {
      // 先进行数量限制
      try {
        const res = await getLimitPackageNumber();
        if (res.status === ResponseCode.OK) {
          if (res.data && res.data.data && res.data.data.total >= res.data.data.limit) {
            messageApi.error(t('modal.uploadLimitReached'));
            return false;
          }
        }
      } catch {
        messageApi.error(t('modal.checkUploadLimitFailed'));
        return false; // 校验接口失败时阻止文件加入列表，确保确认按钮保持禁用
      }
      setErrorInfo('');
      const isLt2M = file.size / 1024 / 1024 < 2;
      if (!isLt2M) {
        messageApi.error(t('modal.fileSizeError'));
        return false;
      }
      const isGzipFile = file.name.endsWith('.tgz') || file.name.endsWith('.gz');
      if (isGzipFile) {
        setFileList([file]); // 替换文件列表，确保只有一个文件
        return false; // 创建应用包取消默认上传行为
      } else {
        messageApi.error(t('modal.fileFormatError'));
        return false;
      }
    },
    fileList,
  };

  const handleUpload = async () => {
    const appFormData = new FormData();
    fileList.forEach((file) => {
      appFormData.append('chart', file);
    });
    setUploading(true);
    let clusterName = checkSessionStorage('cluster') ? sessionStorage.getItem('cluster') : 'host';
    try {
      await axios({
        method: 'post',
        url: `/clusters/${clusterName}/rest/marketplace/v1beta1/helm-charts`,
        data: appFormData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setFileList([]);
      messageApi.success(t('modal.fileUploadSuccess'));
      onCancelFn();
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        setErrorInfo(error.response?.data?.msg);
        messageApi.error(error.response?.data?.msg || t('modal.fileUploadFailed'));
      }
    } finally {
      setUploading(false);
    }
  };

  const handleCancel = () => {
    setFileList([]); // 清空上传文件列表
    setErrorInfo('');
    onCancelFn();
  };
  return (
    <Modal
      onCancel={handleCancel}
      open={openAdd}
      width={720}
      title={t('modal.addPackage')}
      footer={[
        <Button
          onClick={handleCancel}
          className='cancel_btn'
        >
          {t('modal.cancel')}
        </Button>,
        <Button
          className='primary_btn'
          onClick={handleUpload}
          loading={uploading}
          disabled={!fileList.length || uploadPending}
        >
          {t('actions.confirm', { ns: 'common' })}
        </Button>,
      ]}
    >
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className='upload_container'>
        <div>
          <img src={uploadPackage} />
        </div>
        <div>
          <Upload {...props} disabled={uploadPending}>
            <Button icon={<CloudUploadOutlined />} type='text' style={{ padding: '0 8px', borderRadius: 6 }}>
              {t('modal.uploadFile')}
            </Button>
            <div style={{ color: '#89939b' }}>
              {t('modal.fileLimit')}
            </div>
          </Upload>
        </div>
      </div>
      <div style={{ color: '#e7434a' }}>{errorInfo}</div>
    </Modal>
  );
}
