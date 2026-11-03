/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Button, Form, Input, Select } from 'antd';
import '@/styles/pages/clusterUpgrade.less';
import { useIntl } from '../i18n';

export default function UpgradePrepareModal({ open, loading, cancelFn, okFn }) {
  const intl = useIntl();
  const [autoUpgradeForm] = Form.useForm();

  return (
    <Modal
      width={600}
      title={intl.formatMessage({ id: 'upgrade.prepare.title' })}
      open={open}
      onOk={() => okFn(autoUpgradeForm)}
      onCancel={() => cancelFn(autoUpgradeForm)}
      getContainer={false}
      footer={[
        <Button key="cancel" onClick={() => cancelFn(autoUpgradeForm)}>
          {intl.formatMessage({ id: 'upgrade.prepare.cancel' })}
        </Button>,
        <Button key="confirm" type="primary" loading={loading} onClick={() => okFn(autoUpgradeForm)}>
          {intl.formatMessage({ id: 'upgrade.prepare.confirm' })}
        </Button>,
      ]}
    >
      <Form form={autoUpgradeForm} layout="vertical" name="autoUpgradeForm">
        <Form.Item
          name="path"
          label={intl.formatMessage({ id: 'upgrade.prepare.path' })}
          rules={[
            { required: true, message: intl.formatMessage({ id: 'upgrade.prepare.pathRequired' }) },
            { type: 'string', min: 1, message: intl.formatMessage({ id: 'upgrade.prepare.pathEmpty' }) },
            {
              validator: (_, value) => {
                if (!value) return Promise.resolve();
                const pathRegex = /^\/[\w\-./]*$/;
                if (!pathRegex.test(value)) {
                  return Promise.reject(new Error(intl.formatMessage({ id: 'upgrade.prepare.pathInvalid' })));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <Input placeholder={intl.formatMessage({ id: 'upgrade.prepare.pathPlaceholder' })} />
        </Form.Item>
        <Form.Item
          name="filename"
          label={intl.formatMessage({ id: 'upgrade.prepare.fileName' })}
          rules={[
            { required: true, message: intl.formatMessage({ id: 'upgrade.prepare.fileNameRequired' }) },
            { type: 'string', min: 1, message: intl.formatMessage({ id: 'upgrade.prepare.fileNameEmpty' }) },
            {
              validator: (_, value) => {
                if (!value) return Promise.resolve();
                const filenameRegex = /^[a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$/;
                if (!filenameRegex.test(value)) {
                  return Promise.reject(new Error(intl.formatMessage({ id: 'upgrade.prepare.fileNameInvalid' })));
                }
                if (value.includes('.tar.gz')) {
                  return Promise.reject(new Error(intl.formatMessage({ id: 'upgrade.prepare.fileNameNoTarGz' })));
                }
                return Promise.resolve();
              },
            },
          ]}
        >
          <Input placeholder={intl.formatMessage({ id: 'upgrade.prepare.fileNamePlaceholder' })} />
        </Form.Item>
      </Form>
    </Modal>
  );
}
