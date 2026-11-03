/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useState, useEffect, useMemo } from 'openinula';
import { Modal, Button, Form, Input, Select } from 'antd';
import '@/styles/pages/clusterUpgrade.less';
import { WarningFilled } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import useButtonLoading from '@/hooks/useButtonLoading';
import { ResponseCode } from '@/common/constants';

import { getClusterUpgradeVersions } from '@/api/clusterApi';
import { useIntl } from '../i18n';

export default function UpgradeClusterModal({ open, clusterInfo, cancelFn, okFn }) {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [clusterUpgradeForm] = Form.useForm();
  const version = Form.useWatch('openFuyaoVersion', clusterUpgradeForm);
  const [upgradeLoading, setUpgradeLoading, upgradePending] = useButtonLoading();

  const handleOk = async (form) => {
    if (upgradePending) return;
    setUpgradeLoading(true);
    try {
      await okFn(form);
    } catch (error) {
      // 调用方未自行捕获异常时兜底，避免 unhandled rejection 导致按钮卡死
    } finally {
      setUpgradeLoading(false);
    }
  };

  const [patchList, setPatchList] = useState([]);
  useEffect(() => {
    getClusterUpgradeVersions(clusterInfo.name).then((res) => {
      if (res.status === ResponseCode.OK) {
        const {
          data: {
            data: { versions },
          },
        } = res;
        if (versions) {
          setPatchList(versions);
        }
      }
    });
  }, []);

  const patchOptions = useMemo(() => patchList.map((item) => ({ label: item, value: item })), [patchList]);

  return (
    <Modal
      getContainer={() => document.querySelector('#cluster_root .container_content')}
      className={currentTheme === 'dark' ? 'dark_box' : ''}
      width={720}
      open={open}
      title={intl.formatMessage({ id: 'upgrade.title' })}
      footer={[
        <div style={{ marginTop: '32px' }}>
          <Button className="cancel_btn" disabled={upgradePending} onClick={() => cancelFn(clusterUpgradeForm)}>
            {intl.formatMessage({ id: 'upgrade.cancel' })}
          </Button>
          <Button
            className={`${version && !upgradePending ? 'cluster_primary_btn' : 'cancel_btn'}`}
            htmlType="submit"
            style={{ marginLeft: '16px' }}
            disabled={!version || upgradePending}
            loading={upgradeLoading}
            onClick={() => handleOk(clusterUpgradeForm)}
          >
            {intl.formatMessage({ id: 'upgrade.confirm' })}
          </Button>
        </div>,
      ]}
      okText={intl.formatMessage({ id: 'upgrade.confirm' })}
      cancelText={intl.formatMessage({ id: 'upgrade.cancel' })}
      onCancel={() => cancelFn(clusterUpgradeForm)}
      destroyOnHidden
    >
      <Form
        preserve={false}
        form={clusterUpgradeForm}
        style={{ margin: '24px 0 0 0', minHeight: 250 }}
        initialValues={{
          clusterName: clusterInfo.name,
          openFuyaoVersion: clusterInfo.openFuyaoVersion,
        }}
      >
        <div className="upgrade_tooltip">
          <WarningFilled />
          <div>
            <p>{intl.formatMessage({ id: 'upgrade.warning1' })}</p>
            <p>{intl.formatMessage({ id: 'upgrade.warning2' })}</p>
          </div>
        </div>
        <div className="form_flex">
          <Form.Item label={intl.formatMessage({ id: 'upgrade.clusterName' })} name="clusterName" layout="vertical">
            <Input disabled />
          </Form.Item>
          <Form.Item
            label={intl.formatMessage({ id: 'upgrade.currentVersion' })}
            name="openFuyaoVersion"
            layout="vertical"
          >
            <Input disabled />
          </Form.Item>
        </div>
        <div className="row_line_flex">
          <Form.Item
            label={intl.formatMessage({ id: 'upgrade.selectVersion' })}
            name="openFuyaoUpgradeVersion"
            layout="vertical"
            rules={[
              {
                required: true,
                message: intl.formatMessage({ id: 'upgrade.selectVersionRequired' }),
              },
            ]}
          >
            <Select
              options={patchOptions}
              disabled={patchOptions.length === 0}
              placeholder={
                patchOptions.length === 0
                  ? intl.formatMessage({ id: 'upgrade.noAvailableVersion' })
                  : intl.formatMessage({ id: 'upgrade.selectVersionPlaceholder' })
              }
            />
          </Form.Item>
        </div>
      </Form>
    </Modal>
  );
}
