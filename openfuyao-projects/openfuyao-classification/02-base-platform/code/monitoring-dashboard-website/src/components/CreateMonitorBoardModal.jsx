/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Form, Button, Input } from 'antd';
import { Fragment, useEffect } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/components/createMonitorBoardModal.less';

export default function CreateMonitorBoardModal({ title, open, cancelFn, okFn, name = '' }) {
  const intl = useIntl();
  const [form] = Form.useForm();

  const handleOk = async () => {
    await form.validateFields();
    okFn(form.getFieldValue('name'));
    form.setFieldValue('name', '');
  };

  const handlCancel = () => {
    form.setFieldValue('name', name);
    cancelFn();
  };
  useEffect(() => {
    if (form) {
      form.setFieldValue('name', name);
    }
  }, [name]);
  return (
    <Modal
      className="monitor_modal"
      open={open}
      title={title}
      onCancel={handlCancel}
      forceRender
      getContainer={false}
      footer={[
        <Fragment>
          <Button className="cancel_btn normal_btn" onClick={handlCancel}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Button className="monitor_primary_btn normal_btn" htmlType="submit" onClick={handleOk}>
            {intl.formatMessage({ id: 'common.ok' })}
          </Button>
        </Fragment>,
      ]}
    >
      <Form className="monitor_form" form={form} layout="vertical" autoComplete="off">
        <Form.Item
          name="name"
          label={intl.formatMessage({ id: 'dashboard.boardName' })}
          rules={[{ required: true, message: intl.formatMessage({ id: 'dashboard.boardNameRequired' }) }]}
        >
          <Input showCount placeholder={intl.formatMessage({ id: 'common.inputPlaceholder' })} maxLength={50} />
        </Form.Item>
      </Form>
    </Modal>
  );
}
