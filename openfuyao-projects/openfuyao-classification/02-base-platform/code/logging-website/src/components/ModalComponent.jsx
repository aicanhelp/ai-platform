/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Button, Modal, Form } from 'antd';
import { useState } from 'openinula';

export default function ModalComponent({ open, title, onCreate, onCancel, handleOk, initialValues, children }) {
  const [loading] = useState(false);
  const [form] = Form.useForm();

  return (
    <>
      <Modal
        open={open}
        title={title}
        width={724}
        onOk={handleOk}
        maskClosable={false}
        onCancel={onCancel}
        footer={[
          <Button key="back" onClick={onCancel}>
            取消
          </Button>,
          <Button key="submit" type="primary" loading={loading} onClick={handleOk}>
            确定
          </Button>,
        ]}
      >
        {children}
      </Modal>
    </>
  );
}
