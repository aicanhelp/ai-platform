/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Modal, Form, Space, Button, Input } from 'antd';
import { PlusCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { useEffect } from 'openinula';
import { useIntl } from 'inula-intl';

{
  /**
props:{
  dataList //注解列表?标签列表  annotation->注解  label->标签
  type: 类型 注解/标签
  open: 是否打开
  callbackOk: 回调成功
  callbackCancel 回调失败
}
*/
}
export default function AnnotationModal({ dataList = [], type = 'annotation', open, callbackOk, callbackCancel }) {
  const intl = useIntl();
  const [editForm] = Form.useForm();

  const handleEditDoneOk = async () => {
    await editForm.validateFields();
    callbackOk(editForm.getFieldValue(`${type}List`));
  };

  const handleEditCancel = () => {
    editForm.setFieldValue(`${type}List`, dataList); // 还原数据
    callbackCancel();
  };

  useEffect(() => {
    editForm.setFieldValue(`${type}List`, dataList); // 设置初始值
  }, []);

  return (
    <Modal
      width={720}
      getContainer={false}
      open={open}
      title={
        type === 'annotation'
          ? intl.formatMessage({ id: 'annotation.editAnnotation' })
          : intl.formatMessage({ id: 'annotation.editLabel' })
      }
      footer={[
        <>
          <Button className="cancel_btn" onClick={handleEditCancel}>
            {intl.formatMessage({ id: 'common.cancel' })}
          </Button>
          <Button className="primary_btn" htmlType="submit" style={{ marginLeft: '16px' }} onClick={handleEditDoneOk}>
            {intl.formatMessage({ id: 'common.confirm' })}
          </Button>
        </>,
      ]}
      okText={intl.formatMessage({ id: 'common.confirm' })}
      cancelText={intl.formatMessage({ id: 'common.cancel' })}
      forceRender
      onCancel={callbackCancel}
    >
      <Form form={editForm} style={{ marginTop: '24px' }}>
        <Form.List name={`${type}List`}>
          {(fields, { add, remove }) => (
            <>
              {fields.map(({ key, name, ...restField }) => (
                <Space
                  key={key}
                  style={{
                    display: 'flex',
                  }}
                  align="baseline"
                >
                  <Form.Item
                    {...restField}
                    name={[name, 'key']}
                    rules={[
                      {
                        required: true,
                        message: `${type === 'annotation' ? intl.formatMessage({ id: 'annotation.annotationKey' }) : intl.formatMessage({ id: 'annotation.labelKey' })}`,
                      },
                    ]}
                  >
                    <Input placeholder={intl.formatMessage({ id: 'annotation.enterKey' })} style={{ width: '420px' }} />
                  </Form.Item>
                  <Form.Item {...restField} name={[name, 'value']} initialValue="">
                    <Input
                      placeholder={intl.formatMessage({ id: 'annotation.enterValue' })}
                      style={{ width: '192px' }}
                    />
                  </Form.Item>
                  <DeleteOutlined className="common_antd_icon" onClick={() => remove(name)} />
                </Space>
              ))}
              <Form.Item>
                <Button
                  className="icon_btn"
                  type="link"
                  onClick={() => add()}
                  block
                  icon={<PlusCircleOutlined className="primary_icon common_antd_icon" />}
                >
                  {intl.formatMessage({ id: 'annotation.addAnnotation' })}
                </Button>
              </Form.Item>
            </>
          )}
        </Form.List>
      </Form>
    </Modal>
  );
}
