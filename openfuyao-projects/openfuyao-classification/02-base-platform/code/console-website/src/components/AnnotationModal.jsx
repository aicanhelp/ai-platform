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
import { useEffect, useStore, Fragment } from 'openinula';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

{ /**
props:{
  dataList //注解列表?标签列表  annotation->注解  label->标签
  type: 类型 注解/标签
  open: 是否打开
  callbackOk: 回调成功
  callbackCancel 回调失败
}
*/ }
export default function AnnotationModal({ dataList = [], type = 'annotation', open, callbackOk, callbackCancel }) {
  const { t } = useTranslation('common');
  const [editForm] = Form.useForm();
  const [confirmLoading, setConfirmLoading, confirmPending] = useButtonLoading();

  const handleEditDoneOk = async () => {
    await editForm.validateFields(); // 校验失败在此抛出，不进入 loading
    setConfirmLoading(true);
    try {
      await callbackOk(editForm.getFieldValue(`${type}List`));
    } catch {
      // 调用方未自行捕获异常时兜底，避免 unhandled rejection 导致按钮卡死
    } finally {
      setConfirmLoading(false);
    }
  };

  const handleEditCancel = () => {
    editForm.setFieldValue(`${type}List`, dataList); // 还原数据
    callbackCancel();
  };

  useEffect(() => {
    editForm.setFieldValue(`${type}List`, dataList);
  }, [dataList, type]);

  const themeStore = useStore('theme');

  return <Modal className={themeStore.$s.theme === 'dark' ? 'dark_box' : ''} width={720} open={open} title={type === 'annotation' ? t('annotationModal.editAnnotation') : t('annotationModal.editLabel')} footer={
    <Fragment key="annotation-modal-footer">
      <Button className='cancel_btn' onClick={handleEditCancel} disabled={confirmPending}>{t('actions.cancel')}</Button>
      <Button className='primary_btn' htmlType='submit' style={{ marginLeft: '16px' }} onClick={handleEditDoneOk} loading={confirmLoading}>{t('actions.confirm')}</Button>
    </Fragment>
  } okText={t('actions.confirm')} cancelText={t('actions.cancel')} forceRender onCancel={handleEditCancel}>
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
                align='baseline'
              >
                <Form.Item
                  {...restField}
                  name={[name, 'key']}
                  rules={[
                    {
                      required: true,
                      message: type === 'annotation' ? t('annotationModal.keyRequiredAnnotation') : t('annotationModal.keyRequiredLabel'),
                    },
                  ]}
                >
                  <Input placeholder={t('annotationModal.inputPlaceholder')} style={{ width: '420px' }} />
                </Form.Item>
                <Form.Item
                  {...restField}
                  name={[name, 'value']}
                  initialValue=''
                >
                  <Input placeholder={t('annotationModal.inputPlaceholder')} style={{ width: '192px' }} />
                </Form.Item>
                <DeleteOutlined className='common_antd_icon' onClick={() => remove(name)} />
              </Space>
            ))}
            <Form.Item>
              <Button className='icon_btn' type='link' onClick={() => add()} block icon={<PlusCircleOutlined className='primary_icon common_antd_icon' />}>
                {type === 'annotation' ? t('annotationModal.addAnnotation') : t('annotationModal.addLabel')}
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>
    </Form>
  </Modal>;
}