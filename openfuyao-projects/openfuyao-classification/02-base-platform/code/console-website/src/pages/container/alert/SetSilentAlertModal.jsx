/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useState, useStore, useMemo, useEffect } from 'openinula';
import { Select, Form, Button, Modal, DatePicker, Input, InputNumber, message } from 'antd';
import '@/styles/pages/alert.less';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import UserStore from '@/store/userStore';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';
dayjs.extend(utc);
export default function SetSilentAlertModal({ isShowModal, cancelEdit, remarks, submitModel }) {
  const { t } = useTranslation('alert');
  const userStore = UserStore();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const [silentForm] = Form.useForm();
  const [submitLoading, setSubmitLoading, submitPending] = useButtonLoading();
  const [durations, setDurations] = useState(1);
  const [timeType, setTimeType] = useState('d');
  const dateFormat = 'YYYY-MM-DD HH:mm';

  useEffect(() => {
    if (isShowModal) {
      silentForm.setFieldsValue({ remarks: remarks ?? '' });
    }
  }, [isShowModal, remarks]);

  const timeTypeOption = useMemo(() => [
    { label: t('timeUnit.minute'), value: 'm' },
    { label: t('timeUnit.hour'), value: 'h' },
    { label: t('timeUnit.day'), value: 'd' },
  ], [t]);

  const cancelModelEdit = () => {
    silentForm.resetFields();
    cancelEdit();
  };
  const handleChangeTimeType = (value) => {
    setTimeType(value);
  };
  const handleChangeDurations = (value) => {
    setDurations(value);
  };
  const checkRemarks = async (rule, value) => {
    if (!value) {
      throw new Error(t('form.remarksRequired'));
    }
    if (value.length > 64) {
      throw new Error(t('form.remarksMaxLength'));
    }
    return Promise.resolve();
  };
  const submitModelEditSilent = async () => {
    setSubmitLoading(true);
    try {
      await submitModelEditSilentInner();
    } finally {
      setSubmitLoading(false);
    }
  };
  const submitModelEditSilentInner = () => {
    let arr;
    if (silentForm.getFieldsValue().beginTime) {
      let formTime = dayjs(silentForm.getFieldsValue().beginTime.format('YYYY-MM-DD HH:mm')).unix();
      let nowTime = dayjs(dayjs().format('YYYY-MM-DD HH:mm')).unix();
      if (formTime < nowTime) {
        messageApi.error(t('messages.beginTimeError'));
      } else {
        arr = {
          startsAt: silentForm.getFieldsValue().beginTime.utc().format(),
          endsAt: (silentForm.getFieldsValue().beginTime.add(durations, timeType)).utc().format(),
          createdBy: silentForm.getFieldsValue().creators,
          comment: silentForm.getFieldsValue().remarks,
        };
        submitModel(arr);
      }
    } else {
      arr = {
        startsAt: dayjs().utc().format(),
        endsAt: (dayjs().add(durations, timeType)).utc().format(),
        createdBy: silentForm.getFieldsValue().creators,
        comment: silentForm.getFieldsValue().remarks,
      };
      submitModel(arr);
    }
  };
  return (
    <Modal
      className={`alertModel silentModelCommon ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`}
      title={t('modal.setSilentAlert')} open={isShowModal} onCancel={cancelModelEdit} footer={[]}
    >
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <Form
        layout='vertical'
        form={silentForm}
        name='silentForm'
        initialValues={{
          timetype: timeTypeOption[2],
          creators: userStore.user.name,
          durations: 1,
          remarks: remarks ?? '',
        }}
        onFinish={submitModelEditSilent}
      >
        <Form.Item name='beginTime' label={t('form.silentBeginTime')} >
          <DatePicker placeholder={t('form.defaultCurrentTime')} format={dateFormat} style={{ width: '318px', marginRight: '20px' }} minDate={dayjs()} showTime />
        </Form.Item>
        <div className='formTime'>
          <Form.Item name='durations' label={t('form.silentDuration')} rules={[
            { required: true, message: t('form.durationRequired') },
            { type: 'number', min: 1, max: 100, message: t('form.durationRange') },
            { type: 'integer', message: t('form.durationInteger') },
          ]}>
            <InputNumber value={durations} placeholder={t('form.durationPlaceholder')} style={{ width: '190px', marginRight: '8px' }} onChange={handleChangeDurations} />
          </Form.Item>
          <Form.Item name='timetype' label=' '>
            <Select value={timeType} style={{ width: '120px' }} options={timeTypeOption} onChange={handleChangeTimeType} />
          </Form.Item>
        </div>
        <Form.Item name='creators' label={t('form.creator')} rules={[{ required: true, message: t('form.creatorRequired') }]}>
          <Input style={{ width: '318px', marginRight: '20px' }} disabled={true} placeholder={t('form.fillPlaceholder')} />
        </Form.Item>
        <Form.Item name='remarks' label={t('form.remarks')} rules={[{ required: true, validator: checkRemarks }]}>
          <Input style={{ width: '318px' }} placeholder={t('form.fillPlaceholder')} />
        </Form.Item>
        <div className='formBtn' style={{ marginTop: '-32px' }}>
          <Form.Item>
            <Button className='cancel_btn' onClick={cancelModelEdit} style={{ marginRight: '14px' }} disabled={submitPending}>{t('actions.cancel')}</Button>
            <Button className='primary_btn' htmlType='submit' loading={submitLoading}>{t('actions.confirm')}</Button>
          </Form.Item>
        </div>
      </Form>
    </Modal>
  );
}
