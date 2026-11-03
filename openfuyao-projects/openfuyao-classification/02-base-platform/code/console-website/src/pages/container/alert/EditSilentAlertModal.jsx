/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { PlusCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { Select, Button, Space, Form, Input, DatePicker, Modal, InputNumber, message } from 'antd';
import { useEffect, useState, useStore, useMemo } from 'openinula';
import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';
dayjs.extend(utc);
export default function EditSilentAlertModal(props) {
  const { t } = useTranslation('alert');
  const [silentFormEdit] = Form.useForm();
  const themeStore = useStore('theme');
  const [beginTimeEdit, setBegintimeEdit] = useState(props.data?.beginTime);
  const [submitLoading, setSubmitLoading, submitPending] = useButtonLoading();
  const [durationsEdit, setDurationsEdit] = useState(props.data?.timeNum);
  const [timetypeEdit, setTimetypeEdit] = useState(props.data?.timeType);
  const dateFormat = 'YYYY-MM-DD HH:mm';
  const [messageApi, contextHolder] = message.useMessage();

  const timeTypeEditOption = useMemo(() => [
    { label: t('timeUnit.minute'), value: 'm' },
    { label: t('timeUnit.hour'), value: 'h' },
    { label: t('timeUnit.day'), value: 'd' },
  ], [t]);

  const handleChangeTimeEdit = (dates, dateStrings) => {
    if (dates) {
      setBegintimeEdit(dayjs(dateStrings));
    }
  };
  const handleChangeTimetypeEdit = (value) => {
    setTimetypeEdit(value);
  };
  const checkSilentRemarksEdit = async (rule, value) => {
    if (!value) {
      throw new Error(t('form.remarksRequired'));
    }
    if (value.length > 64) {
      throw new Error(t('form.remarksMaxLength'));
    }
    return Promise.resolve();
  };
  const handleChangeDurationsEdit = (value) => {
    setDurationsEdit(value);
  };
  const cancelModelEdit = () => {
    silentFormEdit.resetFields();
    setBegintimeEdit(props.data?.beginTime);
    props.cancelEdit();
  };
  const submitModelSilentEdit = async () => {
    setSubmitLoading(true);
    try {
      await submitModelSilentEditInner();
    } finally {
      setSubmitLoading(false);
    }
  };
  const submitModelSilentEditInner = () => {
    if (durationsEdit === undefined || durationsEdit === null || !timetypeEdit) {
      messageApi.error(t('form.durationRequired'));
      return;
    }
    let _labelList = [];
    silentFormEdit.getFieldsValue().labelList.map(item => {
      _labelList.push({
        name: item.key,
        value: item.value,
        isEqual: true,
        isRegex: false,
      });
    });
    let arr;
    if (silentFormEdit.getFieldsValue().beginTimeEdit) {
      let formTime = dayjs(silentFormEdit.getFieldsValue().beginTimeEdit.format('YYYY-MM-DD HH:mm')).unix();
      let nowTime = dayjs(dayjs().format('YYYY-MM-DD HH:mm')).unix();
      if (formTime < nowTime) {
        messageApi.error(t('messages.beginTimeError'));
      } else {
        arr = {
          startsAt: silentFormEdit.getFieldsValue().beginTimeEdit.utc().format(),
          endsAt: (silentFormEdit.getFieldsValue().beginTimeEdit.add(durationsEdit, timetypeEdit)).utc().format(),
          createdBy: silentFormEdit.getFieldsValue().creators,
          comment: silentFormEdit.getFieldsValue().remarks,
          matchers: _labelList,
        };
        silentFormEdit.resetFields();
        props.submitModel(arr);
      }
    } else {
      arr = {
        startsAt: dayjs().utc().format(),
        endsAt: (dayjs().add(durationsEdit, timetypeEdit)).utc().format(),
        createdBy: silentFormEdit.getFieldsValue().creators,
        comment: silentFormEdit.getFieldsValue().remarks,
        matchers: _labelList,
      };
      silentFormEdit.resetFields();
      props.submitModel(arr);
    }
  };
  useEffect(() => {
    setDurationsEdit(props.data?.timeNum);
    setTimetypeEdit(props.data?.timeType);
    silentFormEdit.setFieldsValue({
      beginTimeEdit: props.data?.beginTime,
      labelList: props.data?.labelList,
      remarks: props.data?.remarks,
      creators: props.data?.creators,
      durations: props.data?.timeNum,
      timetype: props.data?.timeType,
    });
  }, [props.data?.beginTime, props.data?.labelList, props.data?.remarks, props.data?.creators, props.data?.timeType, props.data?.timeNum]);
  return (
    <Modal
      className={`silentModel silentModelCommon ${themeStore.$s.theme === 'dark' ? 'dark_box' : ''}`}
      title={props.formType === 'add' ? t('modal.createSilence') : t('modal.modifySilence')} open={props.isShowModel} onCancel={cancelModelEdit}
      footer={[]}
      forceRender
    >
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <Form
        form={silentFormEdit}
        name='silentForm'
        layout="vertical"
        onFinish={submitModelSilentEdit}
      >
        <div className='formTime'>
          <Form.Item name='beginTimeEdit' label={t('form.silentBeginTime')} >
            <DatePicker value={beginTimeEdit} format={dateFormat} placeholder={t('form.defaultCurrentTime')} style={{ width: '318px', marginRight: '20px' }} minDate={dayjs()} onChange={handleChangeTimeEdit} showTime />
          </Form.Item>
          <Form.Item name='durations' label={t('form.silentDuration')}
            rules={[
              { required: true, message: t('form.durationRequired') },
              { type: 'number', min: 1, max: 100, message: t('form.durationRange') },
              { type: 'integer', message: t('form.durationInteger') },
            ]} >
            <InputNumber value={durationsEdit} placeholder={t('form.durationPlaceholder')} style={{ width: '190px', marginRight: '20px' }} onChange={handleChangeDurationsEdit} />
          </Form.Item>
          <Form.Item name='timetype' label=' '>
            <Select value={timetypeEdit} style={{ width: '120px' }} options={timeTypeEditOption} onChange={handleChangeTimetypeEdit} />
          </Form.Item>
        </div>
        <p><span style={{ color: '#E94547' }}>*</span> {t('form.labels')}</p>
        <div style={{ display: 'flex', margin: '8px 0' }}><p style={{ width: '392px', marginRight: '15px' }}>{t('form.labelKey')}</p><p>{t('form.labelValue')}</p></div>
        <Form.List name={'labelList'} title={t('form.labels')} rules={[
          {
            validator: async (_, labelTemporyList) => {
              if (labelTemporyList.length === 0) {
                return Promise.reject(new Error(t('form.atLeastOneLabel')));
              } else {
                return Promise.resolve();
              }
            },
          },
        ]}>
          {(fields, { add, remove }, { errors }) => (
            <>
              {fields.map(({ key, name, ...restField }) => (
                <Space
                  key={key}
                  className='field_label_edit_silent'
                  style={{
                    display: 'flex',
                  }}
                  align="baseline"
                >
                  <Form.Item
                    {...restField}
                    className='field_label_edit_silent'
                    name={[name, 'key']}
                    rules={[
                      {
                        required: true,
                        message: t('form.labelRequired'),
                      },
                    ]}
                  >
                    <Input className='field_label_edit_silent' placeholder={t('form.enterPlaceholder')} style={{ width: '392px', marginRight: '8px' }} />
                  </Form.Item>
                  <Form.Item
                    {...restField}
                    className='field_value_edit_silent'
                    name={[name, 'value']}
                    rules={[
                      {
                        required: true,
                        message: t('form.labelRequired'),
                      },
                    ]}
                  >
                    <Input className='field_value_edit_silent' placeholder={t('form.enterPlaceholder')} style={{ width: '220px' }} />
                  </Form.Item>
                  <DeleteOutlined className="common_antd_icon delete_common" onClick={() => remove(name)} />
                </Space>
              ))}
              <Form.Item>
                <Button className="icon_btn" type="link" onClick={() => add()} block icon={<PlusCircleOutlined className="primary_icon common_antd_icon" />}>
                  {t('form.addLabel')}
                </Button>
                <Form.ErrorList errors={errors} />
              </Form.Item>
            </>
          )}
        </Form.List>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Form.Item name='creators' label={t('form.creator')} rules={[
            { required: true },
          ]}>
            <Input style={{ width: '318px', marginRight: '20px' }} placeholder={t('form.fillPlaceholder')} disabled={true} />
          </Form.Item>
          <Form.Item name='remarks' label={t('form.remarks')} rules={[{ required: true, validator: checkSilentRemarksEdit }]}>
            <Input style={{ width: '318px' }} placeholder={t('form.fillPlaceholder')} />
          </Form.Item>
        </div>
        <div className='formBtn'>
          <Form.Item>
            <Button className='cancel_btn edit_silent_btn' onClick={cancelModelEdit} style={{ marginRight: '14px' }} disabled={submitPending}>{t('actions.cancel')}</Button>
            <Button className='primary_btn edit_silent_btn' htmlType='submit' loading={submitLoading}>{t('actions.confirm')}</Button>
          </Form.Item>
        </div>
      </Form>
    </Modal >
  );
}
