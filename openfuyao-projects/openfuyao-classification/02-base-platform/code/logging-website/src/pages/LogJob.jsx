/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/log.less';
import { useIntl } from 'inula-intl';
import { Input, Button, Table, Space, Popover, Modal, Form, Select, message } from 'antd';
import { MoreOutlined } from '@ant-design/icons';
import { ResponseCode } from '@/common/constants';
import { getLogJobData, getLogCollectionData, addLogJob, deleteLogJob } from '@/api/containerLogApi';
import { useEffect, useState, useCallback, useRef } from 'openinula';
import { solveAnnotation, sorterFirstAlphabet } from '@/tools/utils';
import DeleteInfoModal from '@/components/DeleteInfoModal';
export default function LogJob() {
  const intl = useIntl();
  const [formType, setFormType] = useState('');
  const [jobDataSource, setJobDataSource] = useState([]);
  const [jobtableSource, setJobtableSource] = useState([]);
  const jobColumns = [
    {
      title: intl.formatMessage({ id: 'job.task' }),
      dataIndex: 'key',
      key: 'key',
      sorter: (a, b) => sorterFirstAlphabet(a.key, b.key),
    },
    {
      title: intl.formatMessage({ id: 'job.path' }),
      dataIndex: 'value',
      key: 'value',
      maxWidth: 500,
      render: (_, record) => (
        <Space size="middle" className="pathWarp">
          {record.value}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      dataIndex: '',
      key: 'x',
      render: (_, record) => (
        <Space>
          <Popover
            placement="bottom"
            content={
              <div className="pop_modal">
                <Button type="link" onClick={() => showModal('edit', record)}>
                  {intl.formatMessage({ id: 'common.edit' })}
                </Button>
                <Button type="link" onClick={() => handleDeleteJob(record.key)}>
                  {intl.formatMessage({ id: 'common.delete' })}
                </Button>
              </div>
            }
            trigger="click"
            open={popOpen === record.key}
            onOpenChange={(newOpen) => (newOpen ? setPopOpen(record.key) : setPopOpen(''))}
          >
            <MoreOutlined className="common_antd_icon primary_color" />
          </Popover>
        </Space>
      ),
    },
  ];
  const inpuRef = useRef(null);
  const [popOpen, setPopOpen] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [jobDelModalOpen, setJobDelModalOpen] = useState(false);
  const [jobName, setJobName] = useState('');
  const [mainPathLength, setMainPathLength] = useState(0);
  const [isJobDelCheck, setIsJobDelCheck] = useState(false);
  const { Search } = Input;
  const successLog = (value) => {
    messageApi.open({
      type: 'success',
      content: value,
    });
  };
  const onSearchLogJob = (value) => {
    if (value) {
      setJobtableSource(jobDataSource.filter((item) => item.key.includes(value)));
    } else {
      setJobtableSource(jobDataSource);
    }
  };
  const showModal = (value, data) => {
    if (value === 'edit') {
      form.setFieldValue('job', data.key);
      jobOption.map((item) => {
        if (data.value.includes(item.value)) {
          form.setFieldValue('childPath', data.value.slice(item.value.length));
          form.setFieldValue('path', item.value);
        }
      });
    } else {
      form.setFieldValue('path', jobOption[0].value);
    }
    setFormType(value);
    setIsModalOpen(true);
  };
  const handleCancel = () => {
    form.resetFields();
    setIsModalOpen(false);
  };
  function handleDeleteJob(value) {
    setJobName(value);
    setJobDelModalOpen(true);
  }
  const [form] = Form.useForm();
  const [jobOption, setJobOption] = useState([{ key: '', value: '' }]);
  const handleChangeJob = (value) => {
    setMainPathLength(value.length);
    inpuRef.current.focus();
    inpuRef.current.blur();
  };
  const getLogJobList = async () => {
    const res = await getLogJobData();
    if (res.status === ResponseCode.OK) {
      setJobDataSource(solveAnnotation(res.data.data));
      setJobtableSource(solveAnnotation(res.data.data));
    }
  };
  const getJobOptions = async () => {
    const res = await getLogCollectionData();
    if (res.status === ResponseCode.OK) {
      let _arr = [];
      res.data.data[Object.keys(res.data.data)[0]].map((item) => {
        _arr.push({ label: item.mountPath, value: item.mountPath });
      });
      setJobOption(_arr);
    }
  };
  const handleOk = () => {
    let arr = { jobName: '', path: '' };
    arr.jobName = form.getFieldsValue().job;
    arr.path = form.getFieldsValue().path + form.getFieldsValue().childPath;
    addJob(arr);
  };
  const addJob = async (value) => {
    if (form.getFieldsValue().job && form.getFieldsValue().childPath) {
      try {
        const res = await addLogJob(value);
        if (res.status === ResponseCode.OK) {
          setIsModalOpen(false);
          successLog(
            formType === 'add'
              ? intl.formatMessage({ id: 'job.addSuccess' })
              : intl.formatMessage({ id: 'job.editSuccess' }),
          );
          form.resetFields();
          getLogJobList();
        }
      } catch (error) {
        if (error.response.status === ResponseCode.Forbidden) {
          messageApi.error(intl.formatMessage({ id: 'job.noPermission' }));
        } else {
          messageApi.error(error.response?.data?.Msg);
        }
      }
    }
  };
  const handlePodCheckFn = (e) => {
    setIsJobDelCheck(e.target.checked);
  };
  const deleteJob = async () => {
    try {
      const res = await deleteLogJob(jobName);
      if (res.status === ResponseCode.OK) {
        successLog(intl.formatMessage({ id: 'job.deleteSuccess' }, { name: jobName }));
        setJobDelModalOpen(false);
        setIsJobDelCheck(false);
        setJobName('');
        getLogJobList();
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(intl.formatMessage({ id: 'job.noPermission' }));
      } else {
        messageApi.error(error.response?.data?.Msg);
      }
    }
  };
  const isValid = (str) => {
    return /^[0-9a-zA-Z_-]+$/.test(str);
  };
  const checkJobname = async (rule, value) => {
    if (!value) {
      throw new Error(intl.formatMessage({ id: 'job.nameRequired' }));
    }
    if (value.length < 1 || value.length > 30) {
      throw new Error(intl.formatMessage({ id: 'job.nameLengthError' }));
    }
    if (!isValid(value)) {
      throw new Error(intl.formatMessage({ id: 'job.nameFormatError' }));
    }
    return Promise.resolve();
  };
  const isPathValid = (str) => {
    return /^[A-Za-z0-9*-/_]*$/.test(str);
  };
  const checkPath = useCallback(async () => {
    let value = form.getFieldsValue().childPath;
    if (!value) {
      throw new Error(intl.formatMessage({ id: 'job.pathRequired' }));
    }
    if (value.length < 6) {
      throw new Error(intl.formatMessage({ id: 'job.pathMinLength' }));
    }
    if (value.length + mainPathLength > 300) {
      throw new Error(intl.formatMessage({ id: 'job.pathMaxLength' }));
    }
    if (!isPathValid(value)) {
      throw new Error(intl.formatMessage({ id: 'job.pathFormatError' }));
    }
    if (value[0] !== '/') {
      throw new Error(intl.formatMessage({ id: 'job.pathStartError' }));
    }
    if (value.substring(value.length - 4) !== '.log') {
      throw new Error(intl.formatMessage({ id: 'job.pathEndError' }));
    }
    return Promise.resolve();
  }, [mainPathLength]);
  const handleDelJobCancel = () => {
    setJobDelModalOpen(false);
    setJobName('');
  };
  useEffect(() => {
    getLogJobList();
    getJobOptions();
  }, []);
  useEffect(() => {
    checkPath();
  }, [checkPath]);
  return (
    <div className="logJob">
      {contextHolder}
      <div className="logJob_top">
        <Search
          placeholder={intl.formatMessage({ id: 'job.searchPlaceholder' })}
          onSearch={onSearchLogJob}
          style={{ width: 200 }}
        />
        <Button className="primary_btn" onClick={() => showModal('add', '')}>
          {intl.formatMessage({ id: 'job.create' })}
        </Button>
      </div>
      <Table
        dataSource={jobtableSource}
        columns={jobColumns}
        pagination={false}
        locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
      />
      <Modal
        getContainer={false}
        title={
          formType === 'add'
            ? intl.formatMessage({ id: 'job.createTitle' })
            : intl.formatMessage({ id: 'job.editTitle' })
        }
        open={isModalOpen}
        onCancel={handleCancel}
        footer={[]}
      >
        <Form
          layout="vertical"
          form={form}
          name="form_in_modal"
          autoComplete="off"
          onFinish={handleOk}
          validateTrigger={['onBlur', 'onChange']}
        >
          <div className="form_model">
            <Form.Item
              name="job"
              rules={[{ required: true, validator: checkJobname }]}
              label={intl.formatMessage({ id: 'job.taskLabel' })}
              style={{ marginTop: '20px' }}
            >
              <Input placeholder={intl.formatMessage({ id: 'job.taskPlaceholder' })} />
            </Form.Item>
            <Form.Item name="path" label={intl.formatMessage({ id: 'job.pathLabel' })}>
              <Select onChange={handleChangeJob} options={jobOption} />
            </Form.Item>
            <Form.Item
              name="childPath"
              rules={[{ required: true, validator: checkPath }]}
              label={intl.formatMessage({ id: 'job.childPathLabel' })}
            >
              <Input placeholder={intl.formatMessage({ id: 'job.childPathPlaceholder' })} ref={inpuRef} />
            </Form.Item>
            <Form.Item style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '-10px' }}>
              <Button className="cancel_btn" onClick={handleCancel} style={{ marginRight: '14px' }}>
                {intl.formatMessage({ id: 'common.cancel' })}
              </Button>
              <Button className="primary_btn" htmlType="submit">
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
            </Form.Item>
          </div>
        </Form>
      </Modal>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'job.deleteTitle' })}
        open={jobDelModalOpen}
        cancelFn={handleDelJobCancel}
        content={[
          intl.formatMessage({ id: 'job.deleteWarning' }),
          intl.formatMessage({ id: 'job.deleteConfirm' }, { name: jobName }),
        ]}
        isCheck={isJobDelCheck}
        showCheck={true}
        checkFn={handlePodCheckFn}
        confirmFn={deleteJob}
      />
    </div>
  );
}
