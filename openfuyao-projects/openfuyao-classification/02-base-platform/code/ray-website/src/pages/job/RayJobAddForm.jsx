/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Space, Select, Checkbox, Table, Button, Form, Radio, message, Input, InputNumber, Popconfirm } from 'antd';

import { ResponseCode } from '@/common/constants';
import { useHistory } from 'inula-router';
import Inula, { useEffect, useState, useContext, useRef } from 'openinula';
import { PlusCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import { workHeadConfigTh, jsonToYaml, yamlTojson, validateName, validateImage } from '@/tools/utils';
import { npuTypeOption, jobAddFormDefault, jobAddFormWorkGroupData } from '@/common/constants';
import { addJob } from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import '@/styles/components/codeMirrorEditor.less';

export default function RayJobAddForm() {
  const intl = useIntl();
  const placeholder = intl.formatMessage({ id: 'common.pleaseEnter' });
  const history = useHistory();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const [messageApi, contextHolder] = message.useMessage();
  const [addJobForm] = Form.useForm();
  const [isJobWorkCreate, setIsJobWorkCreate] = useState(false);
  const [isJobHeadCreate, setIsJobHeadCreate] = useState(false);
  const [jobHeadVolumesYaml, setJobHeadVolumesYaml] = useState();
  const [jobHeadVolumeMountsYaml, setJobHeadVolumeMountsYaml] = useState();
  const [jobWorkVolumesYaml, setJobWorkVolumesYaml] = useState();
  const [jobWorkVolumeMountsYaml, setJobWorkVolumeMountsYaml] = useState();
  // head节点规格配置操作
  const JobEditableContextHead = Inula.createContext(null);
  const EditableRowHead = ({ index, ...props }) => {
    const [form] = Form.useForm();
    return (
      <Form form={form} component={false}>
        <JobEditableContextHead.Provider value={form}>
          <tr {...props} />
        </JobEditableContextHead.Provider>
      </Form>
    );
  };
  const getValidationRules = (dataIndex) => {
    if (dataIndex.includes('nodeConfigName')) {
      return [{ validator: validateName }];
    }
    if (dataIndex.includes('npuType')) {
      return [];
    }
    return [{ validator: jobValidateHead }];
  };

  const JobEditableCellHead = ({
    title,
    editable,
    children,
    dataIndex,
    record,
    handleJobSaveHeadTable,
    ...restProps
  }) => {
    const [jobHeadEditing, setJobHeadEditing] = useState(false);
    const jobHeadInputRef = useRef(null);
    const jobHeadSelectRef = useRef(null);
    const form = useContext(JobEditableContextHead);

    useEffect(() => {
      if (jobHeadEditing) {
        if (dataIndex.includes('npuType')) {
          jobHeadSelectRef.current?.focus();
        } else {
          jobHeadInputRef.current?.focus();
        }
      }
    }, [jobHeadEditing, dataIndex]);

    const toggleEditJobHead = () => {
      setJobHeadEditing(!jobHeadEditing);
      form.setFieldsValue({
        [dataIndex]: record[dataIndex],
      });
    };

    const save = async () => {
      try {
        const values = await form.validateFields();
        toggleEditJobHead();
        handleJobSaveHeadTable({
          ...record,
          ...values,
        });
      } catch (errInfo) {
        console.log('Save failed:', errInfo);
      }
    };

    let childNode = children;

    if (editable) {
      if (jobHeadEditing) {
        const FormElement = dataIndex.includes('npuType') ? (
          <Select ref={jobHeadSelectRef} onBlur={save} style={{ width: 120 }} options={npuTypeOption} />
        ) : (
          <Input ref={jobHeadInputRef} onBlur={save} />
        );

        childNode = (
          <Form.Item style={{ margin: 0 }} name={dataIndex} rules={getValidationRules(dataIndex)}>
            {FormElement}
          </Form.Item>
        );
      } else {
        childNode = (
          <div
            className="editable-cell-value-wrap defaultJobHead"
            style={{
              paddingInlineEnd: 24,
            }}
            onClick={toggleEditJobHead}
          >
            {children}
          </div>
        );
      }
    }

    return <td {...restProps}>{childNode}</td>;
  };

  const jobValidateHead = async (_rule, value) => {
    const regex = /^\d+$/;
    if (value && !regex.test(value)) {
      return Promise.reject(new Error(`${workHeadConfigTh(_rule.field)}类型应为数字！`));
    } else {
      return Promise.resolve();
    }
  };
  const [jobHeadTableData, setJobHeadTableData] = useState([
    {
      key: '0',
      nodeType: 'head',
      nodeConfigName: '请输入',
      memory: '请输入',
      maxCopy: '--',
      minCopy: '--',
      npu: '请输入',
      cpu: '请输入',
      npuType: '请输入',
    },
  ]);
  const defaultJobHeadlTable = [
    {
      title: '节点类型',
      key: 'jobHeadMaxType',
      dataIndex: 'nodeType',
      render: (_, record) => record.nodeType,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>节点配置名称
        </p>
      ),
      key: 'jobHeadName',
      dataIndex: 'nodeConfigName',
      editable: true,
      render: (_, record) => record.nodeConfigName || '请输入',
    },
    {
      title: 'NPU （卡）',
      key: 'jobHeadNpu',
      dataIndex: 'npu',
      editable: true,
      render: (_, record) => record.npu || '请输入',
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {'CPU (Core)'}
        </p>
      ),
      key: 'jobHeadCpu',
      dataIndex: 'cpu',
      editable: true,
      render: (_, record) => record.cpu || '请输入',
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {'内存 (G)'}
        </p>
      ),
      key: 'jobHeadMemory',
      dataIndex: 'memory',
      editable: true,
      render: (_, record) => record.memory || '请输入',
    },
    {
      title: <p style={{ color: '#b0a7a7' }}>最大副本数</p>,
      key: 'jobHeadMaxCopy',
      dataIndex: 'maxCopy',
      editable: false,
      render: (_, record) => record.maxCopy || '--',
    },
    {
      title: <p style={{ color: '#b0a7a7' }}>最小副本数</p>,
      key: 'jobHeadMinCopy',
      dataIndex: 'minCopy',
      editable: false,
      render: (_, record) => record.minCopy || '--',
    },
    {
      title: 'NPU类型',
      key: 'jobHeadType',
      dataIndex: 'npuType',
      editable: true,
      render: (_, record) => record.npuType || '请输入',
    },
  ];
  const handleJobSaveHeadTable = (row) => {
    const newData = [...jobHeadTableData];
    const index = newData.findIndex((item) => row.key === item.key);
    const item = newData[index];
    newData.splice(index, 1, {
      ...item,
      ...row,
    });
    setJobHeadTableData(newData);
  };
  const jobHeadComponents = {
    body: {
      row: EditableRowHead,
      cell: JobEditableCellHead,
    },
  };
  const headColumns = defaultJobHeadlTable.map((col) => {
    if (!col.editable) {
      return col;
    }
    return {
      ...col,
      onCell: (record) => ({
        record,
        editable: col.editable,
        dataIndex: col.dataIndex,
        title: col.title,
        handleJobSaveHeadTable,
      }),
    };
  });

  //添加work节点规格配置操作
  const WorkEditableContext = Inula.createContext(null);
  const WorkEditableRow = ({ index, ...props }) => {
    const [form] = Form.useForm();
    return (
      <Form form={form} component={false}>
        <WorkEditableContext.Provider value={form}>
          <tr {...props} />
        </WorkEditableContext.Provider>
      </Form>
    );
  };
  const WorkEditableCell = ({ title, editable, children, dataIndex, record, handleSaveJobWorkTable, ...restProps }) => {
    const [jobWorkEditing, setJobWorkEditing] = useState(false);
    const jobWorkInputRef = useRef(null);
    const jobWorkSelectRef = useRef(null);
    const form = useContext(WorkEditableContext);

    useEffect(() => {
      if (jobWorkEditing) {
        jobWorkInputRef.current?.focus();
      }
    }, [jobWorkEditing]);

    const toggleEditJobWork = () => {
      setJobWorkEditing(!jobWorkEditing);
      form.setFieldsValue({
        [dataIndex]: record[dataIndex],
      });
    };

    const save = async () => {
      try {
        const values = await form.validateFields();
        toggleEditJobWork();
        handleSaveJobWorkTable({
          ...record,
          ...values,
        });
      } catch (errInfo) {
        console.log('Save failed:', errInfo);
      }
    };

    let childNode = children;

    if (editable) {
      if (jobWorkEditing) {
        let formInput;
        if (!dataIndex.includes('npuType')) {
          formInput = <Input ref={jobWorkInputRef} onPressEnter={save} onBlur={save} />;
        } else {
          formInput = <Select ref={jobWorkSelectRef} onBlur={save} style={{ width: 120 }} options={npuTypeOption} />;
        }

        let validatorFunction;
        if (dataIndex.includes('nodeConfigName')) {
          validatorFunction = validateName;
        } else if (dataIndex.includes('npuType')) {
          validatorFunction = '';
        } else {
          validatorFunction = jobValidateWork;
        }

        childNode = (
          <Form.Item style={{ margin: 0 }} name={dataIndex} rules={[{ validator: validatorFunction }]}>
            {formInput}
          </Form.Item>
        );
      } else {
        childNode = (
          <div
            className="editable-cell-value-wrap defaultJobWork"
            style={{
              paddingInlineEnd: 24,
            }}
            onClick={toggleEditJobWork}
          >
            {children}
          </div>
        );
      }
    }

    return <td {...restProps}>{childNode}</td>;
  };

  const jobValidateWork = async (_rule, value) => {
    const regex = /^\d+$/;
    if (value && !regex.test(value)) {
      return Promise.reject(
        new Error(intl.formatMessage({ id: 'table.shouldBeNumber' }, { field: workHeadConfigTh(_rule.field) })),
      );
    } else {
      return Promise.resolve();
    }
  };
  const [jobWorkTableData, setJobWorkTableData] = useState([
    {
      key: '0',
      nodeType: 'worker',
      nodeConfigName: '请输入',
      npu: '请输入',
      cpu: '请输入',
      maxCopy: '请输入',
      minCopy: '请输入',
      npuType: '请输入',
      memory: '请输入',
    },
  ]);
  const [count, setCount] = useState(1);
  const handleDeleteJobWorkTable = (key) => {
    const newData = jobWorkTableData.filter((item) => item.key !== key);
    setJobWorkTableData(newData);
  };
  const defaultJobWorklTable = [
    {
      title: intl.formatMessage({ id: 'table.nodeType' }),
      key: 'jobWorkType',
      dataIndex: 'nodeType',
      render: (_, record) => record.nodeType,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.nodeConfigName' })}
        </p>
      ),
      key: 'jobWorkName',
      dataIndex: 'nodeConfigName',
      editable: true,
      render: (_, record) => record.nodeConfigName || placeholder,
    },
    {
      title: intl.formatMessage({ id: 'table.npuCard' }),
      key: 'jobWorkNpu',
      dataIndex: 'npu',
      editable: true,
      render: (_, record) => record.npu || placeholder,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {'CPU (Core)'}
        </p>
      ),
      key: 'jobWorkCpu',
      dataIndex: 'cpu',
      editable: true,
      render: (_, record) => record.cpu || placeholder,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.memoryG' })}
        </p>
      ),
      key: 'jobWorkMemory',
      dataIndex: 'memory',
      editable: true,
      render: (_, record) => record.memory || placeholder,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.maxReplicas' })}
        </p>
      ),
      key: 'jobWorkMaxCopy',
      dataIndex: 'maxCopy',
      editable: true,
      render: (_, record) => record.maxCopy || placeholder,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.minReplicas' })}
        </p>
      ),
      key: 'jobWorkMinCopy',
      dataIndex: 'minCopy',
      editable: true,
      render: (_, record) => record.minCopy || placeholder,
    },
    {
      title: intl.formatMessage({ id: 'table.npuType' }),
      key: 'jobWorkType',
      dataIndex: 'npuType',
      editable: true,
      render: (_, record) => record.npuType || placeholder,
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      dataIndex: 'operation',
      render: (_, record) =>
        jobWorkTableData.length > 1 ? (
          <Popconfirm
            title={intl.formatMessage({ id: 'common.confirmDelete' })}
            onConfirm={() => handleDeleteJobWorkTable(record.key)}
          >
            <DeleteOutlined style={{ color: '#3f66f5' }} />
          </Popconfirm>
        ) : (
          <DeleteOutlined style={{ color: '#89939b' }} />
        ),
    },
  ];
  const handleAddJobWorkTable = () => {
    const newData = {
      key: count,
      nodeType: 'worker',
      nodeConfigName: placeholder,
      memory: placeholder,
      maxCopy: placeholder,
      minCopy: placeholder,
      npuType: placeholder,
      npu: placeholder,
      cpu: placeholder,
    };
    setJobWorkTableData([...jobWorkTableData, newData]);
    setCount(count + 1);
  };
  const handleSaveJobWorkTable = (row) => {
    const newData = [...jobWorkTableData];
    const index = newData.findIndex((item) => row.key === item.key);
    const item = newData[index];
    newData.splice(index, 1, {
      ...item,
      ...row,
    });
    setJobWorkTableData(newData);
  };
  const JobWorkComponents = {
    body: {
      row: WorkEditableRow,
      cell: WorkEditableCell,
    },
  };
  const jobWorkColumns = defaultJobWorklTable.map((col) => {
    if (!col.editable) {
      return col;
    }
    return {
      ...col,
      onCell: (record) => ({
        record,
        editable: col.editable,
        dataIndex: col.dataIndex,
        title: col.title,
        handleSaveJobWorkTable,
      }),
    };
  });

  const onFinish = (data) => {
    if (isJobWorkCreate && isJobHeadCreate) {
      submitAdd(data);
    } else {
      messageApi.error(intl.formatMessage({ id: 'ray.job.formError' }));
    }
  };
  // 调用创建接口
  const submitAdd = async (data) => {
    let jobFormData = jobAddFormDefault;
    jobFormData.metadata.name = addJobForm.getFieldsValue().jobName;
    jobFormData.spec.submissionMode = addJobForm.getFieldsValue().submitType;
    jobFormData.spec.entrypoint = addJobForm.getFieldsValue().programEntry;
    jobFormData.spec.rayClusterSpec.rayVersion = addJobForm.getFieldsValue().jobImageversion;
    if (addJobForm.getFieldsValue().programEntryNum) {
      jobFormData.spec.entrypointNumCpus = addJobForm.getFieldsValue().programEntryNum;
    }
    console.log(addJobForm.getFieldsValue());
    if (addJobForm.getFieldsValue().entryResource && addJobForm.getFieldsValue().entryResource.length > 0) {
      let defultEntrypointResources = {};
      jobFormData.spec.entrypointResources = {};
      addJobForm.getFieldsValue().entryResource.map((item) => {
        defultEntrypointResources[item.name] = item.value;
      });
      jobFormData.spec.entrypointResources = jsonToYaml(JSON.stringify(defultEntrypointResources));
    }
    if (addJobForm.getFieldsValue().check) {
      if (
        Object.prototype.hasOwnProperty.call(jobFormData.spec.rayClusterSpec.headGroupSpec.rayStartParams, 'num-cpus')
      ) {
        delete jobFormData.spec.rayClusterSpec.headGroupSpec.rayStartParams['num-cpus'];
      }
      if (jobHeadTableData[0].npu !== placeholder && jobHeadTableData[0].npuType !== '') {
        jobFormData.spec.rayClusterSpec.headGroupSpec.rayStartParams.resources = `{"NPU":${jobHeadTableData[0].npu}}`;
      }
      if (jobHeadTableData[0].npuType !== placeholder && jobHeadTableData[0].npuType !== '') {
        jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].schedulerName = 'volcano';
      }
    } else {
      jobFormData.spec.rayClusterSpec.headGroupSpec.rayStartParams['num-cpus'] = '0';
    }
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].name = jobHeadTableData[0].nodeConfigName;
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].image = `${addJobForm.getFieldsValue().jobImagename}:${addJobForm.getFieldsValue().jobImageversion}`;
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].resources.limits.cpu =
      jobHeadTableData[0].cpu;
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].resources.limits.memory = `${jobHeadTableData[0].memory}G`;
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].resources.requests.cpu =
      jobHeadTableData[0].cpu;
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].resources.requests.memory = `${jobHeadTableData[0].memory}G`;
    if (addJobForm.getFieldsValue().headVolumes) {
      jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.volumes = [
        yamlTojson(addJobForm.getFieldsValue().headVolumes),
      ];
    }
    if (addJobForm.getFieldsValue().headVolumeMounts) {
      jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].volumeMounts = [
        yamlTojson(addJobForm.getFieldsValue().headVolumeMounts),
      ];
    }
    if (jobHeadTableData[0].npu !== placeholder && jobHeadTableData[0].npu !== '') {
      jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].resources.limits[
        jobHeadTableData[0].npuType
      ] = jobHeadTableData[0].npu;
      jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].resources.requests[
        jobHeadTableData[0].npuType
      ] = jobHeadTableData[0].npu;
    }
    jobFormData.spec.rayClusterSpec.headGroupSpec.template.spec.containers[0].env = addJobForm.getFieldsValue().headEnv;
    let jobWorkGroupSpecs = [];
    jobWorkTableData.forEach((item) => {
      let defaultWorkSingleData = jobAddFormWorkGroupData;
      defaultWorkSingleData.maxReplicas = Number(item.maxCopy);
      defaultWorkSingleData.minReplicas = Number(item.minCopy);
      defaultWorkSingleData.template.spec.containers[0].name = item.nodeConfigName;
      defaultWorkSingleData.template.spec.containers[0].image = `${addJobForm.getFieldsValue().jobImagename}:${addJobForm.getFieldsValue().jobImageversion}`;
      defaultWorkSingleData.template.spec.containers[0].resources.limits.cpu = item.cpu;
      defaultWorkSingleData.template.spec.containers[0].resources.limits.memory = `${item.memory}G`;
      defaultWorkSingleData.template.spec.containers[0].resources.requests.cpu = item.cpu;
      defaultWorkSingleData.template.spec.containers[0].resources.requests.memory = `${item.memory}G`;
      if (item.npu !== placeholder && item.npu !== '') {
        defaultWorkSingleData.template.spec.containers[0].resources.limits[item.npuType] = item.npu;
        defaultWorkSingleData.template.spec.containers[0].resources.requests[item.npuType] = item.npu;
      }
      defaultWorkSingleData.template.spec.containers[0].env = addJobForm.getFieldsValue().workEnv;

      if (addJobForm.getFieldsValue().workVolumes) {
        defaultWorkSingleData.template.spec.volumes = [yamlTojson(addJobForm.getFieldsValue().workVolumes)];
      }
      if (addJobForm.getFieldsValue().workVolumeMounts) {
        defaultWorkSingleData.template.spec.containers[0].volumeMounts = [
          yamlTojson(addJobForm.getFieldsValue().workVolumeMounts),
        ];
      }
      jobWorkGroupSpecs.push(defaultWorkSingleData);
    });
    jobFormData.spec.rayClusterSpec.workerGroupSpecs = jobWorkGroupSpecs;
    const appFormData = new FormData();
    appFormData.append('config', jsonToYaml(JSON.stringify(jobFormData)));
    appFormData.append('createBy', creator);
    appFormData.append('isStart', data);
    try {
      const res = await addJob(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.job.createSuccess' }));
        setTimeout(() => {
          history.push('/RayJob');
        }, 500);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.createFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.createFail' })},${msg}`);
      }
    }
  };
  const goBack = () => {
    history.push('/RayJob');
  };
  useEffect(() => {
    let checkWorkvalue = [];
    jobWorkTableData.forEach((item) => {
      let value =
        item.nodeConfigName !== '' &&
        item.nodeConfigName !== placeholder &&
        item.cpu !== '' &&
        item.cpu !== placeholder &&
        item.maxCopy !== '' &&
        item.maxCopy !== placeholder &&
        item.minCopy !== '' &&
        item.minCopy !== placeholder &&
        item.memory !== '' &&
        item.memory !== placeholder &&
        (item.npu !== '' || item.npu !== placeholder) &&
        (item.npuType !== '' || item.npuType !== placeholder) &&
        (item.npu === '' || item.npu === placeholder) &&
        (item.npuType === '' || item.npuType === placeholder);
      if (value) {
        checkWorkvalue.push('true');
      } else {
        checkWorkvalue.push('false');
      }
    });
    if (checkWorkvalue.includes('false')) {
      setIsJobWorkCreate(false);
    } else {
      setIsJobWorkCreate(true);
    }
  }, [jobWorkTableData]);
  useEffect(() => {
    let checkHeadvalue = [];
    jobHeadTableData.forEach((item) => {
      let value =
        item.nodeConfigName !== '' &&
        item.nodeConfigName !== placeholder &&
        item.cpu !== '' &&
        item.cpu !== placeholder &&
        item.memory !== '' &&
        item.memory !== placeholder &&
        (item.npu !== '' || item.npu !== placeholder) &&
        (item.npuType !== '' || item.npuType !== placeholder) &&
        (item.npu === '' || item.npu === placeholder) &&
        (item.npuType === '' || item.npuType === placeholder);
      if (value) {
        checkHeadvalue.push('true');
      } else {
        checkHeadvalue.push('false');
      }
    });
    if (checkHeadvalue.includes('false')) {
      setIsJobHeadCreate(false);
    } else {
      setIsJobHeadCreate(true);
    }
  }, [jobHeadTableData]);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="rayAddForm">
      {contextHolder}
      <div className="rayAddFormContent">
        <Form
          form={addJobForm}
          initialValues={{ submitType: 'K8sJobMode' }}
          onFinish={onFinish}
          autoComplete="off"
          layout={'vertical'}
        >
          <div className="rayAddForm_info">
            <p className="rayAddForm_info_title">{intl.formatMessage({ id: 'ray.job.basicInfo' })}</p>
            <div className="rayAddForm_info_horizontal">
              <Form.Item
                label={intl.formatMessage({ id: 'ray.job.clusterName' })}
                name="jobName"
                rules={[
                  { required: true, message: intl.formatMessage({ id: 'common.inputRequired' }) },
                  { validator: validateName },
                ]}
              >
                <Input showCount maxLength={253} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item
                label={intl.formatMessage({ id: 'ray.job.programEntry' })}
                name="programEntry"
                rules={[{ required: true, message: intl.formatMessage({ id: 'common.inputRequired' }) }]}
              >
                <Input style={{ width: '100%' }} />
              </Form.Item>
            </div>
            <div className="rayAddForm_info_horizontal">
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <p style={{ marginBottom: '15px' }}>{intl.formatMessage({ id: 'ray.job.submitMode' })}</p>
                <Form.Item label="" name="submitType">
                  <Radio.Group
                    style={{ display: 'flex' }}
                    options={[
                      { value: 'K8sJobMode', label: 'K8sJobMode' },
                      { value: 'HTTPMode', label: 'HTTPMode' },
                    ]}
                  />
                </Form.Item>
              </div>
              <Form.Item label={intl.formatMessage({ id: 'ray.job.entryCpuCount' })} name="programEntryNum">
                <InputNumber min={1} max={256} style={{ width: '100%' }} />
              </Form.Item>
            </div>
            <p className="rayAddForm_head_environment_title" style={{ marginTop: '0' }}>
              {intl.formatMessage({ id: 'ray.job.entryResource' })}
            </p>
            <Form.List name="entryResource">
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
                        name={[name, 'name']}
                        rules={[
                          {
                            required: true,
                            message: intl.formatMessage({ id: 'env.enterName' }),
                          },
                        ]}
                      >
                        <Input style={{ width: '320px' }} placeholder={intl.formatMessage({ id: 'env.enterName' })} />
                      </Form.Item>
                      <p className="list_equal">=</p>
                      <Form.Item
                        {...restField}
                        name={[name, 'value']}
                        rules={[
                          {
                            required: true,
                            message: intl.formatMessage({ id: 'env.enterValue' }),
                          },
                        ]}
                      >
                        <Input style={{ width: '320px' }} placeholder={intl.formatMessage({ id: 'env.enterValue' })} />
                      </Form.Item>
                      <DeleteOutlined style={{ color: '#3f66f5' }} onClick={() => remove(name)} />
                    </Space>
                  ))}
                  <Form.Item>
                    <p className="rayAddForm_head_environment" onClick={() => add()}>
                      <span style={{ marginRight: '4px' }}>
                        <PlusCircleOutlined />
                      </span>
                      {intl.formatMessage({ id: 'ray.job.addResource' })}
                    </p>
                  </Form.Item>
                </>
              )}
            </Form.List>
          </div>
          <div className="rayAddForm_info">
            <p className="rayAddForm_info_title">{intl.formatMessage({ id: 'ray.job.rayClusterConfig' })}</p>
            <div className="rayAddForm_images">
              <Form.Item
                label={intl.formatMessage({ id: 'image.name' })}
                name="jobImagename"
                rules={[{ required: true, message: intl.formatMessage({ id: 'image.enterName' }) }]}
              >
                <Input placeholder={intl.formatMessage({ id: 'image.enterNameFirst' })} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item
                label={intl.formatMessage({ id: 'image.version' })}
                name="jobImageversion"
                rules={[
                  { required: true, message: intl.formatMessage({ id: 'image.enterVersion' }) },
                  { validator: validateImage },
                ]}
              >
                <Input
                  showCount
                  maxLength={128}
                  placeholder={intl.formatMessage({ id: 'image.enterVersionFirst' })}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </div>
            <p className="rayAddForm_head">{intl.formatMessage({ id: 'ray.job.headConfig' })}</p>
            <p style={{ color: 'red', fontSize: '12px', marginBottom: '8px' }}>
              {intl.formatMessage({ id: 'ray.job.headNote' })}
            </p>
            <Form.Item label="">
              <Table
                components={jobHeadComponents}
                rowClassName={() => 'editable-row'}
                bordered
                dataSource={jobHeadTableData}
                columns={headColumns}
                className="defaultJobAddForm"
                pagination={false}
              />
            </Form.Item>
            <Form.Item className="defaultJobAddForm" label="" name="check" valuePropName="checked">
              <Checkbox>{intl.formatMessage({ id: 'ray.job.allowHeadWorkload' })}</Checkbox>
            </Form.Item>
            <div className="volumesContent defaultJobAddForm" style={{ marginTop: '32px' }}>
              <p>volumes</p>
              <p>volumeMounts</p>
            </div>
            <div className="rayAddForm_images">
              <Form.Item name="headVolumes defaultJobAddForm" label="">
                <CodeMirror
                  value={jobHeadVolumesYaml}
                  extensions={[javascript()]}
                  onChange={(value, viewUpdate) => setJobHeadVolumesYaml(value)}
                />
              </Form.Item>
              <Form.Item name="headVolumeMounts" label="">
                <CodeMirror
                  value={jobHeadVolumeMountsYaml}
                  extensions={[javascript()]}
                  onChange={(value, viewUpdate) => setJobHeadVolumeMountsYaml(value)}
                />
              </Form.Item>
            </div>
            <p className="rayAddForm_head_environment_title">{intl.formatMessage({ id: 'env.title' })}</p>
            <Form.List name="headEnv">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space
                      key={key}
                      className="defaultJobAddForm"
                      style={{
                        display: 'flex',
                      }}
                      align="baseline"
                    >
                      <Form.Item
                        {...restField}
                        className="defaultJobAddForm"
                        name={[name, 'name']}
                        rules={[
                          {
                            required: true,
                            message: intl.formatMessage({ id: 'env.enterName' }),
                          },
                        ]}
                      >
                        <Input
                          className="defaultJobAddForm"
                          style={{ width: '320px' }}
                          placeholder={intl.formatMessage({ id: 'env.enterName' })}
                        />
                      </Form.Item>
                      <p className="list_equal">=</p>
                      <Form.Item
                        {...restField}
                        className="defaultJobAddForm"
                        name={[name, 'value']}
                        rules={[
                          {
                            required: true,
                            message: intl.formatMessage({ id: 'env.enterValue' }),
                          },
                        ]}
                      >
                        <Input
                          className="defaultJobAddForm"
                          style={{ width: '320px' }}
                          placeholder={intl.formatMessage({ id: 'env.enterValue' })}
                        />
                      </Form.Item>
                      <DeleteOutlined style={{ color: '#3f66f5' }} onClick={() => remove(name)} />
                    </Space>
                  ))}
                  <Form.Item>
                    <p className="rayAddForm_head_environment defaultJobAddForm" onClick={() => add()}>
                      <span style={{ marginRight: '4px' }}>
                        <PlusCircleOutlined />
                      </span>
                      {intl.formatMessage({ id: 'env.add' })}
                    </p>
                  </Form.Item>
                </>
              )}
            </Form.List>
            <p className="split_line" style={{ marginBottom: '32px' }}></p>
            <p className="rayAddForm_head">{intl.formatMessage({ id: 'ray.job.workerConfig' })}</p>
            <p style={{ color: 'red', fontSize: '12px', marginBottom: '8px' }}>
              {intl.formatMessage({ id: 'ray.job.workerNote' })}
            </p>
            <div>
              <Table
                components={JobWorkComponents}
                rowClassName={() => 'editable-row'}
                bordered
                dataSource={jobWorkTableData}
                columns={jobWorkColumns}
                pagination={true}
              />
              <Button
                className="rayAddForm_work_add_button"
                onClick={handleAddJobWorkTable}
                type="dashed"
                icon={<PlusCircleOutlined />}
              >
                {intl.formatMessage({ id: 'ray.job.addNodeConfig' })}
              </Button>
              <div className="volumesContent" style={{ marginTop: '32px' }}>
                <p>volumes</p>
                <p>volumeMounts</p>
              </div>
              <div className="rayAddForm_images">
                <Form.Item name="workVolumes" label="">
                  <CodeMirror
                    value={jobWorkVolumesYaml}
                    extensions={[javascript()]}
                    onChange={(value, viewUpdate) => setJobWorkVolumesYaml(value)}
                  />
                </Form.Item>
                <Form.Item name="workVolumeMounts" label="">
                  <CodeMirror
                    value={jobWorkVolumeMountsYaml}
                    extensions={[javascript()]}
                    onChange={(value, viewUpdate) => setJobWorkVolumeMountsYaml(value)}
                  />
                </Form.Item>
              </div>
              <p className="rayAddForm_head_environment_title">{intl.formatMessage({ id: 'env.title' })}</p>
              <Form.List name="workEnv">
                {(fields, { add, remove }) => (
                  <>
                    {fields.map(({ key, name, ...restField }) => (
                      <Space
                        key={key}
                        className="defaultJobAddForm2"
                        style={{
                          display: 'flex',
                        }}
                        align="baseline"
                      >
                        <Form.Item
                          {...restField}
                          className="defaultJobAddForm2"
                          name={[name, 'name']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'env.enterName' }),
                            },
                          ]}
                        >
                          <Input
                            className="defaultJobAddForm2"
                            style={{ width: '320px' }}
                            placeholder={intl.formatMessage({ id: 'env.enterName' })}
                          />
                        </Form.Item>
                        <p className="list_equal">=</p>
                        <Form.Item
                          {...restField}
                          name={[name, 'value']}
                          className="defaultJobAddForm2"
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'env.enterValue' }),
                            },
                          ]}
                        >
                          <Input
                            className="defaultJobAddForm2"
                            style={{ width: '320px' }}
                            placeholder={intl.formatMessage({ id: 'env.enterValue' })}
                          />
                        </Form.Item>
                        <DeleteOutlined
                          className="defaultJobAddForm2"
                          style={{ color: '#3f66f5' }}
                          onClick={() => remove(name)}
                        />
                      </Space>
                    ))}
                    <Form.Item>
                      <p className="rayAddForm_head_environment defaultJobAddForm2" onClick={() => add()}>
                        <span style={{ marginRight: '4px' }}>
                          <PlusCircleOutlined />
                        </span>
                        {intl.formatMessage({ id: 'env.add' })}
                      </p>
                    </Form.Item>
                  </>
                )}
              </Form.List>
            </div>
          </div>
        </Form>
      </div>
      <div className="rayAddFormButtons">
        <Button className="cancel_btn" onClick={() => goBack()}>
          {intl.formatMessage({ id: 'common.cancel' })}
        </Button>
        <Button className="primary_btn ml" htmlType="submit" onClick={() => onFinish(false)}>
          {intl.formatMessage({ id: 'ray.job.createTemplate' })}
        </Button>
        <Button className="primary_btn ml" htmlType="submit" onClick={() => onFinish(true)}>
          {intl.formatMessage({ id: 'ray.job.createAndStart' })}
        </Button>
      </div>
    </div>
  );
}
