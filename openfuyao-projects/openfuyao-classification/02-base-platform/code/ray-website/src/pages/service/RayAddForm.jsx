/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Space, Select, Checkbox, Table, Button, Form, Input, message, Popconfirm } from 'antd';

import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { useHistory } from 'inula-router';
import Inula, { useEffect, useState, createRef, useContext, useRef } from 'openinula';
import { PlusCircleOutlined, DeleteOutlined, CopyOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { workHeadConfigTh, jsonToYaml, yamlTojson, validateName, validateImage } from '@/tools/utils';
import { npuTypeOption, serviceAddFormDefault, serviceAddFormWorkGroupData } from '@/common/constants';
import { addService } from '@/api/rayApi.js';
import { useIntl } from 'inula-intl';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import '@/styles/components/codeMirrorEditor.less';
export default function RayAddForm() {
  const intl = useIntl();
  const placeholder = intl.formatMessage({ id: 'common.pleaseEnter' });
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const history = useHistory();
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const [messageApi, contextHolder] = message.useMessage();
  const [addServiceForm] = Form.useForm();
  const [isWorkCreate, setIsWorkCreate] = useState(false);
  const [isHeadCreate, setIsHeadCreate] = useState(false);
  const [serveYaml, setServeYaml] = useState('');
  const [headVolumesYaml, setHeadVolumesYaml] = useState();
  const [headVolumeMountsYaml, setHeadVolumeMountsYaml] = useState();
  const [workVolumesYaml, setWorkVolumesYaml] = useState();
  const [workVolumeMountsYaml, setWorkVolumeMountsYaml] = useState();
  // 添加节点规格配置操作
  const EditableContextHeadService = Inula.createContext(null);
  const EditableRowHeadService = ({ index, ...props }) => {
    const [form] = Form.useForm();
    return (
      <Form form={form} component={false}>
        <EditableContextHeadService.Provider value={form}>
          <tr {...props} />
        </EditableContextHeadService.Provider>
      </Form>
    );
  };
  const EditableCellHeadService = ({
    title,
    editable,
    children,
    dataIndex,
    record,
    handleSaveHeadTable,
    ...restProps
  }) => {
    const [serviceHeadEditing, setServiceHeadEditing] = useState(false);
    const serviceHeadSelectRef = useRef(null);
    const serviceHeadInputRef = useRef(null);
    const form = useContext(EditableContextHeadService);

    useEffect(() => {
      if (serviceHeadEditing) {
        if (dataIndex.includes('npuType')) {
          serviceHeadSelectRef.current?.focus();
        } else {
          serviceHeadInputRef.current?.focus();
        }
      }
    }, [serviceHeadEditing, dataIndex]);

    const toggleEditServiceHead = () => {
      setServiceHeadEditing(!serviceHeadEditing);
      form.setFieldsValue({
        [dataIndex]: record[dataIndex],
      });
    };

    const save = async () => {
      try {
        const values = await form.validateFields();
        toggleEditServiceHead();
        handleSaveHeadTable({
          ...record,
          ...values,
        });
      } catch (errInfo) {
        console.log('Validate Failed:', errInfo);
      }
    };

    let childNode = children;

    if (editable) {
      const inputComponent = !dataIndex.includes('npuType') ? (
        <Input ref={serviceHeadInputRef} onBlur={save} />
      ) : (
        <Select ref={serviceHeadSelectRef} onBlur={save} style={{ width: 120 }} options={npuTypeOption} />
      );

      let validator;
      if (dataIndex.includes('nodeConfigName')) {
        validator = validateName;
      } else if (dataIndex.includes('npuType')) {
        validator = '';
      } else {
        validator = validateHeadService;
      }

      childNode = serviceHeadEditing ? (
        <Form.Item style={{ margin: 0 }} name={dataIndex} rules={[{ validator: validator }]}>
          {inputComponent}
        </Form.Item>
      ) : (
        <div
          className="editable-cell-value-wrap defaultServiceHead"
          style={{
            paddingInlineEnd: 24,
          }}
          onClick={toggleEditServiceHead}
        >
          {children}
        </div>
      );
    }

    return <td {...restProps}>{childNode}</td>;
  };
  const validateHeadService = async (_rule, value) => {
    const regex = /^\d+$/;
    if (value && !regex.test(value)) {
      return Promise.reject(
        new Error(intl.formatMessage({ id: 'table.shouldBeNumber' }, { field: workHeadConfigTh(_rule.field) })),
      );
    } else {
      return Promise.resolve();
    }
  };
  const [serviceHeadTableData, setServiceHeadTableData] = useState([
    {
      key: '0',
      nodeType: 'head',
      nodeConfigName: placeholder,
      memory: placeholder,
      maxCopy: '--',
      minCopy: '--',
      npuType: placeholder,
      npu: placeholder,
      cpu: placeholder,
    },
  ]);
  const defaultServiceHeadlTable = [
    {
      title: intl.formatMessage({ id: 'table.nodeType' }),
      key: 'serviceHeadType',
      dataIndex: 'nodeType',
      render: (_, record) => record.nodeType || placeholder,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.nodeConfigName' })}
        </p>
      ),
      key: 'serviceHeadName',
      dataIndex: 'nodeConfigName',
      editable: true,
      render: (_, record) => record.nodeConfigName || placeholder,
    },
    {
      title: intl.formatMessage({ id: 'table.npuCard' }),
      key: 'serviceHeadNpu',
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
      key: 'serviceHeadCpu',
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
      key: 'serviceHeadMemory',
      dataIndex: 'memory',
      editable: true,
      render: (_, record) => record.memory || placeholder,
    },
    {
      title: <p style={{ color: '#b0a7a7' }}>{intl.formatMessage({ id: 'table.maxReplicas' })}</p>,
      key: 'serviceHeadMaxCopy',
      dataIndex: 'maxCopy',
      editable: false,
      render: (_, record) => record.maxCopy || '--',
    },
    {
      title: <p style={{ color: '#b0a7a7' }}>{intl.formatMessage({ id: 'table.minReplicas' })}</p>,
      key: 'serviceHeadMinCopy',
      dataIndex: 'minCopy',
      editable: false,
      render: (_, record) => record.minCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.npuType' }),
      key: 'serviceHeadType',
      dataIndex: 'npuType',
      editable: true,
      render: (_, record) => record.npuType || placeholder,
    },
  ];
  const handleSaveHeadTable = (row) => {
    const newData = [...serviceHeadTableData];
    const index = newData.findIndex((item) => row.key === item.key);
    const item = newData[index];
    newData.splice(index, 1, {
      ...item,
      ...row,
    });
    setServiceHeadTableData(newData);
  };
  const headComponentsService = {
    body: {
      row: EditableRowHeadService,
      cell: EditableCellHeadService,
    },
  };
  const headColumnsService = defaultServiceHeadlTable.map((col) => {
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
        handleSaveHeadTable,
      }),
    };
  });

  // 添加work节点规格配置操作

  const EditableWorkContextService = Inula.createContext(null);
  const EditWorkTableRowService = ({ index, ...props }) => {
    const [form] = Form.useForm();
    return (
      <Form form={form} component={false}>
        <EditableWorkContextService.Provider value={form}>
          <tr {...props} />
        </EditableWorkContextService.Provider>
      </Form>
    );
  };
  const EditWorkTableCellService = ({
    title,
    editable,
    children,
    dataIndex,
    record,
    handleSaveWorkTable,
    ...restProps
  }) => {
    const [serviceWorkEditing, setServiceWorkEditing] = useState(false);
    const serviceWorkInputRef = useRef(null);
    const serviceWokrSelectRef = useRef(null);
    const form = useContext(EditableContext);

    useEffect(() => {
      if (serviceWorkEditing) {
        if (dataIndex.includes('npuType')) {
          serviceWokrSelectRef.current?.focus();
        } else {
          serviceWorkInputRef.current?.focus();
        }
      }
    }, [serviceWorkEditing, dataIndex]);

    const toggleEdit = () => {
      setServiceWorkEditing(!serviceWorkEditing);
      form.setFieldsValue({
        [dataIndex]: record[dataIndex],
      });
    };

    const save = async () => {
      try {
        const values = await form.validateFields();
        setServiceWorkEditing(false);
        handleSaveWorkTable({
          ...record,
          ...values,
        });
      } catch (errInfo) {
        console.log('Save failed:', errInfo);
      }
    };

    let childNode = children;

    if (editable) {
      if (serviceWorkEditing) {
        const isNpuType = dataIndex.includes('npuType');
        const inputComponent = isNpuType ? (
          <Select
            ref={serviceWokrSelectRef}
            onBlur={save}
            onChange={save}
            style={{ width: 120 }}
            options={npuTypeOption}
          />
        ) : (
          <Input ref={serviceWorkInputRef} onPressEnter={save} onBlur={save} />
        );

        let validationRule;
        if (dataIndex.includes('nodeConfigName')) {
          validationRule = validateName;
        } else if (isNpuType) {
          validationRule = null;
        } else {
          validationRule = validateWorkService;
        }
        const rules = validationRule ? [{ validator: validationRule }] : [];

        childNode = (
          <Form.Item style={{ margin: 0 }} name={dataIndex} rules={rules}>
            {inputComponent}
          </Form.Item>
        );
      } else {
        childNode = (
          <div
            className="editable-cell-value-wrap defaultServiceWork"
            style={{
              paddingInlineEnd: 24,
            }}
            onClick={toggleEdit}
          >
            {children}
          </div>
        );
      }
    }

    return <td {...restProps}>{childNode}</td>;
  };

  const validateWorkService = async (_rule, value) => {
    const regex = /^\d+$/;
    if (value && !regex.test(value)) {
      return Promise.reject(
        new Error(intl.formatMessage({ id: 'table.shouldBeNumber' }, { field: workHeadConfigTh(_rule.field) })),
      );
    } else {
      return Promise.resolve();
    }
  };
  const [serviceWorkTableData, setServiceWorkTableData] = useState([
    {
      key: '0',
      nodeType: 'worker',
      nodeConfigName: placeholder,
      maxCopy: placeholder,
      minCopy: placeholder,
      npu: placeholder,
      cpu: placeholder,
      memory: placeholder,
      npuType: placeholder,
    },
  ]);
  const [count, setCount] = useState(1);
  const handleDeleteWorkTableService = (key) => {
    const newData = serviceWorkTableData.filter((item) => item.key !== key);
    setServiceWorkTableData(newData);
  };
  const defaultServiceWorklTable = [
    {
      title: intl.formatMessage({ id: 'table.nodeType' }),
      key: 'serviceWorkType',
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
      key: 'serviceWorkName',
      dataIndex: 'nodeConfigName',
      editable: true,
      render: (_, record) => record.nodeConfigName,
    },
    {
      title: intl.formatMessage({ id: 'table.npuCard' }),
      key: 'serviceWorkNpu',
      dataIndex: 'npu',
      editable: true,
      render: (_, record) => record.npu,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {'CPU (Core)'}
        </p>
      ),
      key: 'serviceWorkCpu',
      dataIndex: 'cpu',
      editable: true,
      render: (_, record) => record.cpu,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.memoryG' })}
        </p>
      ),
      key: 'serviceWorkMemory',
      dataIndex: 'memory',
      editable: true,
      render: (_, record) => record.memory,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.maxReplicas' })}
        </p>
      ),
      key: 'serviceWorkMaxCopy',
      dataIndex: 'maxCopy',
      editable: true,
      render: (_, record) => record.maxCopy,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {intl.formatMessage({ id: 'table.minReplicas' })}
        </p>
      ),
      key: 'serviceWorkMinCopy',
      dataIndex: 'minCopy',
      editable: true,
      render: (_, record) => record.minCopy,
    },
    {
      title: intl.formatMessage({ id: 'table.npuType' }),
      key: 'serviceWorkType',
      dataIndex: 'npuType',
      editable: true,
      render: (_, record) => record.npuType,
    },
    {
      title: intl.formatMessage({ id: 'common.action' }),
      dataIndex: 'operation',
      render: (_, record) =>
        serviceWorkTableData.length > 1 ? (
          <Popconfirm
            title={intl.formatMessage({ id: 'common.confirmDelete' })}
            onConfirm={() => handleDeleteWorkTableService(record.key)}
          >
            <DeleteOutlined style={{ color: '#3f66f5' }} />
          </Popconfirm>
        ) : (
          <DeleteOutlined style={{ color: '#89939b' }} />
        ),
    },
  ];
  const handleAddWorkTable = () => {
    const newData = {
      key: count,
      nodeType: 'worker',
      nodeConfigName: placeholder,
      cpu: placeholder,
      memory: placeholder,
      maxCopy: placeholder,
      minCopy: placeholder,
      npu: placeholder,
      npuType: placeholder,
    };
    setServiceWorkTableData([...serviceWorkTableData, newData]);
    setCount(count + 1);
  };
  const handleSaveWorkTable = (row) => {
    const newData = [...serviceWorkTableData];
    const index = newData.findIndex((item) => row.key === item.key);
    const item = newData[index];
    newData.splice(index, 1, {
      ...item,
      ...row,
    });
    setServiceWorkTableData(newData);
  };
  const workComponentsService = {
    body: {
      row: EditWorkTableRowService,
      cell: EditWorkTableCellService,
    },
  };
  const workColumnsService = defaultServiceWorklTable.map((col) => {
    if (!col.editable) {
      return col;
    }
    return {
      ...col,
      onCell: (record) => ({
        record,
        editable: col.editable,
        title: col.title,
        dataIndex: col.dataIndex,
        handleSaveWorkTable,
      }),
    };
  });
  const onFinish = (data) => {
    if (serveYaml) {
      if (isWorkCreate && isHeadCreate) {
        submitAdd(data);
      } else {
        messageApi.error(intl.formatMessage({ id: 'ray.service.formError' }));
      }
    } else {
      messageApi.error(intl.formatMessage({ id: 'ray.service.enterServeConfig' }));
    }
  };
  // 调用创建接口
  const submitAdd = async (data) => {
    let serviceFormData = serviceAddFormDefault;
    serviceFormData.metadata.name = addServiceForm.getFieldsValue().serviceName;
    serviceFormData.spec.rayClusterConfig.rayVersion = addServiceForm.getFieldsValue().serviceImageversion;
    if (addServiceForm.getFieldsValue().check) {
      if (
        Object.prototype.hasOwnProperty.call(
          serviceFormData.spec.rayClusterConfig.headGroupSpec.rayStartParams,
          'num-cpus',
        )
      ) {
        delete serviceFormData.spec.rayClusterConfig.headGroupSpec.rayStartParams['num-cpus'];
      }
      if (serviceHeadTableData[0].npu !== '请输入' && serviceHeadTableData[0].npu !== '') {
        serviceFormData.spec.rayClusterConfig.headGroupSpec.rayStartParams.resources = `{"NPU":${serviceHeadTableData[0].npu}}`;
      }
      if (serviceHeadTableData[0].npuType !== '请输入' && serviceHeadTableData[0].npuType !== '') {
        serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].schedulerName = 'volcano';
      }
    } else {
      serviceFormData.spec.rayClusterConfig.headGroupSpec.rayStartParams['num-cpus'] = '0';
    }
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].name =
      serviceHeadTableData[0].nodeConfigName;
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].image = `${addServiceForm.getFieldsValue().serviceImagename}:${addServiceForm.getFieldsValue().serviceImageversion}`;
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].resources.limits.cpu =
      serviceHeadTableData[0].cpu;
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].resources.limits.memory = `${serviceHeadTableData[0].memory}G`;
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].resources.requests.cpu =
      serviceHeadTableData[0].cpu;
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].resources.requests.memory = `${serviceHeadTableData[0].memory}G`;
    if (addServiceForm.getFieldsValue().headVolumes) {
      serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.volumes = [
        yamlTojson(addServiceForm.getFieldsValue().headVolumes),
      ];
    }
    if (addServiceForm.getFieldsValue().headVolumeMounts) {
      serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].volumeMounts = [
        yamlTojson(addServiceForm.getFieldsValue().headVolumeMounts),
      ];
    }
    if (serviceHeadTableData[0].npu !== '请输入' && serviceHeadTableData[0].npu !== '') {
      serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].resources.limits[
        serviceHeadTableData[0].npuType
      ] = serviceHeadTableData[0].npu;
      serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].resources.requests[
        serviceHeadTableData[0].npuType
      ] = serviceHeadTableData[0].npu;
    }
    serviceFormData.spec.rayClusterConfig.headGroupSpec.template.spec.containers[0].env =
      addServiceForm.getFieldsValue().headEnvService;
    let workGroupSpecs = [];
    serviceWorkTableData.forEach((item) => {
      let serviceDefaultWorkSingleData = serviceAddFormWorkGroupData;
      serviceDefaultWorkSingleData.maxReplicas = Number(item.maxCopy);
      serviceDefaultWorkSingleData.minReplicas = Number(item.minCopy);
      serviceDefaultWorkSingleData.template.spec.containers[0].name = item.nodeConfigName;
      serviceDefaultWorkSingleData.template.spec.containers[0].image = `${addServiceForm.getFieldsValue().serviceImagename}:${addServiceForm.getFieldsValue().serviceImageversion}`;
      serviceDefaultWorkSingleData.template.spec.containers[0].resources.limits.cpu = item.cpu;
      serviceDefaultWorkSingleData.template.spec.containers[0].resources.limits.memory = `${item.memory}G`;
      serviceDefaultWorkSingleData.template.spec.containers[0].resources.requests.cpu = item.cpu;
      serviceDefaultWorkSingleData.template.spec.containers[0].resources.requests.memory = `${item.memory}G`;
      if (item.npu !== '请输入' && item.npu !== '') {
        serviceDefaultWorkSingleData.template.spec.containers[0].resources.limits[item.npuType] = item.npu;
        serviceDefaultWorkSingleData.template.spec.containers[0].resources.requests[item.npuType] = item.npu;
      }
      serviceDefaultWorkSingleData.template.spec.containers[0].env = addServiceForm.getFieldsValue().workEnvService;
      if (addServiceForm.getFieldsValue().workVolumes) {
        serviceDefaultWorkSingleData.template.spec.volumes = [yamlTojson(addServiceForm.getFieldsValue().workVolumes)];
      }
      if (addServiceForm.getFieldsValue().workVolumeMounts) {
        serviceDefaultWorkSingleData.template.spec.containers[0].volumeMounts = [
          yamlTojson(addServiceForm.getFieldsValue().workVolumeMounts),
        ];
      }
      workGroupSpecs.push(serviceDefaultWorkSingleData);
    });
    serviceFormData.spec.serveConfigV2 = yamlTojson(`|` + `\n` + addServiceForm.getFieldsValue().serviceConfig);
    serviceFormData.spec.rayClusterConfig.workerGroupSpecs = workGroupSpecs;
    const appFormData = new FormData();
    appFormData.append('config', jsonToYaml(JSON.stringify(serviceFormData)));
    appFormData.append('createBy', creator);
    appFormData.append('isStart', data);
    try {
      const res = await addService(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(`创建成功！`);
        setTimeout(() => {
          history.push('/RayService');
        }, 500);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`创建失败,${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`创建失败,${msg}`);
      }
    }
  };
  const handleCopyServeConfig = () => {
    copy(addServiceForm.getFieldsValue().serviceConfig);
    messageApi.success(intl.formatMessage({ id: 'ray.service.copySuccess' }));
  };
  const goBack = () => {
    history.push('/RayService');
  };
  useEffect(() => {
    let checkWorkvalueService = [];
    serviceWorkTableData.forEach((item) => {
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
        checkWorkvalueService.push('true');
      } else {
        checkWorkvalueService.push('false');
      }
    });
    if (checkWorkvalueService.includes('false')) {
      setIsWorkCreate(false);
    } else {
      setIsWorkCreate(true);
    }
  }, [serviceWorkTableData]);
  useEffect(() => {
    let checkHeadvalueService = [];
    serviceHeadTableData.forEach((item) => {
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
        checkHeadvalueService.push('true');
      } else {
        checkHeadvalueService.push('false');
      }
    });
    if (checkHeadvalueService.includes('false')) {
      setIsHeadCreate(false);
    } else {
      setIsHeadCreate(true);
    }
  }, [serviceHeadTableData]);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="rayAddForm">
      {contextHolder}
      <div className="rayAddFormContent">
        <Form
          form={addServiceForm}
          initialValues={{ remember: true }}
          onFinish={onFinish}
          autoComplete="off"
          layout={'vertical'}
        >
          <div className="rayAddForm_info">
            <p className="rayAddForm_info_title">{intl.formatMessage({ id: 'ray.service.basicInfo' })}</p>
            <div className="rayAddForm_images">
              <Form.Item
                label={intl.formatMessage({ id: 'ray.service.serviceName' })}
                name="serviceName"
                rules={[
                  { required: true, message: intl.formatMessage({ id: 'ray.service.enterServiceName' }) },
                  { validator: validateName },
                ]}
              >
                <Input showCount maxLength={253} style={{ width: '100%' }} />
              </Form.Item>
            </div>
          </div>
          <div className="rayAddForm_info">
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <p className="rayAddForm_info_title">{intl.formatMessage({ id: 'ray.service.serveConfig' })}</p>
              <div className="tool_word_group" onClick={handleCopyServeConfig}>
                <CopyOutlined className="common_antd_icon primary_color" />
                <span>{intl.formatMessage({ id: 'ray.service.copy' })}</span>
              </div>
            </div>
            <Form.Item name="serviceConfig" label="">
              <CodeMirror
                value={serveYaml}
                extensions={[javascript()]}
                onChange={(value, viewUpdate) => setServeYaml(value)}
              />
            </Form.Item>
          </div>
          <div className="rayAddForm_info">
            <p className="rayAddForm_info_title">{intl.formatMessage({ id: 'ray.service.rayClusterConfig' })}</p>
            <div className="rayAddForm_images">
              <Form.Item
                label={intl.formatMessage({ id: 'image.name' })}
                name="serviceImagename"
                rules={[{ required: true, message: intl.formatMessage({ id: 'image.enterName' }) }]}
              >
                <Input placeholder={intl.formatMessage({ id: 'image.enterNameFirst' })} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item
                label={intl.formatMessage({ id: 'image.version' })}
                name="serviceImageversion"
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
            <p className="rayAddForm_head" style={{ fontWeight: 'normal' }}>
              {intl.formatMessage({ id: 'ray.service.headConfig' })}
            </p>
            <p style={{ color: 'red', fontSize: '12px', marginBottom: '8px' }}>
              {intl.formatMessage({ id: 'ray.service.headNote' })}
            </p>
            <Form.Item label="">
              <Table
                components={headComponentsService}
                rowClassName={() => 'editable-row'}
                bordered
                dataSource={serviceHeadTableData}
                columns={headColumnsService}
                className="defaultServiceAddForm"
                pagination={false}
              />
            </Form.Item>
            <Form.Item className="defaultServiceAddForm" label="" name="check" valuePropName="checked">
              <Checkbox>{intl.formatMessage({ id: 'ray.service.allowHeadWorkload' })}</Checkbox>
            </Form.Item>
            <div className="volumesContent defaultServiceAddForm" style={{ marginTop: '32px' }}>
              <p>volumes</p>
              <p>volumeMounts</p>
            </div>
            <div className="rayAddForm_images">
              <Form.Item name="headVolumes defaultServiceAddForm" label="">
                <CodeMirror
                  value={headVolumesYaml}
                  extensions={[javascript()]}
                  onChange={(value, viewUpdate) => setHeadVolumesYaml(value)}
                />
              </Form.Item>
              <Form.Item name="headVolumeMounts defaultServiceAddForm" label="">
                <CodeMirror
                  value={headVolumeMountsYaml}
                  extensions={[javascript()]}
                  className="defaultServiceAddForm"
                  onChange={(value, viewUpdate) => setHeadVolumeMountsYaml(value)}
                />
              </Form.Item>
            </div>
            <p className="rayAddForm_head_environment_title defaultServiceAddForm">
              {intl.formatMessage({ id: 'env.title' })}
            </p>
            <Form.List name="headEnvService">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space
                      key={key}
                      className="defaultServiceAddForm"
                      style={{
                        display: 'flex',
                      }}
                      align="baseline"
                    >
                      <Form.Item
                        {...restField}
                        className="defaultServiceAddForm"
                        name={[name, 'name']}
                        rules={[
                          {
                            required: true,
                            message: intl.formatMessage({ id: 'env.enterName' }),
                          },
                        ]}
                      >
                        <Input
                          className="defaultServiceAddForm"
                          style={{ width: '320px' }}
                          placeholder={intl.formatMessage({ id: 'env.enterName' })}
                        />
                      </Form.Item>
                      <p className="list_equal">=</p>
                      <Form.Item
                        {...restField}
                        className="defaultServiceAddForm"
                        name={[name, 'value']}
                        rules={[
                          {
                            required: true,
                            message: intl.formatMessage({ id: 'env.enterValue' }),
                          },
                        ]}
                      >
                        <Input
                          className="defaultServiceAddForm"
                          style={{ width: '320px' }}
                          placeholder={intl.formatMessage({ id: 'env.enterValue' })}
                        />
                      </Form.Item>
                      <DeleteOutlined
                        className="defaultServiceAddForm"
                        style={{ color: '#3f66f5' }}
                        onClick={() => remove(name)}
                      />
                    </Space>
                  ))}
                  <Form.Item>
                    <p className="rayAddForm_head_environment defaultServiceAddForm" onClick={() => add()}>
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
            <p className="rayAddForm_head" style={{ fontWeight: 'normal' }}>
              {intl.formatMessage({ id: 'ray.service.workerConfig' })}
            </p>
            <p style={{ color: 'red', fontSize: '12px', marginBottom: '8px' }}>
              {intl.formatMessage({ id: 'ray.service.workerNote' })}
            </p>
            <div>
              <Table
                components={workComponentsService}
                rowClassName={() => 'editable-row'}
                bordered
                dataSource={serviceWorkTableData}
                columns={workColumnsService}
                pagination={true}
              />
              <Button
                className="rayAddForm_work_add_button defaultServiceAddForm"
                onClick={handleAddWorkTable}
                type="dashed"
                icon={<PlusCircleOutlined />}
              >
                {intl.formatMessage({ id: 'ray.service.addNodeConfig' })}
              </Button>
              <div className="volumesContent defaultServiceAddForm" style={{ marginTop: '32px' }}>
                <p>volumes</p>
                <p>volumeMounts</p>
              </div>
              <div className="rayAddForm_images defaultServiceAddForm">
                <Form.Item name="workVolumes" label="">
                  <CodeMirror
                    value={workVolumesYaml}
                    extensions={[javascript()]}
                    onChange={(value, viewUpdate) => setWorkVolumesYaml(value)}
                  />
                </Form.Item>
                <Form.Item name="workVolumeMounts defaultServiceAddForm" label="">
                  <CodeMirror
                    value={workVolumeMountsYaml}
                    extensions={[javascript()]}
                    onChange={(value, viewUpdate) => setWorkVolumeMountsYaml(value)}
                  />
                </Form.Item>
              </div>
              <p className="rayAddForm_head_environment_title defaultServiceAddForm">
                {intl.formatMessage({ id: 'env.title' })}
              </p>
              <Form.List name="workEnvService">
                {(fields, { add, remove }) => (
                  <>
                    {fields.map(({ key, name, ...restField }) => (
                      <Space
                        key={key}
                        className="defaultServiceAddForm2"
                        style={{
                          display: 'flex',
                        }}
                        align="baseline"
                      >
                        <Form.Item
                          {...restField}
                          name={[name, 'name']}
                          className="defaultServiceAddForm2"
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'env.enterName' }),
                            },
                          ]}
                        >
                          <Input
                            className="defaultServiceAddForm2"
                            style={{ width: '320px' }}
                            placeholder={intl.formatMessage({ id: 'env.enterName' })}
                          />
                        </Form.Item>
                        <p className="list_equal">=</p>
                        <Form.Item
                          {...restField}
                          className="defaultServiceAddForm2"
                          name={[name, 'value']}
                          rules={[
                            {
                              required: true,
                              message: intl.formatMessage({ id: 'env.enterValue' }),
                            },
                          ]}
                        >
                          <Input
                            className="defaultServiceAddForm2"
                            style={{ width: '320px' }}
                            placeholder={intl.formatMessage({ id: 'env.enterValue' })}
                          />
                        </Form.Item>
                        <DeleteOutlined
                          className="defaultServiceAddForm2"
                          style={{ color: '#3f66f5' }}
                          onClick={() => remove(name)}
                        />
                      </Space>
                    ))}
                    <Form.Item>
                      <p className="rayAddForm_head_environment defaultServiceAddForm2" onClick={() => add()}>
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
          {intl.formatMessage({ id: 'ray.service.createTemplate' })}
        </Button>
        <Button className="primary_btn ml" htmlType="submit" onClick={() => onFinish(true)}>
          {intl.formatMessage({ id: 'ray.service.createAndStart' })}
        </Button>
      </div>
    </div>
  );
}
