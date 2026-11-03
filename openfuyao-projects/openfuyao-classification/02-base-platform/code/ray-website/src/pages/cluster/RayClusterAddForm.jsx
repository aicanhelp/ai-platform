/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Space, Select, Checkbox, Table, Button, Form, message, Input, Popconfirm } from 'antd';

import { ResponseCode } from '@/common/constants';
import { useHistory } from 'inula-router';
import Inula, { useEffect, useState, useContext, useRef } from 'openinula';
import { useIntl } from 'inula-intl';
import { PlusCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import { workHeadConfigTh, jsonToYaml, yamlTojson, validateName, validateImage } from '@/tools/utils';
import { npuTypeOption, addFormDefault, addFormWorkGroupData } from '@/common/constants';
import { addCluster } from '@/api/rayApi.js';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import '@/styles/components/codeMirrorEditor.less';
export default function RayClusterAddForm() {
  const intl = useIntl();
  const placeholder = intl.formatMessage({ id: 'common.pleaseEnter' });
  const history = useHistory();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const creator = localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).name : '';
  const [messageApi, contextHolder] = message.useMessage();
  const [addClusterForm] = Form.useForm();
  const [isWorkCreate, setIsWorkCreate] = useState(false);
  const [isHeadCreate, setIsHeadCreate] = useState(false);
  const [isStart, setIsStart] = useState(false);
  const [headVolumesYaml, setHeadVolumesYaml] = useState();
  const [headVolumeMountsYaml, setHeadVolumeMountsYaml] = useState();
  const [workVolumesYaml, setWorkVolumesYaml] = useState();
  const [workVolumeMountsYaml, setWorkVolumeMountsYaml] = useState();
  // head节点规格配置操作
  const EditableContextHead = Inula.createContext(null);
  const EditableRowHead = ({ index, ...props }) => {
    const [form] = Form.useForm();
    return (
      <Form form={form} component={false}>
        <EditableContextHead.Provider value={form}>
          <tr {...props} />
        </EditableContextHead.Provider>
      </Form>
    );
  };
  const EditableCellHead = ({ title, editable, children, dataIndex, record, handleSaveHeadTable, ...restProps }) => {
    const [clusterHeadEditing, setClusterHeadEditing] = useState(false);
    const clusterHeadInputRef = useRef(null);
    const clusterHeadSelectRef = useRef(null);
    const form = useContext(EditableContextHead);
    useEffect(() => {
      if (clusterHeadEditing) {
        clusterHeadInputRef.current?.focus();
        clusterHeadSelectRef.current?.focus();
      }
    }, [clusterHeadEditing]);
    const toggleEditClusterHead = () => {
      setClusterHeadEditing(!clusterHeadEditing);
      form.setFieldsValue({
        [dataIndex]: record[dataIndex],
      });
    };
    const save = async () => {
      try {
        const values = await form.validateFields();
        toggleEditClusterHead();
        handleSaveHeadTable({
          ...record,
          ...values,
        });
      } catch (errInfo) {
        console.log('Save failed:', errInfo);
      }
    };

    let childNode = children;

    if (editable) {
      if (clusterHeadEditing) {
        let inputElement;
        if (!dataIndex.includes('npuType')) {
          inputElement = <Input ref={clusterHeadInputRef} onBlur={save} />;
        } else {
          inputElement = (
            <Select ref={clusterHeadSelectRef} onBlur={save} style={{ width: 120 }} options={npuTypeOption} />
          );
        }

        let validator;
        if (dataIndex.includes('nodeConfigName')) {
          validator = validateName;
        } else if (!dataIndex.includes('npuType')) {
          validator = validateHead;
        }

        const formItemRules = validator ? [{ validator }] : [];

        childNode = (
          <Form.Item style={{ margin: 0 }} name={dataIndex} rules={formItemRules}>
            {inputElement}
          </Form.Item>
        );
      } else {
        childNode = (
          <div
            className="editable-cell-value-wrap defaultClusterHead"
            style={{
              paddingInlineEnd: 24,
            }}
            onClick={toggleEditClusterHead}
          >
            {children}
          </div>
        );
      }
    }

    return <td {...restProps}>{childNode}</td>;
  };

  const validateHead = async (_rule, value) => {
    const regex = /^\d+$/;
    if (value && !regex.test(value)) {
      return Promise.reject(new Error(`${workHeadConfigTh(_rule.field)}类型应为数字！`));
    } else {
      return Promise.resolve();
    }
  };
  const [clusterHeadTableData, setClusterHeadTableData] = useState([
    {
      key: '0',
      nodeType: 'head',
      cpu: '请输入',
      memory: '请输入',
      maxCopy: '--',
      minCopy: '--',
      nodeConfigName: '请输入',
      npu: '请输入',
      npuType: '请输入',
    },
  ]);
  const defaultClusterHeadlTable = [
    {
      title: '节点类型',
      key: 'clusterHeadType',
      dataIndex: 'nodeType',
      render: (_, record) => record.nodeType || '请输入',
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>节点配置名称
        </p>
      ),
      key: 'clusterHeadName',
      dataIndex: 'nodeConfigName',
      editable: true,
      render: (_, record) => record.nodeConfigName || '请输入',
    },
    {
      title: 'NPU （卡）',
      key: 'clusterHeadNpu',
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
      key: 'clusterHeadCpu',
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
      key: 'clusterHeadMemory',
      dataIndex: 'memory',
      editable: true,
      render: (_, record) => record.memory || '请输入',
    },
    {
      title: <p style={{ color: '#b0a7a7' }}>最大副本数</p>,
      key: 'clusterHeadMaxCopy',
      dataIndex: 'maxCopy',
      editable: false,
      render: (_, record) => record.maxCopy || '--',
    },
    {
      title: <p style={{ color: '#b0a7a7' }}>最小副本数</p>,
      key: 'clusterHeadMinCopy',
      dataIndex: 'minCopy',
      editable: false,
      render: (_, record) => record.minCopy || '--',
    },
    {
      title: 'NPU类型',
      key: 'clusterHeadType',
      dataIndex: 'npuType',
      editable: true,
      render: (_, record) => record.npuType || '请输入',
    },
  ];
  const handleSaveHeadTable = (row) => {
    const newData = [...clusterWorkTableData];
    const index = newData.findIndex((item) => row.key === item.key);
    const item = newData[index];
    newData.splice(index, 1, {
      ...item,
      ...row,
    });
    setClusterHeadTableData(newData);
  };
  const headComponents = {
    body: {
      row: EditableRowHead,
      cell: EditableCellHead,
    },
  };
  const headColumns = defaultClusterHeadlTable.map((col) => {
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

  //添加work节点规格配置操作
  const EditableContext = Inula.createContext(null);
  const EditableRow = ({ index, ...props }) => {
    const [form] = Form.useForm();
    return (
      <Form form={form} component={false}>
        <EditableContext.Provider value={form}>
          <tr {...props} />
        </EditableContext.Provider>
      </Form>
    );
  };
  const EditableCell = ({ title, editable, children, dataIndex, record, handleSaveWorkTable, ...restProps }) => {
    const [clusterWorkEditing, setClusterWorkEditing] = useState(false);
    const clusterWorkInputRef = useRef(null);
    const clusterWorkSelectRef = useRef(null);
    const form = useContext(EditableContext);

    useEffect(() => {
      if (clusterWorkEditing) {
        if (dataIndex.includes('npuType')) {
          clusterWorkSelectRef.current?.focus();
        } else {
          clusterWorkInputRef.current?.focus();
        }
      }
    }, [clusterWorkEditing, dataIndex]);

    const toggleEditClusterWork = () => {
      setClusterWorkEditing(!clusterWorkEditing);
      form.setFieldsValue({
        [dataIndex]: record[dataIndex],
      });
    };

    const save = async () => {
      try {
        const values = await form.validateFields();
        toggleEditClusterWork();
        handleSaveWorkTable({
          ...record,
          ...values,
        });
      } catch (errInfo) {
        console.log('Validate Failed:', errInfo);
      }
    };

    let childNode = children;

    if (editable) {
      if (clusterWorkEditing) {
        let validatorRule;
        if (dataIndex.includes('nodeConfigName')) {
          validatorRule = validateName;
        } else if (dataIndex.includes('npuType')) {
          validatorRule = '';
        } else {
          validatorRule = validateWork;
        }

        let formItemContent;
        if (dataIndex.includes('npuType')) {
          formItemContent = (
            <Select ref={clusterWorkSelectRef} onBlur={save} style={{ width: 120 }} options={npuTypeOption} />
          );
        } else {
          formItemContent = <Input ref={clusterWorkInputRef} onPressEnter={save} onBlur={save} />;
        }

        childNode = (
          <Form.Item style={{ margin: 0 }} name={dataIndex} rules={[{ validator: validatorRule }]}>
            {formItemContent}
          </Form.Item>
        );
      } else {
        childNode = (
          <div
            className="editable-cell-value-wrap defaultClusterWork"
            style={{
              paddingInlineEnd: 24,
            }}
            onClick={toggleEditClusterWork}
          >
            {children}
          </div>
        );
      }
    }

    return <td {...restProps}>{childNode}</td>;
  };

  const validateWork = async (_rule, value) => {
    const regex = /^\d+$/;
    if (value && !regex.test(value)) {
      return Promise.reject(new Error(`${workHeadConfigTh(_rule.field)}类型应为数字！`));
    } else {
      return Promise.resolve();
    }
  };
  const [clusterWorkTableData, setClusterWorkTableData] = useState([
    {
      key: '0',
      nodeType: 'worker',
      nodeConfigName: '请输入',
      cpu: '请输入',
      memory: '请输入',
      maxCopy: '请输入',
      npu: '请输入',
      minCopy: '请输入',
      npuType: '请输入',
    },
  ]);
  const [count, setCount] = useState(1);
  const handleDeleteWorkTable = (key) => {
    const newData = clusterWorkTableData.filter((item) => item.key !== key);
    setClusterWorkTableData(newData);
  };
  const defaultClusterWorklTable = [
    {
      title: '节点类型',
      key: 'clusterWorkType',
      dataIndex: 'nodeType',
      render: (_, record) => record.nodeType || '请输入',
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>节点配置名称
        </p>
      ),
      key: 'clusterWorkName',
      dataIndex: 'nodeConfigName',
      editable: true,
      render: (_, record) => record.nodeConfigName || '请输入',
    },
    {
      title: 'NPU （卡）',
      key: 'clusterWorkNpu',
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
      key: 'clusterWorkCpu',
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
      key: 'clusterWorkMemory',
      dataIndex: 'memory',
      editable: true,
      render: (_, record) => record.memory || '请输入',
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {'最大副本数'}
        </p>
      ),
      key: 'clusterWorkMaxCopy',
      dataIndex: 'maxCopy',
      editable: true,
      render: (_, record) => record.maxCopy || 8,
    },
    {
      title: (
        <p>
          <span style={{ color: 'red', lineHeight: '22px', marginRight: '4px' }}>*</span>
          {'最小副本数'}
        </p>
      ),
      key: 'clusterWorkMinCopy',
      dataIndex: 'minCopy',
      editable: true,
      render: (_, record) => record.minCopy || 0,
    },
    {
      title: 'NPU类型',
      key: 'clusterWorkType',
      dataIndex: 'npuType',
      editable: true,
      render: (_, record) => record.npuType || '请输入',
    },
    {
      title: '操作',
      dataIndex: 'operation',
      render: (_, record) =>
        clusterWorkTableData.length > 1 ? (
          <Popconfirm title="确认删除?" onConfirm={() => handleDeleteWorkTable(record.key)}>
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
      memory: '请输入',
      nodeType: 'worker',
      nodeConfigName: '请输入',
      npu: '请输入',
      cpu: '请输入',
      maxCopy: '请输入',
      minCopy: '请输入',
      npuType: '请输入',
    };
    setClusterWorkTableData([...clusterWorkTableData, newData]);
    setCount(count + 1);
  };
  const handleSaveWorkTable = (row) => {
    const newData = [...clusterWorkTableData];
    const index = newData.findIndex((item) => row.key === item.key);
    const item = newData[index];
    newData.splice(index, 1, {
      ...item,
      ...row,
    });
    setClusterWorkTableData(newData);
  };
  const workComponents = {
    body: {
      row: EditableRow,
      cell: EditableCell,
    },
  };
  const workColumns = defaultClusterWorklTable.map((col) => {
    if (!col.editable) {
      return col;
    }
    return {
      ...col,
      onCell: (record) => ({
        record,
        dataIndex: col.dataIndex,
        title: col.title,
        editable: col.editable,
        handleSaveWorkTable,
      }),
    };
  });

  const onFinish = (data) => {
    console.log(clusterHeadTableData, clusterWorkTableData);
    setIsStart(data);
    if (isWorkCreate && isHeadCreate) {
      submitAdd(data);
    } else {
      messageApi.error('表单格式有问题，请重新填写！');
    }
  };
  // 调用创建接口
  const submitAdd = async (data) => {
    let clustetFormData = addFormDefault;
    clustetFormData.metadata.name = addClusterForm.getFieldsValue().clusterName;
    clustetFormData.spec.rayVersion = addClusterForm.getFieldsValue().rayVersion;
    if (addClusterForm.getFieldsValue().check) {
      if (Object.prototype.hasOwnProperty.call(clustetFormData.spec.headGroupSpec.rayStartParams, 'num-cpus')) {
        delete clustetFormData.spec.headGroupSpec.rayStartParams['num-cpus'];
      }
      if (clusterHeadTableData[0].npu !== '请输入' && clusterHeadTableData[0].npu !== '') {
        clustetFormData.spec.headGroupSpec.rayStartParams[resources] = `{"NPU":${clusterHeadTableData[0].npu}}`;
      }
      if (clusterHeadTableData[0].npuType !== '请输入' && clusterHeadTableData[0].npuType !== '') {
        clustetFormData.spec.headGroupSpec.template.spec.containers[0].schedulerName = 'volcano';
      }
    } else {
      clustetFormData.spec.headGroupSpec.rayStartParams['num-cpus'] = '0';
    }
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].name = clusterHeadTableData[0].nodeConfigName;
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].image = `${addClusterForm.getFieldsValue().clusterImagename}:${addClusterForm.getFieldsValue().clusterImageversion}`;
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].resources.limits.cpu = clusterHeadTableData[0].cpu;
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].resources.limits.memory = `${clusterHeadTableData[0].memory}G`;
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].resources.requests.cpu = clusterHeadTableData[0].cpu;
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].resources.requests.memory = `${clusterHeadTableData[0].memory}G`;
    if (addClusterForm.getFieldsValue().headVolumes) {
      clustetFormData.spec.headGroupSpec.template.spec.volumes = [
        yamlTojson(addClusterForm.getFieldsValue().headVolumes),
      ];
    }
    if (addClusterForm.getFieldsValue().headVolumeMounts) {
      clustetFormData.spec.headGroupSpec.template.spec.containers[0].volumeMounts = [
        yamlTojson(addClusterForm.getFieldsValue().headVolumeMounts),
      ];
    }
    if (clusterHeadTableData[0].npu !== '请输入' && clusterHeadTableData[0].npu !== '') {
      clustetFormData.spec.headGroupSpec.template.spec.containers[0].resources.limits[clusterHeadTableData[0].npuType] =
        clusterHeadTableData[0].npu;
      clustetFormData.spec.headGroupSpec.template.spec.containers[0].resources.requests[
        clusterHeadTableData[0].npuType
      ] = clusterHeadTableData[0].npu;
    }
    clustetFormData.spec.headGroupSpec.template.spec.containers[0].env = addClusterForm.getFieldsValue().headEnv;
    let workGroupSpecs = [];
    clusterWorkTableData.forEach((item) => {
      let defaultWorkSingleData = addFormWorkGroupData;
      defaultWorkSingleData.maxReplicas = Number(item.maxCopy);
      defaultWorkSingleData.minReplicas = Number(item.minCopy);
      defaultWorkSingleData.template.spec.containers[0].name = item.nodeConfigName;
      defaultWorkSingleData.template.spec.containers[0].image = `${addClusterForm.getFieldsValue().clusterImagename}:${addClusterForm.getFieldsValue().clusterImageversion}`;
      defaultWorkSingleData.template.spec.containers[0].resources.limits.cpu = item.cpu;
      defaultWorkSingleData.template.spec.containers[0].resources.limits.memory = `${item.memory}G`;
      defaultWorkSingleData.template.spec.containers[0].resources.requests.cpu = item.cpu;
      defaultWorkSingleData.template.spec.containers[0].resources.requests.memory = `${item.memory}G`;
      if (item.npu !== '请输入' && item.npu !== '') {
        defaultWorkSingleData.template.spec.containers[0].resources.limits[item.npuType] = item.npu;
        defaultWorkSingleData.template.spec.containers[0].resources.requests[item.npuType] = item.npu;
      }
      defaultWorkSingleData.template.spec.containers[0].env = addClusterForm.getFieldsValue().workEnv;

      if (addClusterForm.getFieldsValue().workVolumes) {
        defaultWorkSingleData.template.spec.volumes = [yamlTojson(addClusterForm.getFieldsValue().workVolumes)];
      }
      if (addClusterForm.getFieldsValue().workVolumeMounts) {
        defaultWorkSingleData.template.spec.containers[0].volumeMounts = [
          yamlTojson(addClusterForm.getFieldsValue().workVolumeMounts),
        ];
      }
      workGroupSpecs.push(defaultWorkSingleData);
    });
    clustetFormData.spec.workerGroupSpecs = workGroupSpecs;
    const appFormData = new FormData();
    appFormData.append('config', jsonToYaml(JSON.stringify(clustetFormData)));
    appFormData.append('createBy', creator);
    appFormData.append('isStart', data);
    try {
      const res = await addCluster(appFormData);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(`创建成功！`);
        setTimeout(() => {
          history.push('/RayCluster');
        }, 500);
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
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
  const goBack = () => {
    history.push('/RayCluster');
  };
  useEffect(() => {
    let checkWorkvalue = [];
    clusterWorkTableData.forEach((item) => {
      let value =
        item.nodeConfigName !== '' &&
        item.nodeConfigName !== '请输入' &&
        item.cpu !== '' &&
        item.cpu !== '请输入' &&
        item.maxCopy !== '' &&
        item.maxCopy !== '请输入' &&
        item.minCopy !== '' &&
        item.minCopy !== '请输入' &&
        item.memory !== '' &&
        item.memory !== '请输入' &&
        (item.npu !== '' || item.npu !== '请输入') &&
        (item.npuType !== '' || item.npuType !== '请输入') &&
        (item.npu === '' || item.npu === '请输入') &&
        (item.npuType === '' || item.npuType === '请输入');
      if (value) {
        checkWorkvalue.push('true');
      } else {
        checkWorkvalue.push('false');
      }
    });
    if (checkWorkvalue.includes('false')) {
      setIsWorkCreate(false);
    } else {
      setIsWorkCreate(true);
    }
  }, [clusterWorkTableData]);
  useEffect(() => {
    let checkHeadvalue = [];
    clusterHeadTableData.forEach((item) => {
      let value =
        item.nodeConfigName !== '' &&
        item.nodeConfigName !== '请输入' &&
        item.cpu !== '' &&
        item.cpu !== '请输入' &&
        item.memory !== '' &&
        item.memory !== '请输入' &&
        (item.npu !== '' || item.npu !== '请输入') &&
        (item.npuType !== '' || item.npuType !== '请输入') &&
        (item.npu === '' || item.npu === '请输入') &&
        (item.npuType === '' || item.npuType === '请输入');
      console.log(item.npu !== '' || item.npu !== '请输入', item.npuType !== '' || item.npuType !== '请输入');
      if (value) {
        checkHeadvalue.push('true');
      } else {
        checkHeadvalue.push('flase');
      }
    });
    if (checkHeadvalue.includes('false')) {
      setIsHeadCreate(false);
    } else {
      setIsHeadCreate(true);
    }
  }, [clusterHeadTableData]);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="rayAddForm">
      {contextHolder}
      <div className="rayAddFormContent">
        <Form form={addClusterForm} initialValues={{ remember: true }} autoComplete="off" layout={'vertical'}>
          <div className="rayAddForm_info">
            <p className="rayAddForm_info_title">基本信息</p>
            <div className="rayAddForm_images">
              <Form.Item
                label="集群名称"
                name="clusterName"
                rules={[{ required: true, message: '请输入!' }, { validator: validateName }]}
              >
                <Input showCount maxLength={253} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item
                label="Ray版本"
                name="rayVersion"
                rules={[{ required: true, message: '请输入!' }, { validator: validateImage }]}
              >
                <Input showCount maxLength={128} style={{ width: '100%' }} />
              </Form.Item>
            </div>
          </div>
          <div className="rayAddForm_info">
            <p className="rayAddForm_info_title">算法框架</p>
            <div className="rayAddForm_images">
              <Form.Item
                label="镜像名称"
                name="clusterImagename"
                rules={[{ required: true, message: '请输入镜像名称!' }]}
              >
                <Input placeholder="请先输入" style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item
                label="镜像版本"
                name="clusterImageversion"
                rules={[{ required: true, message: '请输入镜像版本!' }, { validator: validateImage }]}
              >
                <Input showCount maxLength={128} placeholder="请先输入镜像名称" style={{ width: '100%' }} />
              </Form.Item>
            </div>
          </div>
          <div className="rayAddForm_info">
            <p className="rayAddForm_head rayAddForm_head_bold">head节点规格配置</p>
            <p style={{ color: 'red', fontSize: '12px', marginBottom: '8px' }}>
              注：CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'请输入'
            </p>
            <Form.Item label="">
              <Table
                components={headComponents}
                rowClassName={() => 'editable-row'}
                bordered
                dataSource={clusterHeadTableData}
                className="defaultClusterAddForm"
                columns={headColumns}
                pagination={false}
              />
            </Form.Item>
            <Form.Item className="defaultClusterAddForm" label="" name="check" valuePropName="checked">
              <Checkbox>允许头节点处理工作负载</Checkbox>
            </Form.Item>
            <div className="volumesContent defaultClusterAddForm" style={{ marginTop: '32px' }}>
              <p>volumes</p>
              <p>volumeMounts</p>
            </div>
            <div className="rayAddForm_images">
              <Form.Item name="headVolumes defaultClusterAddForm" label="">
                <CodeMirror
                  value={headVolumesYaml}
                  extensions={[javascript()]}
                  className="defaultClusterAddForm"
                  onChange={(value, viewUpdate) => setHeadVolumesYaml(value)}
                />
              </Form.Item>
              <Form.Item name="headVolumeMounts defaultClusterAddForm" label="">
                <CodeMirror
                  value={headVolumeMountsYaml}
                  extensions={[javascript()]}
                  className="defaultClusterAddForm"
                  onChange={(value, viewUpdate) => setHeadVolumeMountsYaml(value)}
                />
              </Form.Item>
            </div>
            <p className="rayAddForm_head_environment_title defaultClusterAddForm">环境变量</p>
            <Form.List name="headEnv">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space
                      key={key}
                      className="defaultClusterAddForm"
                      style={{
                        display: 'flex',
                      }}
                      align="baseline"
                    >
                      <Form.Item
                        {...restField}
                        name={[name, 'name']}
                        className="defaultClusterAddForm"
                        rules={[
                          {
                            required: true,
                            message: '请输入名称',
                          },
                        ]}
                      >
                        <Input style={{ width: '320px' }} placeholder="请输入名称" />
                      </Form.Item>
                      <p className="list_equal">=</p>
                      <Form.Item
                        {...restField}
                        name={[name, 'value']}
                        className="defaultClusterAddForm"
                        rules={[
                          {
                            required: true,
                            message: '请输入值',
                          },
                        ]}
                      >
                        <Input className="defaultClusterAddForm" style={{ width: '320px' }} placeholder="请输入值" />
                      </Form.Item>
                      <DeleteOutlined
                        className="defaultClusterAddForm"
                        style={{ color: '#3f66f5' }}
                        onClick={() => remove(name)}
                      />
                    </Space>
                  ))}
                  <Form.Item>
                    <p className="rayAddForm_head_environment defaultClusterAddForm" onClick={() => add()}>
                      <span style={{ marginRight: '4px' }}>
                        <PlusCircleOutlined />
                      </span>
                      添加环境变量
                    </p>
                  </Form.Item>
                </>
              )}
            </Form.List>
          </div>
          <div className="rayAddForm_info">
            <p className="rayAddForm_head rayAddForm_head_bold">worker节点规格配置</p>
            <p style={{ color: 'red', fontSize: '12px', marginBottom: '8px' }}>
              注：节点配置名称、CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'请输入'
            </p>
            <Table
              components={workComponents}
              rowClassName={() => 'editable-row'}
              bordered
              dataSource={clusterWorkTableData}
              columns={workColumns}
              pagination={true}
            />
            <Button
              className="rayAddForm_work_add_button defaultClusterAddForm"
              onClick={handleAddWorkTable}
              type="dashed"
              icon={<PlusCircleOutlined />}
            >
              添加节点配置
            </Button>
            <div className="volumesContent defaultClusterAddForm" style={{ marginTop: '32px' }}>
              <p>volumes</p>
              <p>volumeMounts</p>
            </div>
            <div className="rayAddForm_images defaultClusterAddForm">
              <Form.Item name="workVolumes" label="">
                <CodeMirror
                  value={workVolumesYaml}
                  extensions={[javascript()]}
                  onChange={(value, viewUpdate) => setWorkVolumesYaml(value)}
                />
              </Form.Item>
              <Form.Item name="workVolumeMounts defaultClusterAddForm" label="">
                <CodeMirror
                  value={workVolumeMountsYaml}
                  extensions={[javascript()]}
                  onChange={(value, viewUpdate) => setWorkVolumeMountsYaml(value)}
                />
              </Form.Item>
            </div>
            <p className="rayAddForm_head_environment_title defaultClusterAddForm">环境变量</p>
            <Form.List name="workEnv">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space
                      key={key}
                      className="defaultClusterAddForm2"
                      style={{
                        display: 'flex',
                      }}
                      align="baseline"
                    >
                      <Form.Item
                        {...restField}
                        className="defaultClusterAddForm2"
                        name={[name, 'name']}
                        rules={[
                          {
                            required: true,
                            message: '请输入名称',
                          },
                        ]}
                      >
                        <Input className="defaultClusterAddForm2" style={{ width: '320px' }} placeholder="请输入名称" />
                      </Form.Item>
                      <p className="list_equal">=</p>
                      <Form.Item
                        {...restField}
                        className="defaultClusterAddForm2"
                        name={[name, 'value']}
                        rules={[
                          {
                            required: true,
                            message: '请输入值',
                          },
                        ]}
                      >
                        <Input className="defaultClusterAddForm2" style={{ width: '320px' }} placeholder="请输入值" />
                      </Form.Item>
                      <DeleteOutlined
                        className="defaultClusterAddForm2"
                        style={{ color: '#3f66f5' }}
                        onClick={() => remove(name)}
                      />
                    </Space>
                  ))}
                  <Form.Item>
                    <p className="rayAddForm_head_environment defaultClusterAddForm2" onClick={() => add()}>
                      <span style={{ marginRight: '4px' }}>
                        <PlusCircleOutlined />
                      </span>
                      添加环境变量
                    </p>
                  </Form.Item>
                </>
              )}
            </Form.List>
          </div>
          <div className="rayAddFormButtons">
            <Button className="cancel_btn" onClick={() => goBack()}>
              取消
            </Button>
            <Button className="primary_btn ml" htmlType="submit" onClick={() => onFinish(false)}>
              创建模板
            </Button>
            <Button className="primary_btn ml" htmlType="submit" onClick={() => onFinish(true)}>
              创建并启动
            </Button>
          </div>
        </Form>
      </div>
    </div>
  );
}
