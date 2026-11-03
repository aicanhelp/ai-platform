/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import {
  SyncOutlined,
  MoreOutlined,
  CloseOutlined,
  QuestionCircleOutlined,
  CheckOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
} from '@ant-design/icons';
import {
  Form,
  Table,
  Space,
  Input,
  Button,
  Popover,
  message,
  Switch,
  Modal,
  Checkbox,
  Row,
  Col,
  Tag,
  Tooltip,
} from 'antd';

import { useEffect, useState, useCallback } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/ColocationWorkloadManagementPage.less';
import { sorterFirstAlphabet, firstAlphabetUp, compareIPs } from '@/tools/utils';
import ToolTipComponent from '@/components/ToolTipComponent';
import { getNodeListData, modifyColocationStatus, modifyNodeConfig } from '@/api/colocationApi';
import { ResponseCode, nodeStatus } from '@/common/constants';

const FeatureTag = ({ label, value, intl }) => {
  const featureLabels = {
    cpuPreemption: intl.formatMessage({ id: 'config.feature.cpuPreemption' }),
    memoryPreemption: intl.formatMessage({ id: 'config.feature.memoryPreemption' }),
    quotaTurbo: intl.formatMessage({ id: 'config.feature.quotaTurbo' }),
    dynMemory: intl.formatMessage({ id: 'config.feature.dynMemory' }),
    dynCache: intl.formatMessage({ id: 'config.feature.dynCache' }),
    psiDetection: intl.formatMessage({ id: 'config.feature.psiDetection' }),
  };
  const featureValueMap = {
    检测中: intl.formatMessage({ id: 'config.feature.detecting' }),
    '内核缺少cpu.qos_level接口': intl.formatMessage({ id: 'config.feature.missing_cpu_qos_level' }),
    '内核缺少memory.qos_level接口': intl.formatMessage({ id: 'config.feature.missing_memory_qos_level' }),
    '内核缺少CPU.CFS控制接口': intl.formatMessage({ id: 'config.feature.missing_cpu_cfs' }),
    '内核缺少memory.high_async_ratio接口': intl.formatMessage({ id: 'config.feature.missing_memory_high_async_ratio' }),
    '硬件不支持RDT/MPAM特性': intl.formatMessage({ id: 'config.feature.unsupported_rdt_mpam' }),
    内核不支持PSI特性: intl.formatMessage({ id: 'config.feature.unsupported_psi' }),
  };
  return (
    <Tag className="feature-support-tag" color={value === 'OK' ? '#09aa71' : '#e7434a'}>
      {featureLabels[label] || label}
      {value === 'OK' ? (
        <CheckCircleFilled />
      ) : (
        <>
          <CloseCircleFilled />
          <span className="falseTips">{featureValueMap[value] || value}</span>
        </>
      )}
    </Tag>
  );
};

export default function ColocationConfigurationPage() {
  const intl = useIntl();
  const [configForm] = Form.useForm();
  const [isShow, setIsShow] = useState('');
  const [filterNodeStatus, setFilterNodeStatus] = useState([]);
  const [configLoading, setConfigLoading] = useState(false);
  const [isParamModalOpen, setIsParamModalOpen] = useState(false);
  // 节点信息
  const [colocationConfig, setColocationConfig] = useState({});
  const [configList, setConfigList] = useState([]);
  const [originalList, setOriginalList] = useState([]);
  // 表格分页
  const [configPage, setConfigPage] = useState(1);
  const [configPageSize, setConfigPageSize] = useState(10);

  const [messageApi, contextHolder] = message.useMessage();

  // 策略配置子表单
  const [volcanoSchedulerForm] = Form.useForm();
  const [quotaTurboForm] = Form.useForm();
  const [dynMemoryForm] = Form.useForm();
  const [dynCacheForm] = Form.useForm();
  const [psiForm] = Form.useForm();
  const [rubikForm] = Form.useForm();
  // 策略配置表单启用状态
  const [quotaTurboDisable, setQuotaTurboDisable] = useState(false);
  const [dynMemoryDisable, setDynmemoryDisable] = useState(false);
  const [dynCacheDisable, setDyncacheDisable] = useState(false);
  const [psiDisable, setPsiDisable] = useState(false);
  const [quotaTurboDisableMsg, setQuotaTurboDisableMsg] = useState('');
  const [dynMemoryDisableMsg, setDynmemoryDisableMsg] = useState('');
  const [dynCacheDisableMsg, setDyncacheDisableMsg] = useState('');
  const [psiDisableMsg, setPsiDisableMsg] = useState('');

  const enabledThreshold = Form.useWatch('enabledThreshold', volcanoSchedulerForm);
  const rubikThreshold = Form.useWatch('enabledThreshold', rubikForm);
  const quotaTurboThreshold = Form.useWatch('enabledThreshold', quotaTurboForm);
  const dynMemoryThreshold = Form.useWatch('enabledThreshold', dynMemoryForm);
  const dynCacheThreshold = Form.useWatch('enabledThreshold', dynCacheForm);
  const psiThreshold = Form.useWatch('enabledThreshold', psiForm);
  const [volcanoCheck, setVolcanoCheck] = useState(false);
  const handleConfigShowChange = () => {
    setIsShow('cant_show');
  };

  const searchNode = (totalData = originalList, isChange = true) => {
    const nodeConfigName = configForm.getFieldValue('nodeConfigName');
    let temporyList = totalData;
    if (nodeConfigName) {
      temporyList = temporyList.filter((item) => item.name.toLowerCase().includes(nodeConfigName.toLowerCase()));
    }
    setConfigList([...temporyList]);
    isChange ? setConfigPage(1) : null;
  };

  const handleRerfesh = async () => {
    getConfigList();
  };

  const handleOpenParamModal = () => {
    setIsParamModalOpen(true);
  };

  // 判断节点混部是否禁用，并生成禁用提示
  const isNodeSwitchDisabled = (record) => {
    const { featureSupport } = record;
    const notFulfillOption =
      (quotaTurboForm.getFieldValue('enabledThreshold') && featureSupport.quotaTurbo !== 'OK') ||
      (dynMemoryForm.getFieldValue('enabledThreshold') && featureSupport.dynMemory !== 'OK') ||
      (dynCacheForm.getFieldValue('enabledThreshold') && featureSupport.dynCache !== 'OK') ||
      (psiForm.getFieldValue('enabledThreshold') && featureSupport.psiDetection !== 'OK');
    if (featureSupport.cpuPreemption !== 'OK' || featureSupport.memoryPreemption !== 'OK') return true;
    if (notFulfillOption) return true;
    return false;
  };
  const getNodeSwitchDisableMsg = (record) => {
    const { featureSupport } = record;
    const featureLabel = (key) => intl.formatMessage({ id: `config.feature.${key}` });
    let arr = [];
    if (featureSupport.cpuPreemption !== 'OK' || featureSupport.memoryPreemption !== 'OK') {
      if (featureSupport.cpuPreemption !== 'OK') {
        arr.push(featureLabel('cpuPreemption'));
      }
      if (featureSupport.memoryPreemption !== 'OK') {
        arr.push(featureLabel('memoryPreemption'));
      }
      return intl.formatMessage({ id: 'config.nodeFeatureNotSupported' }, { features: arr.join(', ') });
    }
    arr = [];
    if (quotaTurboForm.getFieldValue('enabledThreshold') && featureSupport.quotaTurbo !== 'OK') {
      arr.push(featureLabel('quotaTurbo'));
    }
    if (dynMemoryForm.getFieldValue('enabledThreshold') && featureSupport.dynMemory !== 'OK') {
      arr.push(featureLabel('dynMemory'));
    }
    if (psiForm.getFieldValue('enabledThreshold') && featureSupport.psiDetection !== 'OK') {
      arr.push(featureLabel('psiDetection'));
    }
    if (dynCacheForm.getFieldValue('enabledThreshold') && featureSupport.dynCache !== 'OK') {
      arr.push(featureLabel('dynCache'));
    }
    if (arr.length > 0)
      return intl.formatMessage({ id: 'config.nodeAdvancedFeatureNotSupported' }, { features: arr.join(', ') });
    return '';
  };

  const DisabledTipSwitch = ({ disabled, disableMsg, ...props }) => {
    // 判断是否禁用
    if (disabled) {
      return (
        <Tooltip title={disableMsg}>
          <span>
            <Switch disabled {...props} />
          </span>
        </Tooltip>
      );
    }
    return <Switch {...props} />;
  };

  const configColumns = [
    {
      title: intl.formatMessage({ id: 'table.column.nodeName' }),
      key: 'nodeConfigName',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => record.name,
    },
    Table.EXPAND_COLUMN,
    {
      title: intl.formatMessage({ id: 'table.column.colocationCapability' }),
      key: 'operator',
      render: (_, record) => {
        if (record.featureSupport.cpuPreemption === 'OK' && record.featureSupport.memoryPreemption === 'OK') {
          const supportOptionFeatures =
            record.featureSupport.quotaTurbo === 'OK' &&
            record.featureSupport.dynCache === 'OK' &&
            record.featureSupport.dynMemory === 'OK' &&
            record.featureSupport.psiDetection === 'OK';
          if (supportOptionFeatures) {
            return intl.formatMessage({ id: 'config.fullSupport' });
          } else {
            return intl.formatMessage({ id: 'config.partialSupport' });
          }
        } else {
          return intl.formatMessage({ id: 'config.unavailable' });
        }
      },
    },
    {
      title: intl.formatMessage({ id: 'common.status' }),
      key: 'status',
      filters: filterNodeStatus,
      filterMultiple: false,
      onFilter: (value, record) => record.status.toLowerCase() === value.toLowerCase(),
      sorter: (a, b) => sorterFirstAlphabet(a.status, b.status),
      render: (_, record) => (
        <p className={`resource_status ${record.status.toLowerCase()}_circle`}>{firstAlphabetUp(record.status)}</p>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.ipAddress' }),
      key: 'ip',
      sorter: (a, b) => compareIPs(a.addresses, b.addresses),
      render: (_, record) => (
        <>
          {record?.addresses
            ? record.addresses.map((item) => {
                return (
                  <div className="table_tuning_box">
                    <p>{item || '--'}</p>
                  </div>
                );
              })
            : '--'}
        </>
      ),
    },
    {
      title: intl.formatMessage({ id: 'table.column.osVersion' }),
      key: 'os_version',
      sorter: (a, b) => sorterFirstAlphabet(a.osImage, b.osImage),
      render: (_, record) => record.osImage || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.column.enableColocation' }),
      key: 'is_open',
      sorter: (a, b) => sorterFirstAlphabet(a.isColocated, b.isColocated),
      render: (_, record) => (
        <DisabledTipSwitch
          checked={record.isColocated}
          onChange={(checked) => handleControlNode(checked, record.name)}
          disabled={isNodeSwitchDisabled(record)}
          disableMsg={getNodeSwitchDisableMsg(record)}
        />
      ),
    },
  ];

  const handleControlNode = async (checked, name) => {
    try {
      const res = await modifyColocationStatus(name, checked);
      if (res.status === ResponseCode.OK) {
        messageApi.success(
          intl.formatMessage(
            { id: 'config.nodeColocationEnabled' },
            {
              name,
              status: checked
                ? intl.formatMessage({ id: 'config.nodeEnabled' })
                : intl.formatMessage({ id: 'config.nodeDisabled' }),
            },
          ),
        );
        // 更新数据
        getConfigList(false);
      }
    } catch (e) {
      if (e) {
        messageApi.error(
          intl.formatMessage({ id: 'config.modifyNodeStatusFailed' }, { name, error: e.response.data.message }),
        );
      }
    }
  };

  const handleParamOk = async () => {
    try {
      await volcanoSchedulerForm.validateFields();
      await rubikForm.validateFields();
      await quotaTurboForm.validateFields();
      await dynMemoryForm.validateFields();
      await dynCacheForm.validateFields();
      await psiForm.validateFields();
    } catch {
      messageApi.error(intl.formatMessage({ id: 'config.validationFailed' }));
      return;
    }

    const eviction = rubikThreshold
      ? {
          enable: rubikThreshold,
          cpuEvict: {
            threshold: Number(rubikForm.getFieldValue('cpuEvict')) || 0,
          },
          memoryEvict: {
            threshold: Number(rubikForm.getFieldValue('memoryEvict')) || 0,
          },
        }
      : {
          enable: false,
        };
    const quotaTurbo = quotaTurboThreshold
      ? {
          enable: quotaTurboThreshold,
          highWaterMark: Number(quotaTurboForm.getFieldValue('highWaterMark')) || 0,
          alarmWaterMark: Number(quotaTurboForm.getFieldValue('alarmWaterMark')) || 0,
        }
      : {
          enable: false,
        };
    const dynMemory = { enable: dynMemoryThreshold };
    const dynCache = dynCacheThreshold
      ? {
          enable: dynCacheThreshold,
          membandPercent: {
            low: Number(dynCacheForm.getFieldValue('memBandLow')) || 0,
            mid: Number(dynCacheForm.getFieldValue('memBandMid')) || 0,
            high: Number(dynCacheForm.getFieldValue('memBandHigh')) || 0,
          },
          l3Percent: {
            low: Number(dynCacheForm.getFieldValue('l3PercentLow')) || 0,
            mid: Number(dynCacheForm.getFieldValue('l3PercentMid')) || 0,
            high: Number(dynCacheForm.getFieldValue('l3PercentHigh')) || 0,
          },
        }
      : {
          enable: false,
        };
    const psi = psiThreshold
      ? {
          enable: psiThreshold,
          resource: psiForm.getFieldValue('resource'),
          avg10Threshold: Number(psiForm.getFieldValue('avg10Threshold')) || 0,
        }
      : {
          enable: false,
        };
    const volcanoSchedulerConfig = {
      usagePlugin: {
        enable: volcanoCheck,
        usageThreshold:
          volcanoCheck && volcanoSchedulerForm.getFieldValue('enabledThreshold')
            ? {
                enable: volcanoSchedulerForm.getFieldValue('enabledThreshold'),
                cpu: Number(volcanoSchedulerForm.getFieldValue('cpuEvict')),
                memory: Number(volcanoSchedulerForm.getFieldValue('memoryEvict')),
              }
            : {
                enable: false,
              },
      },
    };

    const data = {
      nodeInfo: originalList,
      rubikConfig: { eviction, quotaTurbo, dynMemory, dynCache, psi },
      volcanoSchedulerConfig,
    };

    try {
      const res = await modifyNodeConfig(data);
      if (res.status === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'config.modifySuccess' }));
        setIsParamModalOpen(false);
        getConfigList();
      }
    } catch (e) {
      if (e) {
        messageApi.error(intl.formatMessage({ id: 'config.modifyFailed' }));
      }
    }
  };

  const updateModalForm = (configInfo) => {
    rubikForm.setFieldsValue(
      configInfo.rubikConfig.eviction.enable
        ? {
            enabledThreshold: configInfo.rubikConfig.eviction.enable,
            cpuEvict: String(configInfo.rubikConfig.eviction.cpuevict.threshold ?? '60'),
            memoryEvict: String(configInfo.rubikConfig.eviction.memoryevict.threshold ?? '60'),
          }
        : {
            enabledThreshold: configInfo.rubikConfig.eviction.enable,
            cpuEvict: '60',
            memoryEvict: '60',
          },
    );
    setVolcanoCheck(configInfo.volcanoSchedulerConfig.usagePlugin.enable);
    volcanoSchedulerForm.setFieldsValue(
      configInfo.volcanoSchedulerConfig.usagePlugin.usageThreshold.enable
        ? {
            enabledThreshold: configInfo.volcanoSchedulerConfig.usagePlugin.usageThreshold.enable,
            cpuEvict: String(configInfo.volcanoSchedulerConfig.usagePlugin.usageThreshold.cpu ?? '60.0'),
            memoryEvict: String(configInfo.volcanoSchedulerConfig.usagePlugin.usageThreshold.memory ?? '60.0'),
          }
        : {
            enabledThreshold: configInfo.volcanoSchedulerConfig.usagePlugin.usageThreshold.enable,
            cpuEvict: '60.0',
            memoryEvict: '60.0',
          },
    );
    quotaTurboForm.setFieldsValue(
      configInfo.rubikConfig.quotaTurbo.enable
        ? {
            enabledThreshold: configInfo.rubikConfig.quotaTurbo.enable ?? false,
            highWaterMark: String(configInfo.rubikConfig.quotaTurbo.highWaterMark ?? '60'),
            alarmWaterMark: String(configInfo.rubikConfig.quotaTurbo.alarmWaterMark ?? '60'),
          }
        : {
            enabledThreshold: configInfo.rubikConfig.quotaTurbo.enable ?? false,
            highWaterMark: '60',
            alarmWaterMark: '80',
          },
    );
    dynMemoryForm.setFieldsValue({
      enabledThreshold: configInfo.rubikConfig.dynMemory.enable ?? false,
    });
    dynCacheForm.setFieldsValue(
      configInfo.rubikConfig.dynCache.enable
        ? {
            enabledThreshold: configInfo.rubikConfig.dynCache.enable ?? false,
            l3PercentLow: String(configInfo.rubikConfig.dynCache.l3Percent.low ?? '20'),
            l3PercentMid: String(configInfo.rubikConfig.dynCache.l3Percent.mid ?? '30'),
            l3PercentHigh: String(configInfo.rubikConfig.dynCache.l3Percent.high ?? '50'),
            memBandLow: String(configInfo.rubikConfig.dynCache.memBandPercent.low ?? '20'),
            memBandMid: String(configInfo.rubikConfig.dynCache.memBandPercent.mid ?? '30'),
            memBandHigh: String(configInfo.rubikConfig.dynCache.memBandPercent.high ?? '50'),
          }
        : {
            enabledThreshold: configInfo.rubikConfig.dynCache.enable ?? false,
            l3PercentLow: '20',
            l3PercentMid: '30',
            l3PercentHigh: '50',
            memBandLow: '20',
            memBandMid: '30',
            memBandHigh: '50',
          },
    );
    psiForm.setFieldsValue(
      configInfo.rubikConfig.psi.enable
        ? {
            enabledThreshold: configInfo.rubikConfig.psi.enable ?? false,
            resource: configInfo.rubikConfig.psi.resource,
            avg10Threshold: String(configInfo.rubikConfig.psi.avg10Threshold ?? '5.0'),
          }
        : {
            enabledThreshold: configInfo.rubikConfig.psi.enable ?? false,
            resource: ['cpu', 'memory'],
            avg10Threshold: '5.0',
          },
    );
  };

  // 关闭弹窗时清空未提交的配置
  useEffect(() => {
    if (!isParamModalOpen) {
      updateModalForm(colocationConfig);
    }
  }, [isParamModalOpen]);

  const handleParamCancel = () => {
    setIsParamModalOpen(false);
  };

  const handleConfigTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setConfigPage(pagination.page);
      setConfigPageSize(pagination.pageSize);
    }
    if (extra.action === 'filter') {
      setConfigPage(1);
    }
  };

  const getConfigList = useCallback(async (isChange = true) => {
    setConfigLoading(true);
    getNodeListData()
      .then((res) => {
        if (res.status === ResponseCode.OK) {
          setOriginalList(res.data.data.nodeInfo);
          let { rubikConfig, volcanoSchedulerConfig } = res.data.data;
          let colocationData = { rubikConfig, volcanoSchedulerConfig };
          setColocationConfig(colocationData);
          updateModalForm(colocationData);
          searchNode(res.data.data.nodeInfo, isChange); // 先搜索
        }
      })
      .catch((e) => {
        setOriginalList([]);
        setConfigList([]); // 数组为空
        setColocationConfig({});
        if (e) {
          messageApi.error(intl.formatMessage({ id: 'message.dataFetchFailed' }));
        }
      })
      .finally(() => {
        setConfigLoading(false);
      });
  }, []);

  const validateInteger = (value, min, max) => {
    if (!value) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.inputValue' })));
    }
    if (!/^[1-9]\d?$/.test(value)) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.inputIntegerRange' }, { min, max })));
    }
    let num = parseInt(value, 10);
    if ((min && num < min) || (max && num > max)) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.inputIntegerRange' }, { min, max })));
    }
    return Promise.resolve();
  };

  const validateFloat = (value, min, max, maxDigits) => {
    if (!value) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.inputValue' })));
    }
    if (!/^\d*\.?\d+$/.test(value)) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.inputFloatRange' }, { min, max })));
    }
    if ((min && value < min) || (max && value > max)) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.inputFloatRange' }, { min, max })));
    }
    const [, decimalPart, ...resets] = value.split('.');
    if (decimalPart && decimalPart.length > maxDigits) {
      return Promise.reject(new Error(intl.formatMessage({ id: 'config.maxDecimals' }, { max: maxDigits })));
    }
    return Promise.resolve();
  };

  useEffect(() => {
    getConfigList();
  }, [getConfigList]);

  useEffect(() => {
    const newFilterStatus = [];
    Object.keys(nodeStatus).map((item) => {
      newFilterStatus.push({ text: item, value: item });
    });
    setFilterNodeStatus(newFilterStatus);
  }, []);

  useEffect(() => {
    let quotaTurboFlag = false;
    let dynMemoryFlag = false;
    let dynCacheFlag = false;
    let psiDetectionFlag = false;
    let quotaTurboBadNodes = [];
    let dynMemoryBadNodes = [];
    let dynCacheBadNodes = [];
    let psiDetectionBadNodes = [];
    originalList.forEach((item) => {
      if (item.isColocated) {
        if (item.featureSupport.quotaTurbo !== 'OK') {
          quotaTurboFlag = true;
          quotaTurboBadNodes.push(item.name);
        }
        if (item.featureSupport.dynMemory !== 'OK') {
          dynMemoryFlag = true;
          dynMemoryBadNodes.push(item.name);
        }
        if (item.featureSupport.dynCache !== 'OK') {
          dynCacheFlag = true;
          dynCacheBadNodes.push(item.name);
        }
        if (item.featureSupport.psiDetection !== 'OK') {
          psiDetectionFlag = true;
          psiDetectionBadNodes.push(item.name);
        }
      }
    });
    setQuotaTurboDisable(quotaTurboFlag);
    setDynmemoryDisable(dynMemoryFlag);
    setDyncacheDisable(dynCacheFlag);
    setPsiDisable(psiDetectionFlag);
    if (quotaTurboFlag) {
      setQuotaTurboDisableMsg(
        intl.formatMessage({ id: 'config.colocationNodeNotSupportFeature' }) + quotaTurboBadNodes.join(', '),
      );
    }
    if (dynMemoryFlag) {
      setDynmemoryDisableMsg(
        intl.formatMessage({ id: 'config.colocationNodeNotSupportFeature' }) + dynMemoryBadNodes.join(', '),
      );
    }
    if (dynCacheFlag) {
      setDyncacheDisableMsg(
        intl.formatMessage({ id: 'config.colocationNodeNotSupportFeature' }) + dynCacheBadNodes.join(', '),
      );
    }
    if (psiDetectionFlag) {
      setPsiDisableMsg(
        intl.formatMessage({ id: 'config.colocationNodeNotSupportFeature' }) + psiDetectionBadNodes.join(', '),
      );
    }
  }, [originalList]);

  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: intl.formatMessage({ id: 'colocation.title' }), path: `/ColocationConfiguration` },
          { title: intl.formatMessage({ id: 'config.title' }), path: `/` },
        ]}
      />
      <div className="workload_content hybrid_detail_content_box">
        <div className="workload_detail_content">
          <div className={`${isShow}`} style={{ marginTop: '20px' }}>
            <ToolTipComponent>
              <div
                className="podDetail"
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              >
                <div className="promot-box">{intl.formatMessage({ id: 'message.colocationOnlyOnColocationNode' })}</div>
                <CloseOutlined onClick={handleConfigShowChange} />
              </div>
            </ToolTipComponent>
          </div>
          <div className={`${isShow}workload_detail_content_box`}>
            <div className="workload_detail_form">
              <Form className="workload-searchForm form_padding_bottom" form={configForm}>
                <div className="workload-searchForm-box">
                  <Form.Item name="nodeConfigName" className="workload-search-input" style={{ marginRight: '20px' }}>
                    <Input.Search
                      placeholder={intl.formatMessage({ id: 'search.node' })}
                      onSearch={() => searchNode()}
                      autoComplete="off"
                      maxLength={53}
                    />
                  </Form.Item>
                </div>
                <Form.Item>
                  <Space>
                    <Button className="workload_detail_form_button" onClick={handleOpenParamModal}>
                      {intl.formatMessage({ id: 'config.paramConfigTitle' })}
                    </Button>
                    <Button
                      icon={<SyncOutlined />}
                      onClick={handleRerfesh}
                      className="reset_btn"
                      style={{ marginLeft: '16px' }}
                    ></Button>
                  </Space>
                </Form.Item>
              </Form>
            </div>
            <div className={`workLoad-colocation-table-no-padding workload_detail_content_table`}>
              {contextHolder}
              <Table
                locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
                rowKey="name"
                scroll={{ y: 720 }}
                loading={configLoading}
                columns={configColumns}
                dataSource={configList}
                expandable={{
                  expandedRowRender: (record) => (
                    <div className="feature-support">
                      <p className="feature-support-line">
                        {intl.formatMessage({ id: 'config.requiredFeatures' })}
                        <FeatureTag label="cpuPreemption" value={record.featureSupport.cpuPreemption} intl={intl} />
                        <FeatureTag
                          label="memoryPreemption"
                          value={record.featureSupport.memoryPreemption}
                          intl={intl}
                        />
                      </p>
                      <p className="feature-support-line">
                        {intl.formatMessage({ id: 'config.advancedFeatures' })}
                        <FeatureTag label="quotaTurbo" value={record.featureSupport.quotaTurbo} intl={intl} />
                        <FeatureTag label="dynMemory" value={record.featureSupport.dynMemory} intl={intl} />
                        <FeatureTag label="psiDetection" value={record.featureSupport.psiDetection} intl={intl} />
                        <FeatureTag label="dynCache" value={record.featureSupport.dynCache} intl={intl} />
                      </p>
                    </div>
                  ),
                }}
                pagination={{
                  className: 'colocation-website-page',
                  current: configPage,
                  pageSize: configPageSize,
                  showTotal: (total) => intl.formatMessage({ id: 'pagination.total' }, { total }),
                  showSizeChanger: true,
                  showQuickJumper: true,
                  pageSizeOptions: [10, 20, 50],
                }}
                onChange={handleConfigTableChange}
              ></Table>
            </div>
          </div>
        </div>
      </div>
      {/* 混部配置策略表单 */}
      <Modal
        className="modal_flex_colocation modal_set_config"
        width="1120px"
        height="720px"
        open={isParamModalOpen}
        title={
          <p>
            {intl.formatMessage({ id: 'config.paramConfigTitle' })}{' '}
            <Popover
              color="#333333"
              placement="top"
              content={
                <p style={{ color: '#fff', margin: 0 }}>{intl.formatMessage({ id: 'config.paramConfigTip' })}</p>
              }
              trigger="click"
            >
              <QuestionCircleOutlined />
            </Popover>
          </p>
        }
        destroyOnHidden={true}
        getContainer={() => document.getElementsByClassName('hybrid_detail_content_box')[0]}
        transitionName=""
        maskTransitionName=""
        onCancel={handleParamCancel}
        footer={[
          <div className="modal_flex_colocation_bottom" style={{ justifyContent: 'flex-end' }}>
            <Button className="back_cancel_btn" onClick={handleParamCancel}>
              {intl.formatMessage({ id: 'common.cancel' })}
            </Button>
            <Button className={'back_config_btn'} onClick={handleParamOk}>
              {intl.formatMessage({ id: 'common.confirm' })}
            </Button>
          </div>,
        ]}
      >
        <div className="modal_config_box">
          <div className="deployment_scheduling_box">
            <div className="deployment_header">
              <span className="title">{intl.formatMessage({ id: 'config.volcano.title' })}</span>
              <Popover
                color="#333333"
                placement="top"
                content={<p style={{ color: '#fff', margin: 0 }}>{intl.formatMessage({ id: 'config.volcano.tip' })}</p>}
                trigger="click"
              >
                <QuestionCircleOutlined />
              </Popover>
              <Switch value={volcanoCheck} onChange={(checked) => setVolcanoCheck(checked)} />
            </div>
            {volcanoCheck && (
              <div className="deployment_content">
                <Form form={volcanoSchedulerForm} autoComplete="off">
                  <>
                    <Form.Item
                      colon={false}
                      label={intl.formatMessage({ id: 'config.volcano.thresholdSwitch' })}
                      name="enabledThreshold"
                      className="special_font_size"
                      style={{ display: 'block' }}
                    >
                      <Switch />
                    </Form.Item>
                    <Form.Item
                      noStyle
                      shouldUpdate={(prevValues, currentValues) =>
                        prevValues.enabledThreshold !== currentValues.enabledThreshold
                      }
                    >
                      {({ getFieldValue }) =>
                        getFieldValue('enabledThreshold') && (
                          <>
                            <Form.Item
                              rules={[
                                { required: true, message: '' },
                                { validator: (_, value) => validateFloat(value, 0, 100, 2) },
                              ]}
                              colon={false}
                              label={intl.formatMessage({ id: 'config.volcano.cpuThreshold' })}
                              name="cpuEvict"
                              extra={intl.formatMessage({ id: 'config.volcano.cpuThresholdTip' })}
                              initialValue="60"
                            >
                              <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                            </Form.Item>
                            <Form.Item
                              rules={[
                                { required: true, message: '' },
                                { validator: (_, value) => validateFloat(value, 0, 100, 2) },
                              ]}
                              colon={false}
                              label={intl.formatMessage({ id: 'config.volcano.memoryThreshold' })}
                              name="memoryEvict"
                              extra={intl.formatMessage({ id: 'config.volcano.memoryThresholdTip' })}
                              initialValue="60"
                            >
                              <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                            </Form.Item>
                          </>
                        )
                      }
                    </Form.Item>
                  </>
                </Form>
              </div>
            )}
          </div>

          <div className="deployment_quotaTurbo" style={{ marginTop: '32px' }}>
            <Form form={quotaTurboForm} autoComplete="off">
              <div className="deployment_header" style={{ flexDirection: 'row' }}>
                <span className="title">{intl.formatMessage({ id: 'config.quotaTurbo.title' })}</span>
                <Popover
                  color="#333333"
                  placement="top"
                  content={
                    <p style={{ background: '#333333', color: '#fff', margin: 0 }}>
                      {intl.formatMessage({ id: 'config.quotaTurbo.tip' })}
                    </p>
                  }
                  trigger="click"
                >
                  <QuestionCircleOutlined />
                </Popover>
                <Form.Item
                  colon={false}
                  label=""
                  name="enabledThreshold"
                  className="special_font_size"
                  style={{ marginBottom: 0 }}
                >
                  <DisabledTipSwitch disabled={quotaTurboDisable} disableMsg={quotaTurboDisableMsg} />
                </Form.Item>
              </div>
              <Form.Item
                noStyle
                shouldUpdate={(prevValues, currentValues) =>
                  prevValues.enabledThreshold !== currentValues.enabledThreshold
                }
              >
                {({ getFieldValue }) =>
                  getFieldValue('enabledThreshold') && (
                    <div className="deployment_content">
                      <Form.Item
                        rules={[
                          { required: true, message: '' },
                          {
                            validator: (_, value) => validateInteger(value, 0, 100),
                          },
                        ]}
                        colon={false}
                        label={intl.formatMessage({ id: 'config.quotaTurbo.highWaterMark' })}
                        name="highWaterMark"
                        extra={intl.formatMessage({ id: 'config.quotaTurbo.highWaterMarkTip' })}
                        initialValue="60"
                      >
                        <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                      </Form.Item>
                      <Form.Item
                        rules={[
                          { required: true, message: '' },
                          {
                            validator: (_, value) => validateInteger(value, 0, 100),
                          },
                        ]}
                        colon={false}
                        label={intl.formatMessage({ id: 'config.quotaTurbo.alarmWaterMark' })}
                        name="alarmWaterMark"
                        extra={intl.formatMessage({ id: 'config.quotaTurbo.alarmWaterMarkTip' })}
                        initialValue="80"
                        style={{ whiteSpace: 'pre-wrap' }}
                      >
                        <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                      </Form.Item>
                    </div>
                  )
                }
              </Form.Item>
            </Form>
          </div>
          <div className="deployment_dynmemory" style={{ marginTop: '32px' }}>
            <Form form={dynMemoryForm} autoComplete="off">
              <div className="deployment_header" style={{ flexDirection: 'row' }}>
                <span className="title">{intl.formatMessage({ id: 'config.dynMemory.title' })}</span>
                <Popover
                  color="#333333"
                  placement="top"
                  content={
                    <p style={{ background: '#333333', color: '#fff', margin: 0 }}>
                      {intl.formatMessage({ id: 'config.dynMemory.tip' })}
                    </p>
                  }
                  trigger="click"
                >
                  <QuestionCircleOutlined />
                </Popover>
                <Form.Item
                  colon={false}
                  label=""
                  name="enabledThreshold"
                  className="special_font_size"
                  style={{ marginBottom: 0 }}
                >
                  <DisabledTipSwitch disabled={dynMemoryDisable} disableMsg={dynMemoryDisableMsg} />
                </Form.Item>
              </div>
            </Form>
          </div>
          <div className="deployment_dyncache" style={{ marginTop: '32px' }}>
            <Form form={dynCacheForm} autoComplete="off">
              <div className="deployment_header" style={{ flexDirection: 'row' }}>
                <span className="title">{intl.formatMessage({ id: 'config.dynCache.title' })}</span>
                <Popover
                  color="#333333"
                  placement="top"
                  content={
                    <p style={{ background: '#333333', color: '#fff', margin: 0 }}>
                      {intl.formatMessage({ id: 'config.dynCache.tip' })}
                    </p>
                  }
                  trigger="click"
                >
                  <QuestionCircleOutlined />
                </Popover>
                <Form.Item
                  colon={false}
                  label=""
                  name="enabledThreshold"
                  className="special_font_size"
                  style={{ marginBottom: 0 }}
                >
                  <DisabledTipSwitch disabled={dynCacheDisable} disableMsg={dynCacheDisableMsg} />
                </Form.Item>
              </div>
              <Form.Item
                noStyle
                shouldUpdate={(prevValues, currentValues) =>
                  prevValues.enabledThreshold !== currentValues.enabledThreshold
                }
              >
                {({ getFieldValue }) =>
                  getFieldValue('enabledThreshold') && (
                    <div className="deployment_content">
                      <p className="dyncache-label">
                        {intl.formatMessage({ id: 'config.dynCache.memoryBandWatermark' })}
                      </p>
                      <Form.Item colon={false} extra={intl.formatMessage({ id: 'config.dynCache.memoryBandTip' })}>
                        <Form.Item
                          rules={[
                            { required: true, message: '' },
                            {
                              validator: (_, value) => validateInteger(value, 0, 100),
                            },
                          ]}
                          colon={false}
                          label={intl.formatMessage({ id: 'config.dynCache.low' })}
                          name="memBandLow"
                          initialValue={20}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                        </Form.Item>
                        <Form.Item
                          rules={[
                            { required: true, message: '' },
                            {
                              validator: (_, value) => validateInteger(value, 0, 100),
                            },
                          ]}
                          colon={false}
                          label={intl.formatMessage({ id: 'config.dynCache.mid' })}
                          name="memBandMid"
                          initialValue={30}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                        </Form.Item>
                        <Form.Item
                          rules={[
                            { required: true, message: '' },
                            {
                              validator: (_, value) => validateInteger(value, 0, 100),
                            },
                          ]}
                          colon={false}
                          label={intl.formatMessage({ id: 'config.dynCache.high' })}
                          name="memBandHigh"
                          initialValue={50}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                        </Form.Item>
                      </Form.Item>
                      <p className="dyncache-label">{intl.formatMessage({ id: 'config.dynCache.l3CacheWatermark' })}</p>
                      <Form.Item colon={false} extra={intl.formatMessage({ id: 'config.dynCache.l3CacheTip' })}>
                        <Form.Item
                          rules={[
                            { required: true, message: '' },
                            {
                              validator: (_, value) => validateInteger(value, 0, 100),
                            },
                          ]}
                          colon={false}
                          label={intl.formatMessage({ id: 'config.dynCache.low' })}
                          name="l3PercentLow"
                          initialValue={20}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                        </Form.Item>
                        <Form.Item
                          rules={[
                            { required: true, message: '' },
                            {
                              validator: (_, value) => validateInteger(value, 0, 100),
                            },
                          ]}
                          colon={false}
                          label={intl.formatMessage({ id: 'config.dynCache.mid' })}
                          name="l3PercentMid"
                          initialValue={30}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                        </Form.Item>
                        <Form.Item
                          rules={[
                            { required: true, message: '' },
                            {
                              validator: (_, value) => validateInteger(value, 0, 100),
                            },
                          ]}
                          colon={false}
                          label={intl.formatMessage({ id: 'config.dynCache.high' })}
                          name="l3PercentHigh"
                          initialValue={50}
                        >
                          <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                        </Form.Item>
                      </Form.Item>
                    </div>
                  )
                }
              </Form.Item>
            </Form>
          </div>
          <div className="deployment_psi" style={{ marginTop: '32px' }}>
            <Form form={psiForm} autoComplete="off">
              <div className="deployment_header" style={{ flexDirection: 'row' }}>
                <span className="title">{intl.formatMessage({ id: 'config.psi.title' })}</span>
                <Popover
                  color="#333333"
                  placement="top"
                  content={
                    <p style={{ background: '#333333', color: '#fff', margin: 0 }}>
                      {intl.formatMessage({ id: 'config.psi.tip' })}
                    </p>
                  }
                  trigger="click"
                >
                  <QuestionCircleOutlined />
                </Popover>
                <Form.Item
                  colon={false}
                  label=""
                  name="enabledThreshold"
                  className="special_font_size"
                  style={{ marginBottom: 0 }}
                >
                  <DisabledTipSwitch disabled={psiDisable} disableMsg={psiDisableMsg} />
                </Form.Item>
              </div>
              <Form.Item
                noStyle
                shouldUpdate={(prevValues, currentValues) =>
                  prevValues.enabledThreshold !== currentValues.enabledThreshold
                }
              >
                {({ getFieldValue }) =>
                  getFieldValue('enabledThreshold') && (
                    <div className="deployment_content">
                      <Form.Item
                        colon={false}
                        label={intl.formatMessage({ id: 'config.psi.resourceLabel' })}
                        name="resource"
                      >
                        <Checkbox.Group defaultValue={['cpu', 'memory']}>
                          <Row>
                            <Col>
                              <Checkbox value="cpu" style={{ lineHeight: '32px' }}>
                                CPU
                              </Checkbox>
                            </Col>
                            <Col>
                              <Checkbox value="memory" style={{ lineHeight: '32px' }}>
                                {intl.formatMessage({ id: 'config.psi.memory' })}
                              </Checkbox>
                            </Col>
                            <Col>
                              <Checkbox value="io" style={{ lineHeight: '32px' }}>
                                IO
                              </Checkbox>
                            </Col>
                          </Row>
                        </Checkbox.Group>
                      </Form.Item>
                      <br />
                      <Form.Item
                        rules={[
                          { required: true, message: '' },
                          {
                            validator: (_, value) => validateFloat(value, 5, 100, 2),
                          },
                        ]}
                        colon={false}
                        label={intl.formatMessage({ id: 'config.psi.threshold' })}
                        name="avg10Threshold"
                        initialValue={5.0}
                        extra={intl.formatMessage({ id: 'config.psi.thresholdTip' })}
                      >
                        <Input placeholder={intl.formatMessage({ id: 'placeholder.input5to100' })} />
                      </Form.Item>
                    </div>
                  )
                }
              </Form.Item>
            </Form>
          </div>
          <div className="deployment_water_level_expulsion" style={{ marginTop: '32px' }}>
            <Form form={rubikForm} autoComplete="off">
              <div className="deployment_header" style={{ flexDirection: 'row' }}>
                <span className="title">{intl.formatMessage({ id: 'config.rubik.title' })}</span>
                <Form.Item
                  colon={false}
                  label=""
                  name="enabledThreshold"
                  className="special_font_size"
                  style={{ marginLeft: '16px', marginBottom: 0 }}
                >
                  <Switch />
                </Form.Item>
              </div>
              <Form.Item
                noStyle
                shouldUpdate={(prevValues, currentValues) =>
                  prevValues.enabledThreshold !== currentValues.enabledThreshold
                }
              >
                {({ getFieldValue }) =>
                  getFieldValue('enabledThreshold') && (
                    <div className="deployment_content">
                      <Form.Item
                        rules={[
                          { required: true, message: '' },
                          {
                            validator: (_, value) => validateInteger(value, 0, 100),
                          },
                        ]}
                        colon={false}
                        label={intl.formatMessage({ id: 'config.rubik.cpuEvictionThreshold' })}
                        name="cpuEvict"
                        extra={intl.formatMessage({ id: 'config.rubik.cpuEvictionTip' })}
                        initialValue="60"
                      >
                        <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                      </Form.Item>
                      <Form.Item
                        rules={[
                          { required: true, message: '' },
                          {
                            validator: (_, value) => validateInteger(value, 0, 100),
                          },
                        ]}
                        colon={false}
                        label={intl.formatMessage({ id: 'config.rubik.memoryEvictionThreshold' })}
                        name="memoryEvict"
                        extra={intl.formatMessage({ id: 'config.rubik.memoryEvictionTip' })}
                        initialValue="60"
                      >
                        <Input placeholder={intl.formatMessage({ id: 'placeholder.input0to100' })} />
                      </Form.Item>
                    </div>
                  )
                }
              </Form.Item>
            </Form>
          </div>
        </div>
      </Modal>
    </div>
  );
}
