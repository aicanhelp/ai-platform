/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Radio, Form, Select, DatePicker, Button, message, Tooltip } from 'antd';
import ToastMsg from '@/components/ToastMsg';
import axios from 'axios';
import { useCallback, useEffect, useRef, useState, useStore, useMemo } from 'openinula';
import { SyncOutlined } from '@ant-design/icons';
import Dayjs from 'dayjs';
import {
  timePeriodOptionsKeys,
  refreshTimeOptionsKeys,
  typeOptions,
  ResponseCode,
  stepList,
} from '@/common/constants';
import ClusterMonitor from '@/pages/container/monitor/monitorDashboard/resourceSelect/ClusterMonitor';
import NodeMonitor from '@/pages/container/monitor/monitorDashboard/resourceSelect/NodeMonitor';
import WorkloadMonitor from '@/pages/container/monitor/monitorDashboard/resourceSelect/WorkloadMonitor';
import PodMonitor from '@/pages/container/monitor/monitorDashboard/resourceSelect/PodMonitor';
import ContainerMonitor from '@/pages/container/monitor/monitorDashboard/resourceSelect/ContainerMonitor';
import {
  getResourceControllMonitor,
  getNodeList,
  getNamespaceList,
  getWorkloadList,
} from '@/api/monitorApi';
import { getServiceTimeZoneInterface } from '@/api/containerApi';
import { solveMonitorComponentData, debounce, timelayDataSolve } from '@/tools/utils';
import infoStore from '@/store/infoStore';
import DiffTimeInfo from '@/components/DiffTimeInfo';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

let lastPerformanceTime = window.performance.now();
let lastDate = new Date().getTime();
let checkResourceTimeChangeTimer = null;
let resourceTimer = null;

export default function ResourceMonitor({ propsType, propsColor }) {
  const { t, i18n } = useTranslation('monitoring');
  const isFirstLoadedRef = useRef(true); // 首次加载
  let abortController = null;
  const [messageApi, contextHolder] = message.useMessage();
  const [selectedKey, setSelectedKeys] = useState('cluster');
  const infoStoreExample = infoStore();
  const [timeForm] = Form.useForm();
  const [clusterForm] = Form.useForm();
  const clusterValue = Form.useWatch('cluster', clusterForm);
  const refreshTimeValue = Form.useWatch('refreshTime', timeForm);
  const endTimeValue = Form.useWatch('endTime', timeForm);
  const [intervalId, setInterValId] = useState(0);
  const [clusterOptions, setClusterOptions] = useState([]);
  const [instanceOptions, setInstanceOptions] = useState([]);
  const [namespaceOptions, setNamespaceOptions] = useState([]);
  const [workloadOptions, setWorkloadOptions] = useState([]);
  const [podOptions, setPodOptions] = useState([]);
  const [containerOptions, setContainerOptions] = useState([]);
  const [podContainerTotalData, setPodContainerTotalData] = useState([]);
  const [detailed, setDetailed] = useState(false);
  const [refreshLoading, setRefreshLoading] = useButtonLoading();

  const [clusterData, setClusterData] = useState({});
  const [nodeData, setNodeData] = useState({});
  const [workloadData, setWorkloadData] = useState({});
  const [podData, setPodData] = useState({});
  const [containerData, setContainerData] = useState({});
  const themeStore = useStore('theme');
  const [diffTime, setDiffTime] = useState(0);
  const [timelyData, setTimelyData] = useState();

  const handleResetMonitor = async () => {
    setRefreshLoading(true);
    try {
      await renderRequest(selectedKey, clusterValue, false);
    } finally {
      setRefreshLoading(false);
    }
  };

  const getResourceList = useCallback(async () => {
    const res = await getNodeList();
    if (res.status === ResponseCode.OK) {
      let options = [];
      res.data.items.map(item => {
        if (!options.filter(opItem => opItem.value === item.metadata.name).length) {
          options.push({ label: item.metadata.name, value: item.metadata.name });
        }
      });
      setInstanceOptions([...options]);
      options.length
        ? clusterForm.setFieldValue('instance', [options[0].value])
        : clusterForm.setFieldValue('instance', undefined);
    }
  }, []);

  const getNamespaceMonitorList = useCallback(async () => {
    const res = await getNamespaceList();
    if (res.status === ResponseCode.OK) {
      let options = [];
      res.data.items.map(item => {
        if (!options.filter(opItem => opItem.value === item.metadata.name).length) {
          options.push({ label: item.metadata.name, value: item.metadata.name });
        }
      });
      setNamespaceOptions([...options]);
      options.length
        ? clusterForm.setFieldValue('namespace', options[0].value)
        : clusterForm.setFieldValue('namespace', undefined);
    }
  }, []);

  const temporyHashMap = {
    cluster: data => setClusterData(data),
    node: data => setNodeData(data),
    workload: data => setWorkloadData(data),
    pod: data => setPodData(data),
    container: data => setContainerData(data),
  };

  const getResourceMonitor = useCallback(async (key, cluster) => {
    if (isFirstLoadedRef.current) {
      setDetailed(false);
    }
    const formValues = timeForm.getFieldsValue();
    let { instance, workload, pod, container, namespace, type } = clusterForm.getFieldsValue();
    let resourceMonitorStart = Dayjs(formValues.endTime).subtract(parseInt(formValues.period), formValues.period.slice(-1));
    let resourceMonitorEnd = Dayjs(formValues.endTime);
    const offset = infoStoreExample.$s.offsetCompareLocal;
    if (offset !== 0) {
      if (offset > 0) {
        resourceMonitorStart = resourceMonitorStart.subtract(offset, 'minute');
        resourceMonitorEnd = resourceMonitorEnd.subtract(offset, 'minute');
      } else {
        resourceMonitorStart = resourceMonitorStart.add(Math.abs(offset), 'minute');
        resourceMonitorEnd = resourceMonitorEnd.add(Math.abs(offset), 'minute');
      }
    }
    resourceMonitorStart = resourceMonitorStart.format('YYYY-MM-DD HH:mm:ss');
    resourceMonitorEnd = resourceMonitorEnd.format('YYYY-MM-DD HH:mm:ss');
    instance ? instance = instance.join('|') : '';
    workload ? workload = workload.join('|') : '';
    container ? container = container.join('|') : '';
    pod && key === 'pod' ? pod = pod.join('|') : '';
    let conditions = {
      step: stepList[formValues.period],
      instance,
      namespace,
      type,
      workload,
      pod,
      container,
    };
    if (key !== 'workload') {
      getTimelyData(cluster, key, { ...conditions, step: '' }).catch(e => messageApi.error(t('messages.fetchFailed', { error: e?.response?.data ?? e?.message ?? '' })));
    }
    let data = {};
    try {
      if (abortController) {
        abortController.abort();
      }
      abortController = new AbortController();
      const { signal } = abortController;
      const res = await getResourceControllMonitor({ cluster, value: key, start: resourceMonitorStart, end: resourceMonitorEnd, conditions }, signal);
      if (res.status === ResponseCode.OK) {
        data = solveMonitorComponentData(key, res.data.results);
      }
    } catch (e) {
      if (axios.isCancel(e)) {
        // 主动取消的请求，忽略
      } else if (e?.response?.status !== ResponseCode.OK) {
        messageApi.error(t('messages.fetchFailed', { error: e?.response?.data ?? e?.message ?? '' }));
      }
    }
    temporyHashMap[key](data);
    setDetailed(true);
    isFirstLoadedRef.current = false;
  }, []);

  const getWorkloadMonitorList = useCallback(async (namespace = clusterForm.getFieldsValue().namespace, type = clusterForm.getFieldsValue().type) => {
    const res = await getWorkloadList(namespace, type);
    if (res.status === ResponseCode.OK) {
      let options = [];
      res.data.items.map(item => {
        if (!options.filter(opItem => opItem.value === item.metadata.name).length) {
          options.push({ label: item.metadata.name, value: item.metadata.name });
        }
      });
      setWorkloadOptions([...options]);
      clusterForm.setFieldValue('workload', options.length ? [options[0].value] : undefined);
    }
  }, []);

  const getPodMonitorList = useCallback(async (namespace = clusterForm.getFieldsValue().namespace, key = 'pod') => {
    const res = await getWorkloadList(namespace, 'pod');
    if (res.status === ResponseCode.OK) {
      let containerList = [];
      let options = [];
      res.data.items.map(item => {
        if (!options.filter(opItem => opItem.value === item.metadata.name).length) {
          options.push({ label: item.metadata.name, value: item.metadata.name });
          containerList.push(item.spec.containers);
        }
      });
      setPodContainerTotalData(res.data.items);
      setPodOptions([...options]);
      if (key === 'pod') {
        clusterForm.setFieldValue('pod', options.length ? [options[0].value] : undefined);
      }
      if (key === 'container') {
        clusterForm.setFieldValue('pod', options.length ? options[0].value : undefined);
        let needContainerList = [];
        const [needContainerArr] = containerList;
        if (needContainerArr) {
          needContainerArr.map(containerItem => {
            needContainerList.push({ label: containerItem.name, value: containerItem.name });
          });
        }
        clusterForm.setFieldValue('container', needContainerList.length ? [needContainerList[0].value] : undefined);
        setContainerOptions([...needContainerList]);
      }
    }
  }, []);

  const getContainerList = (pod) => {
    let needContainerList = [];
    podContainerTotalData.map(item => {
      if (item.metadata.name === pod) {
        item.spec.containers.map(containerItem => {
          needContainerList.push({ label: containerItem.name, value: containerItem.name });
        });
      }
    });
    clusterForm.setFieldValue('container', needContainerList.length ? [needContainerList[0].value] : undefined);
    setContainerOptions([...needContainerList]);
  };

  const renderRequest = useCallback(async (key, value, isRunInstance = true) => {
    if (isFirstLoadedRef.current) {
      setDetailed(false);
    }
    if (isRunInstance) {
      if (key === 'node') {
        await getResourceList();
      }
      if (key === 'workload') {
        await getNamespaceMonitorList();
        await getWorkloadMonitorList();
      }
      if (key === 'pod') {
        await getNamespaceMonitorList();
        await getPodMonitorList();
      }
      if (key === 'container') {
        await getNamespaceMonitorList();
        await getPodMonitorList(clusterForm.getFieldsValue().namespace, 'container');
      }
    }
    getTimeServiceResource();
    await getResourceMonitor(key, value);
  }, []);

  const getTimelyData = async (cluster, value, conditions) => {
    const res = await getResourceControllMonitor({ cluster, value, conditions });
    if (res.status === ResponseCode.OK) {
      setTimelyData(timelayDataSolve(value, res.data.results));
    }
  };

  useEffect(() => {
    if (!isNaN(infoStoreExample.$s.offsetCompareLocal)) {
      if (clusterValue) {
        renderRequest(selectedKey, clusterValue);
      }
    }
  }, [clusterValue, selectedKey, infoStoreExample.$s.offsetCompareLocal]);

  const checkTime = () => {
    let currentPerformanceTime = window.performance.now();
    let currentDate = new Date().getTime();
    let performanceDiff = currentPerformanceTime - lastPerformanceTime;
    let dateDiff = currentDate - lastDate;
    if (Math.abs(performanceDiff - dateDiff) > 1000) {
      getTimeServiceResource();
      timeForm.setFieldValue('endTime', Dayjs(currentDate));
      renderRequest(selectedKey, clusterValue);
    }
    lastPerformanceTime = currentPerformanceTime;
    lastDate = currentDate;
    if (checkResourceTimeChangeTimer) {
      clearTimeout(checkResourceTimeChangeTimer);
    }
    checkResourceTimeChangeTimer = setTimeout(checkTime, 1000);
  };

  useEffect(() => {
    checkResourceTimeChangeTimer = setTimeout(checkTime, 1000);
    return () => clearTimeout(checkResourceTimeChangeTimer);
  }, [selectedKey, clusterValue]);

  useEffect(() => {
    if (propsType === 'component') {
      clearInterval(intervalId);
    }
    return () => clearInterval(intervalId);
  }, [propsType, intervalId]);

  const filterOption = (input, option) =>
    (option?.label ?? '').toLowerCase().includes(input.toLowerCase());

  const setupAutoRefresh = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setInterValId(0);
    }
    const refreshSec = Number(refreshTimeValue);
    clearInterval(resourceTimer);
    if (!Number.isFinite(refreshSec) || refreshSec <= 0) {
      return;
    }
    const delayMs = refreshSec * 1000;
    resourceTimer = setInterval(() => {
      getTimeServiceResource();
    }, delayMs);
    const id = setInterval(() => {
      timeForm.setFieldValue('endTime', Dayjs());
      getResourceMonitor(selectedKey, clusterValue);
    }, delayMs);
    setInterValId(id);
  }, [refreshTimeValue, selectedKey, clusterValue]);

  useEffect(() => {
    setupAutoRefresh();
    return () => clearInterval(resourceTimer);
  }, [setupAutoRefresh]);

  const getClusterList = useCallback(async () => {
    let options = [
      {
        label: 'main',
        value: 'main',
      },
    ];
    setClusterOptions([...options]);
    if (clusterForm) {
      clusterForm.setFieldsValue({
        cluster: options[0].value,
      });
    }
  }, []);

  const handleSelectedMonitorKeys = (e) => {
    setSelectedKeys(e.target.value);
  };

  const handleValueChange = (changeValues) => {
    if (Object.prototype.hasOwnProperty.call(changeValues, 'endTime')) {
      timeForm.setFieldValue('refreshTime', '0');
    }
    if (!Object.prototype.hasOwnProperty.call(changeValues, 'refreshTime')) {
      debounce(() => renderRequest(selectedKey, clusterValue, false), 3000)();
    }
  };

  const timeCallbackFn = (key, value, isLoad = false, endTime = Dayjs()) => {
    timeForm.setFieldValue('endTime', endTime);
    renderRequest(key, value, isLoad, true);
  };

  const handleClusterChange = async (changeValues) => {
    if (Object.prototype.hasOwnProperty.call(changeValues, 'type')) {
      await getWorkloadMonitorList(clusterForm.getFieldsValue().namespace, changeValues.type);
    }
    if (Object.prototype.hasOwnProperty.call(changeValues, 'namespace')) {
      if (selectedKey === 'workload') {
        await getWorkloadMonitorList(clusterForm.getFieldsValue().namespace, clusterForm.getFieldsValue().type);
      }
      if (selectedKey === 'pod') {
        await getPodMonitorList(clusterForm.getFieldsValue().namespace, 'pod');
      }
      if (selectedKey === 'container') {
        await getPodMonitorList(clusterForm.getFieldsValue().namespace, 'container');
      }
    }
    if (selectedKey === 'container') {
      if (Object.prototype.hasOwnProperty.call(changeValues, 'pod')) {
        getContainerList(changeValues.pod);
      }
    }
    getResourceMonitor(selectedKey, clusterForm.getFieldValue('cluster'));
  };

  useEffect(() => {
    getClusterList();
  }, [getClusterList]);

  const getTimeServiceResource = useCallback(async () => {
    const res = await getServiceTimeZoneInterface();
    if (res.status === ResponseCode.OK) {
      if (res.data.data?.currentServerTime) {
        setDiffTime(Math.abs(Math.floor(new Date().getTime() / 1000) - (Number(res.data.data.currentServerTime))));
      }
    }
  }, []);

  const timePeriodOptions = useMemo(() => {
    return timePeriodOptionsKeys.map(item => ({
      label: t(`timePeriod.${item}`, { ns: 'common' }),
      value: item,
    }));
  }, [t, i18n.language]);

  const refreshTimeOptions = useMemo(() => {
    return refreshTimeOptionsKeys.map(item => ({
      label: t(`refreshTime.${item}`, { ns: 'common' }),
      value: item,
      disabled: !(item === '0' || Dayjs(endTimeValue).isSame(Dayjs(), 's')),
    }));
  }, [t, i18n.language, endTimeValue]);

  useEffect(() => {
    timeForm.setFieldValue('endTime', Dayjs());
  }, [t, i18n.language]);

  return <div className="monitor_namespace">
    <ToastMsg contextHolder={contextHolder} />
    {diffTime >= 10 && <DiffTimeInfo time={diffTime} className={'monitor_diff_time'} />}
    <div className='monitor_content_header'>
      <Radio.Group
        value={selectedKey}
        onChange={handleSelectedMonitorKeys}
      >
        <Radio.Button value="cluster">{t('resource.cluster')}</Radio.Button>
        <Radio.Button value="node">{t('resource.node')}</Radio.Button>
        <Radio.Button value="workload">{t('resource.workload')}</Radio.Button>
        <Radio.Button value="pod">{t('resource.pod')}</Radio.Button>
        <Radio.Button value="container">{t('resource.container')}</Radio.Button>
      </Radio.Group>
      <Form form={timeForm} layout='inline' onValuesChange={handleValueChange} initialValues={{
        refreshTime: refreshTimeOptionsKeys[2],
        period: timePeriodOptionsKeys[0],
        endTime: Dayjs(),
      }}>
        <Form.Item name="period" label={t('form.timeRange')} className='ResourceMonitor'>
          <Select options={timePeriodOptions} style={{ minWidth: '100px' }} />
        </Form.Item>
        <Form.Item name="endTime" label={t('form.endTime')} className='ResourceMonitor'>
          <DatePicker showTime />
        </Form.Item>
        <Tooltip title={t('form.refreshTooltip')}>
          <Form.Item name="refreshTime" label={t('form.refreshInterval')} className='ResourceMonitor'>
            <Select className='refresh_time_select' options={refreshTimeOptions} />
          </Form.Item>
        </Tooltip>
        <Form.Item className='ResourceMonitor'>
          <Button icon={<SyncOutlined />} onClick={handleResetMonitor} loading={refreshLoading} className="reset_btn"></Button>
        </Form.Item>
      </Form>
    </div>
    <div className='monitor_search' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <Form form={clusterForm} layout='inline' onValuesChange={handleClusterChange}>
        <Form.Item label={t('form.cluster')} name="cluster">
          <Select options={clusterOptions} style={{ width: 300 }} disabled />
        </Form.Item>
        {selectedKey === 'node' && <Form.Item label={t('component.instance')} name="instance">
          <Select
            mode="multiple"
            maxTagCount={2}
            showSearch
            filterOption={filterOption}
            options={instanceOptions}
            style={{ minWidth: 300 }}
          />
        </Form.Item>}
        {(selectedKey === 'workload' || selectedKey === 'pod' || selectedKey === 'container') &&
          <Form.Item label={t('form.namespace')} name="namespace">
            <Select showSearch filterOption={filterOption} options={namespaceOptions} style={{ width: 300 }} />
          </Form.Item>}
        {(selectedKey === 'workload') &&
          <Form.Item label={t('form.type')} name="type" initialValue={typeOptions[0].value}>
            <Select options={typeOptions} style={{ width: 300 }} />
          </Form.Item>}
        {(selectedKey === 'workload') &&
          <Form.Item label={t('form.workload')} name="workload">
            <Select
              mode="multiple"
              maxTagCount={2}
              showSearch
              filterOption={filterOption}
              options={workloadOptions}
              style={{ minWidth: 300 }}
            />
          </Form.Item>}
        {(selectedKey === 'pod' || selectedKey === 'container') &&
          <Form.Item label={t('form.pod')} name="pod">
            <Select
              mode={selectedKey === 'pod' ? 'multiple' : '-'}
              maxTagCount={2}
              showSearch
              filterOption={filterOption}
              options={podOptions}
              style={{ minWidth: 300 }}
            />
          </Form.Item>}
        {(selectedKey === 'container') &&
          <Form.Item label={t('form.container')} name="container">
            <Select
              showSearch
              mode="multiple"
              maxTagCount={2}
              filterOption={filterOption}
              options={containerOptions}
              style={{ width: 300 }}
            />
          </Form.Item>}
      </Form>
    </div>
    {
      selectedKey === 'cluster' && <ClusterMonitor
        data={clusterData}
        timelyData={timelyData}
        loaded={isFirstLoadedRef.current ? detailed : true}
        propsColor={propsColor}
        refreshFn={() => timeCallbackFn('cluster', clusterValue)}
      />
    }
    {
      selectedKey === 'node' && <NodeMonitor
        data={nodeData}
        timelyData={timelyData}
        loaded={isFirstLoadedRef.current ? detailed : true}
        propsColor={propsColor}
        refreshFn={() => timeCallbackFn('node', clusterValue)}
      />
    }
    {
      selectedKey === 'workload' &&
      <WorkloadMonitor
        data={workloadData}
        loaded={isFirstLoadedRef.current ? detailed : true}
        propsColor={propsColor}
        refreshFn={() => timeCallbackFn('workload', clusterValue)}
      />
    }
    {
      selectedKey === 'pod' &&
      <PodMonitor
        data={podData}
        timelyData={timelyData}
        loaded={isFirstLoadedRef.current ? detailed : true}
        propsColor={propsColor}
        refreshFn={() => timeCallbackFn('pod', clusterValue)}
      />
    }
    {
      selectedKey === 'container' &&
      <ContainerMonitor
        data={containerData}
        timelyData={timelyData}
        loaded={isFirstLoadedRef.current ? detailed : true}
        propsColor={propsColor}
        refreshFn={() => timeCallbackFn('container', clusterValue)}
      />
    }
  </div >;
}
