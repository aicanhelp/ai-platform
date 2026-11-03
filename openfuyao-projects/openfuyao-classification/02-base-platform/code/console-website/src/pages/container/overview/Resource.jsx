/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { useState, useEffect, useMemo, useRef, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import * as echarts from 'echarts';
import { StockOutlined } from '@ant-design/icons';
import '@/styles/pages/overview.less';
import { ResponseCode } from '@/common/constants';
import OverviewEcharts from '@/components/OverviewEcharts';
import overviewCpu from '@/assets/images/overviewCpu.png';
import overviewMemory from '@/assets/images/overviewMemory.png';
import overviewDisk from '@/assets/images/overviewDisk.png';
import { Select, Progress, Form, message } from 'antd';
import { getClusterData, getClusterCpuMemoryDisk } from '@/api/monitorApi';
import { filterOverViewAreaEcharts } from '@/utils/common';
import infoStore from '@/store/infoStore';
import { timePeriodOptionsKeys } from '@/common/constants';
import Dayjs from 'dayjs';
import { transformUnifiedUnit } from '@/tools/utils';

const initialResourceData = {
  cpuUsage: 0, cpuTotal: 0, cpuUtilisation: 0, cpuRequestsTotal: 0, cpuRequestsUtilisation: 0, cpuRequestsUsage: 0, cpuLimitsTotal: 0, cpuLimitsUtilisation: 0, cpuLimitUsage: 0,
  memoryUsage: 0, memoryTotal: 0, memoryUtilisation: 0, memoryRequestsTotal: 0, memoryRequestsUtilisation: 0, memoryRequestsUsage: 0, memoryLimitsTotal: 0, memoryLimitsUtilisation: 0, memoryLimitsUsage: 0,
  diskUsage: 0, diskTotal: 0, diskUtilisation: 0, diskInodeUsage: 0, diskInodeTotal: 0, diskInodeUtilisation: 0,
};

export default function Resource() {
  const { t, i18n } = useTranslation(['overview']);
  const intervalStore = useStore('interval');
  const [resourceForm] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();
  const [resourceData, setResourceData] = useState(initialResourceData);
  const [resourceDatas, setResourceDatas] = useState([]);
  const setTimeoutIdRef = useRef(0);
  const [cpuData, setCpuData] = useState([]); // cpu data
  const [cpuOption, setCpuOption] = useState(null);
  const [cpuDataName, setCpuDataName] = useState('');
  const [isCpuDataChose, setIsCpuDataChose] = useState('node_cpu_utilisation');
  const [isCpuLegend, setIsCpuLegend] = useState(false);
  const [memoryData, setMemoryData] = useState([]); // memory data
  const [memoryOption, setMemoryOption] = useState(null);
  const [memoryDataName, setMemoryDataName] = useState('');
  const [isMemoryDataChose, setisMemoryDataChose] = useState('node_memory_utilisation');
  const [isMemoryLegend, setIsMemoryLegend] = useState(false);
  const [storageData, setStorageData] = useState([]); // storage data
  const [storageOption, setStorageOption] = useState(null);
  const [storageDataName, setStorageDataName] = useState('');
  const [isStorageDataChose, setIsStorageDataChose] = useState('node_disk_size_utilisation');
  const [isDiskLegend, setIsDiskLegend] = useState(false);
  const infoStoreExample = infoStore();
  const resourceColor = ['#43CBB8', '#4B8BEA', '#F9A975', '#B4A2FF', '#EB96F5'];
  const [beginTime, setBeginTime] = useState(Dayjs().subtract(10, 'm').format('YYYY-MM-DD HH:mm:ss'));
  const [endTime, setEndTime] = useState(Dayjs().format('YYYY-MM-DD HH:mm:ss'));
  const [step, setStep] = useState('1m');
  const themeStore = useStore('theme');
  // Change CPU left selection
  const changeCpuUsed = (value) => {
    let chose = '';
    let _name = '';
    switch (value) {
      case 'node_cpu_utilisation': {
        chose = 'node_cpu_utilisation';
        _name = `${t('common:common.node')}CPU${t('resource.usageRate')}`;
        break;
      }
      case 'node_cpu_requests_utilisation': {
        chose = 'node_cpu_requests_utilisation';
        _name = `${t('common:common.node')}CPU${t('resource.requestRate')}`;
        break;
      }
      case 'node_cpu_limits_utilisation': {
        chose = 'node_cpu_limits_utilisation';
        _name = `${t('common:common.node')}CPU${t('resource.limitRate')}`;
        break;
      }
      default: '';
    }
    setIsCpuDataChose(chose);
    setCpuDataName(_name);
  };

  // Change memory left selection
  const changeMemoryUsed = (value) => {
    let chose = '';
    let _name = '';
    switch (value) {
      case 'node_memory_utilisation': {
        chose = 'node_memory_utilisation';
        _name = `${t('common:common.node')}${t('resource.memory')}${t('resource.usageRate')}`;
        break;
      }
      case 'node_memory_requests_utilisation': {
        chose = 'node_memory_requests_utilisation';
        _name = `${t('common:common.node')}${t('resource.memory')}${t('resource.requestRate')}`;
        break;
      }
      case 'node_memory_limits_utilisation': {
        chose = 'node_memory_limits_utilisation';
        _name = `${t('common:common.node')}${t('resource.memory')}${t('resource.limitRate')}`;
        break;
      }
      default: '';
    }
    setisMemoryDataChose(chose);
    setMemoryDataName(_name);
  };

  // Change storage left selection
  const changeStorageUsed = (value) => {
    let chose = '';
    let _name = '';
    switch (value) {
      case 'node_disk_size_utilisation': {
        chose = 'node_disk_size_utilisation';
        _name = t('resource.diskUsage');
        break;
      }
      case 'node_disk_inode_utilisation': {
        chose = 'node_disk_inode_utilisation';
        _name = t('resource.inodeUsage');
        break;
      }
      default: '';
    }
    setIsStorageDataChose(chose);
    setStorageDataName(_name);
  };
  // Fetch resource data
  const getClusterResourceData = async () => {
    const res = await getClusterData();
    if (res.status === ResponseCode.OK) {
      setResourceDatas(res.data.results);
    }
  };
  const clusterResourceData = () => {
    let _resourceArr = { ...initialResourceData };
    resourceDatas.map(item => {
      const cpuUsage = 'cluster_cpu_usage';
      const cpuTotal = 'cluster_cpu_total';
      const cpuRequestsTotal = 'cluster_cpu_requests_total';
      const cpuRequestsUtilisation = 'cluster_cpu_requests_utilisation';
      const cpuLimitsTotal = 'cluster_cpu_limits_total';
      const cpuLimitsUtilisation = 'cluster_cpu_limits_utilisation';
      const memoryUsage = 'cluster_memory_usage';
      const memoryTotal = 'cluster_memory_total';
      const memoryRequestsTotal = 'cluster_memory_requests_total';
      const memoryRequestsUtilisation = 'cluster_memory_requests_utilisation';
      const memoryLimitsTotal = 'cluster_memory_limits_total';
      const memoryLimitsUtilisation = 'cluster_memory_limits_utilisation';
      const diskUsage = 'cluster_disk_size_usage';
      const diskTotal = 'cluster_disk_size_total';
      const diskUtilisation = 'cluster_disk_size_utilisation';
      const diskInodeUsage = 'cluster_disk_inode_usage';
      const diskInodeTotal = 'cluster_disk_inode_total';
      const diskInodeUtilisation = 'cluster_disk_inode_utilisation';
      if (item.data.result.length <= 0) {
        return;
      }
      switch (item.metricName) {
        case cpuUsage:
          _resourceArr.cpuUsage = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case cpuTotal:
          _resourceArr.cpuTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case cpuRequestsTotal:
          _resourceArr.cpuRequestsTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case cpuRequestsUtilisation:
          _resourceArr.cpuRequestsUtilisation = Number(item.data.result[0].sample.value.toFixed(3));
          break;
        case cpuLimitsTotal:
          _resourceArr.cpuLimitsTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case cpuLimitsUtilisation:
          _resourceArr.cpuLimitsUtilisation = Number(item.data.result[0].sample.value.toFixed(3));
          break;
        case memoryUsage:
          _resourceArr.memoryUsage = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case memoryTotal:
          _resourceArr.memoryTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case memoryRequestsTotal:
          _resourceArr.memoryRequestsTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case memoryRequestsUtilisation:
          _resourceArr.memoryRequestsUtilisation = Number(item.data.result[0].sample.value.toFixed(3));
          break;
        case memoryLimitsTotal:
          _resourceArr.memoryLimitsTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case memoryLimitsUtilisation:
          _resourceArr.memoryLimitsUtilisation = Number(item.data.result[0].sample.value.toFixed(3));
          break;
        case diskUsage:
          _resourceArr.diskUsage = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case diskTotal:
          _resourceArr.diskTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case diskUtilisation:
          _resourceArr.diskUtilisation = Number(item.data.result[0].sample.value.toFixed(3));
          break;
        case diskInodeUsage:
          _resourceArr.diskInodeUsage = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case diskInodeTotal:
          _resourceArr.diskInodeTotal = Number(item.data.result[0].sample.value.toFixed(2));
          break;
        case diskInodeUtilisation:
          _resourceArr.diskInodeUtilisation = Number(item.data.result[0].sample.value.toFixed(3));
          break;
        default: '';
      }
    });
    _resourceArr.memoryUtilisation = _resourceArr.memoryTotal ? Number((_resourceArr.memoryUsage / _resourceArr.memoryTotal).toFixed(3)) : 0;
    _resourceArr.cpuUtilisation = _resourceArr.cpuTotal ? Number((_resourceArr.cpuUsage / _resourceArr.cpuTotal).toFixed(3)) : 0;
    _resourceArr.cpuRequestsUsage = _resourceArr.cpuRequestsTotal ? Number((_resourceArr.cpuRequestsTotal * _resourceArr.cpuRequestsUtilisation).toFixed(2)) : 0;
    _resourceArr.cpuLimitUsage = _resourceArr.cpuLimitsTotal ? Number((_resourceArr.cpuLimitsTotal * _resourceArr.cpuLimitsUtilisation).toFixed(2)) : 0;
    _resourceArr.memoryRequestsUsage = _resourceArr.memoryRequestsTotal ? Number((_resourceArr.memoryRequestsTotal * _resourceArr.memoryRequestsUtilisation).toFixed(2)) : 0;
    _resourceArr.memoryLimitsUsage = _resourceArr.memoryLimitsTotal ? Number((_resourceArr.memoryLimitsTotal * _resourceArr.memoryLimitsUtilisation).toFixed(2)) : 0;
    setResourceData(_resourceArr);
    return;
  };
  const getOverviewEchartsData = async () => {
    let dataType = {
      cpu: isCpuDataChose,
      memory: isMemoryDataChose,
      disk: isStorageDataChose,
    };
    let overviewStart = Dayjs(beginTime);
    let overviewEnd = Dayjs(endTime);
    // Handle timezone issue
    const offset = infoStoreExample.$s.offsetCompareLocal;
    if (offset !== 0) {
      if (offset > 0) {
        overviewStart = overviewStart.subtract(offset, 'minute');
        overviewEnd = overviewEnd.subtract(offset, 'minute');
      } else {
        overviewStart = overviewStart.add(Math.abs(offset), 'minute');
        overviewEnd = overviewEnd.add(Math.abs(offset), 'minute');
      }
    }
    overviewStart = overviewStart.format('YYYY-MM-DD HH:mm:ss');
    overviewEnd = overviewEnd.format('YYYY-MM-DD HH:mm:ss');
    try {
      const res = await getClusterCpuMemoryDisk(dataType, overviewStart, overviewEnd, step);
      if (res?.status === ResponseCode.OK) {
        let _cpuData = [];
        let _memoryData = [];
        let _diskData = [];
        res.data.results.map(item => {
          let cpuSeries = [];
          if (item.metricName === isCpuDataChose) {
            if (item.data.result.length > 0) {
              let _dataList = item.data.result;
              _dataList.map(subitem => {
                let _num = 0;
                subitem.series.map(subitems => {
                  _num += subitems.value;
                });
                subitem.value = _num;
              });
              _dataList = _dataList.sort((a, b) => b.value - a.value).slice(0, 5);
              let _cpuTypeList = [];
              _dataList.map(subitem => {
                _cpuTypeList.push(subitem.labels.instance);
                subitem.series.map(subitems => {
                  _cpuData.push({
                    type: subitem.labels.instance,
                    timestamp: subitems.timestamp,
                    value: subitems.value === -1 ? NaN : Math.floor(subitems.value * 100),
                  });
                });
              });
              const _cpuFilteredList = filterOverViewAreaEcharts(_cpuTypeList, _cpuData);
              if (_cpuTypeList.length >= 3) {
                for (let i = 0; i < _cpuFilteredList.length; i++) {
                  cpuSeries.push(
                    {
                      name: _cpuFilteredList[i].name,
                      type: 'line',
                      color: resourceColor[i],
                      emphasis: {
                        focus: 'series',
                      },
                      data: _cpuFilteredList[i].data,
                    },
                  );
                }
              } else {
                for (let i = 0; i < _cpuFilteredList.length; i++) {
                  cpuSeries.push(
                    {
                      name: _cpuFilteredList[i].name,
                      type: 'line',
                      color: resourceColor[i],
                      areaStyle: {
                        color: new echarts.graphic.LinearGradient(
                          0, 0, 0, 1,
                          [
                            { offset: 1, color: `${resourceColor[i]}00` }, // 0% position color
                            { offset: 0, color: resourceColor[i] }, // 100% position color
                          ],
                        ),
                      },
                      emphasis: {
                        focus: 'series',
                      },
                      data: _cpuFilteredList[i].data,
                    },
                  );
                }
              }
              let cpuOptions = {
                tooltip: {
                  trigger: 'axis',
                  axisPointer: {
                    type: 'cross',
                    label: {
                      backgroundColor: '#6a7985',
                    },
                  },
                },
                grid: {
                  left: '3%',
                  top: '15%',
                  right: '5%',
                  bottom: '5%',
                  containLabel: true,
                },
                xAxis: [
                  {
                    type: 'category',
                    boundaryGap: false,
                    data: _cpuFilteredList[0].time,
                  },
                ],
                yAxis: [
                  {
                    type: 'value',
                  },
                ],
                series: cpuSeries,
              };
              setCpuData(_cpuTypeList);
              setCpuOption(cpuOptions);
            }
          }
        });
        res.data.results.map(item => {
          let memorySeries = [];
          if (item.metricName === isMemoryDataChose) {
            if (item.data.result.length > 0) {
              let _dataList = item.data.result;
              _dataList.map(subitem => {
                let _num = 0;
                subitem.series.map(subitems => {
                  _num += subitems.value;
                });
                subitem.value = _num;
              });
              _dataList = _dataList.sort((a, b) => b.value - a.value).slice(0, 5);
              let _memoryTypeList = [];
              _dataList.map(subitem => {
                _memoryTypeList.push(subitem.labels.instance);
                subitem.series.map(subitems => {
                  _memoryData.push({
                    type: subitem.labels.instance,
                    timestamp: subitems.timestamp,
                    value: subitems.value === -1 ? NaN : Math.floor(subitems.value * 100),
                  });
                });
              });
              const _memoryFilteredList = filterOverViewAreaEcharts(_memoryTypeList, _memoryData);
              if (_memoryTypeList.length >= 3) {
                for (let i = 0; i < _memoryFilteredList.length; i++) {
                  memorySeries.push(
                    {
                      name: _memoryFilteredList[i].name,
                      type: 'line',
                      color: resourceColor[i],
                      emphasis: {
                        focus: 'series',
                      },
                      data: _memoryFilteredList[i].data,
                    },
                  );
                }
              } else {
                for (let i = 0; i < _memoryFilteredList.length; i++) {
                  memorySeries.push(
                    {
                      name: _memoryFilteredList[i].name,
                      type: 'line',
                      color: resourceColor[i],
                      areaStyle: {
                        color: new echarts.graphic.LinearGradient(
                          0, 0, 0, 1,
                          [
                            { offset: 1, color: `${resourceColor[i]}00` }, // 0% position color
                            { offset: 0, color: resourceColor[i] }, // 100% position color
                          ],
                        ),
                      },
                      emphasis: {
                        focus: 'series',
                      },
                      data: _memoryFilteredList[i].data,
                    },
                  );
                }
              }
              let memoryOptions = {
                tooltip: {
                  trigger: 'axis',
                  axisPointer: {
                    type: 'cross',
                    label: {
                      backgroundColor: '#6a7985',
                    },
                  },
                },
                grid: {
                  left: '3%',
                  top: '15%',
                  right: '5%',
                  bottom: '5%',
                  containLabel: true,
                },
                xAxis: [
                  {
                    type: 'category',
                    boundaryGap: false,
                    data: _memoryFilteredList[0].time,
                  },
                ],
                yAxis: [
                  {
                    type: 'value',
                  },
                ],
                series: memorySeries,
              };
              setMemoryData(_memoryTypeList);
              setMemoryOption(memoryOptions);
            }
          }
        });
        res.data.results.map(item => {
          let diskSeries = [];
          if (item.metricName === isStorageDataChose) {
            if (item.data.result.length > 0) {
              let _dataList = item.data.result;
              _dataList.map(subitem => {
                let _num = 0;
                subitem.series.map(subitems => {
                  _num += subitems.value;
                });
                subitem.value = _num;
              });
              _dataList = _dataList.sort((a, b) => b.value - a.value).slice(0, 5);
              let _diskTypeList = [];
              _dataList.map(subitem => {
                _diskTypeList.push(subitem.labels.instance);
                subitem.series.map(subitems => {
                  _diskData.push({
                    type: subitem.labels.instance,
                    timestamp: subitems.timestamp,
                    value: subitems.value === -1 ? NaN : Math.floor(subitems.value * 100),
                  });
                });
              });
              const _diskFilteredList = filterOverViewAreaEcharts(_diskTypeList, _diskData);
              if (_diskTypeList.length >= 3) {
                for (let i = 0; i < _diskFilteredList.length; i++) {
                  diskSeries.push(
                    {
                      name: _diskFilteredList[i].name,
                      type: 'line',
                      color: resourceColor[i],
                      emphasis: {
                        focus: 'series',
                      },
                      data: _diskFilteredList[i].data,
                    },
                  );
                }
              } else {
                for (let i = 0; i < _diskFilteredList.length; i++) {
                  diskSeries.push(
                    {
                      name: _diskFilteredList[i].name,
                      type: 'line',
                      color: resourceColor[i],
                      areaStyle: {
                        color: new echarts.graphic.LinearGradient(
                          0, 0, 0, 1,
                          [
                            { offset: 1, color: `${resourceColor[i]}00` }, // 0% position color
                            { offset: 0, color: resourceColor[i] }, // 100% position color
                          ],
                        ),
                      },
                      emphasis: {
                        focus: 'series',
                      },
                      data: _diskFilteredList[i].data,
                    },
                  );
                }
              }
              let storageOptions = {
                tooltip: {
                  trigger: 'axis',
                  axisPointer: {
                    type: 'cross',
                    label: {
                      backgroundColor: '#6a7985',
                    },
                  },
                },
                grid: {
                  left: '3%',
                  top: '15%',
                  right: '5%',
                  bottom: '5%',
                  containLabel: true,
                },
                xAxis: [
                  {
                    type: 'category',
                    boundaryGap: false,
                    data: _diskFilteredList[0].time,
                  },
                ],
                yAxis: [
                  {
                    type: 'value',
                  },
                ],
                series: diskSeries,
              };
              setStorageData(_diskTypeList);
              setStorageOption(storageOptions);
            }
          }
        });
      }
    } catch (e) {
      messageApi.error(`${t('error.dataFetchFailed', { ns: 'common' })}${e?.response?.data ?? e?.message ?? ''}`);
    }
  };
  const selectTime = (e) => {
    let _beginTime = '';
    let _step = '';
    if (e === '10m') {
      _beginTime = Dayjs().subtract(10, 'm').format('YYYY-MM-DD HH:mm:ss');
      _step = '1m';
    }
    if (e === '30m') {
      _beginTime = Dayjs().subtract(30, 'm').format('YYYY-MM-DD HH:mm:ss');
      _step = '6m';
    }
    if (e === '1h') {
      _beginTime = Dayjs().subtract(1, 'h').format('YYYY-MM-DD HH:mm:ss');
      _step = '12m';
    }
    if (e === '3h') {
      _beginTime = Dayjs().subtract(3, 'h').format('YYYY-MM-DD HH:mm:ss');
      _step = '36m';
    }
    if (e === '6h') {
      _beginTime = Dayjs().subtract(6, 'h').format('YYYY-MM-DD HH:mm:ss');
      _step = '72m';
    }
    if (e === '1d') {
      _beginTime = Dayjs().subtract(1, 'd').format('YYYY-MM-DD HH:mm:ss');
      _step = '288m';
    }
    if (e === '3d') {
      _beginTime = Dayjs().subtract(3, 'd').format('YYYY-MM-DD HH:mm:ss');
      _step = '864m';
    }
    if (e === '7d') {
      _beginTime = Dayjs().subtract(7, 'd').format('YYYY-MM-DD HH:mm:ss');
      _step = '2016m';
    }
    if (e === '14d') {
      _beginTime = Dayjs().subtract(14, 'd').format('YYYY-MM-DD HH:mm:ss');
      _step = '4032m';
    }
    if (!_beginTime || !_step) {
      return;
    }
    setBeginTime(_beginTime);
    setEndTime(Dayjs().format('YYYY-MM-DD HH:mm:ss'));
    setStep(_step);
  };
  const timeSearch = () => {
    getClusterResourceData();
    getOverviewEchartsData();
    const id = setTimeout(() => {
      timeSearch();
    }, 30000);
    setTimeoutIdRef.current = id;
  };
  useEffect(() => {
    if (!isNaN(infoStoreExample.$s.offsetCompareLocal) && intervalStore.$s.loginStatus) {
      timeSearch();
    }
    return () => clearTimeout(setTimeoutIdRef.current);
  }, [isCpuDataChose, isMemoryDataChose, isStorageDataChose, step, beginTime, endTime, infoStoreExample.$s.offsetCompareLocal, intervalStore.$s.loginStatus]);
  useEffect(() => {
    clusterResourceData();
  }, [resourceDatas]);

  const timePeriodOptions = useMemo(() => {
    return timePeriodOptionsKeys.map(item => ({
      label: t(`timePeriod.${item}`, { ns: 'common' }),
      value: item,
    }));
  }, [t, i18n.language]);

  return <div className="overview_flex overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
    {contextHolder}
    <div className="overview_flex_resource_box">
      <div className="overview_flex_resource_title">
        <p className="overview_card_title" style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.title')}</p>
        <Form form={resourceForm} layout='inline' initialValues={{ resource: '10m' }}>
          <Form.Item name="resource">
            <Select options={timePeriodOptions} className="overview_card_select" onChange={selectTime} />
          </Form.Item>
        </Form>
      </div>
      <div className="overview_flex_resource">
        <div className="overview_flex_resource_left">
          <div className="overview_flex_resource_left_title">
            <img src={overviewCpu} className="overview_flex_resource_left_cpuimg" alt="" />
            <p style={{ fontWeight: 'bold', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7', margin: '0 8px' }}>{t('resource.cpu')}</p>
            <p style={{ fontWeight: 'bold', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.total')}</p>
            <p style={{ fontWeight: 'bold', color: '#13c2c2', margin: '0 2px 0 8px' }}>{!transformUnifiedUnit(resourceData.cpuTotal) ? resourceData.cpuTotal : '--'}</p>
            <p style={{ marginLeft: '2px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.core')}</p>
          </div>
          <div className={`overview_flex_resource_left_progress ${isCpuDataChose === 'node_cpu_utilisation' ? 'overview_card_selected_cpu_color' : ''}`} onClick={() => changeCpuUsed('node_cpu_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.usageRate')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{!transformUnifiedUnit(resourceData.cpuUtilisation) ? (resourceData.cpuUtilisation * 100).toFixed(1) : '--'}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{!transformUnifiedUnit(resourceData.cpuUsage) ? resourceData.cpuUsage : '--'}Core<span>&nbsp;/&nbsp;</span></p><p>{!transformUnifiedUnit(resourceData.cpuTotal) ? resourceData.cpuTotal : '--'}Core</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={!transformUnifiedUnit(resourceData.cpuUtilisation) ? Number((resourceData.cpuUtilisation * 100).toFixed(1)) : 0} showInfo={false} strokeColor={'#13c2c2'} trailColor={'#dcdcdc'} />
            </div>
          </div>
          <div className={`overview_flex_resource_left_progress ${isCpuDataChose === 'node_cpu_requests_utilisation' ? 'overview_card_selected_cpu_color' : ''}`} onClick={() => changeCpuUsed('node_cpu_requests_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.requestRate')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{!transformUnifiedUnit(resourceData.cpuRequestsUtilisation) ? (resourceData.cpuRequestsUtilisation * 100).toFixed(1) : '--'}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{!transformUnifiedUnit(resourceData.cpuRequestsUsage) ? resourceData.cpuRequestsUsage : '--'}Core<span>&nbsp;/&nbsp;</span></p><p>{!transformUnifiedUnit(resourceData.cpuRequestsTotal) ? resourceData.cpuRequestsTotal : '--'}Core</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={!transformUnifiedUnit(resourceData.cpuRequestsUtilisation) ? Number((resourceData.cpuRequestsUtilisation * 100).toFixed(1)) : 0} showInfo={false} strokeColor={'#13c2c2'} trailColor={'#dcdcdc'} />
            </div>
          </div>
          <div className={`overview_flex_resource_left_progress ${isCpuDataChose === 'node_cpu_limits_utilisation' ? 'overview_card_selected_cpu_color' : ''}`} onClick={() => changeCpuUsed('node_cpu_limits_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.limitRate')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{!transformUnifiedUnit(resourceData.cpuLimitsUtilisation) ? (resourceData.cpuLimitsUtilisation * 100).toFixed(1) : '--'}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{!transformUnifiedUnit(resourceData.cpuLimitUsage) ? resourceData.cpuLimitUsage : '--'}Core<span>&nbsp;/&nbsp;</span></p><p>{!transformUnifiedUnit(resourceData.cpuLimitsTotal) ? resourceData.cpuLimitsTotal : '--'}Core</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={!transformUnifiedUnit(resourceData.cpuLimitsUtilisation) ? Number((resourceData.cpuLimitsUtilisation * 100).toFixed(1)) : '--'} showInfo={false} strokeColor={'#13c2c2'} trailColor={'#dcdcdc'} />
            </div>
          </div>
        </div>
        <div className="overview_flex_resource_right">
          <div className="overview_flex_resource_right_top">
            <div className="overview_flex_resource_right_legendList" style={{ display: isCpuLegend ? 'block' : 'none' }}>
              {cpuData.map((item, index) => {
                return <div className='overview_flex_resource_right_legendList_single'>
                  <span className='legend_color' style={{ background: `${resourceColor[index]}` }}></span>
                  <span className='legend_text'>{item}</span>
                </div>;
              })}
            </div>
            <p className='overview_flex_resource_right_title' style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{`${cpuDataName}` + 'Top 5（%）'}</p>
            <div className='overview_flex_resource_right_legend' onClick={() => setIsCpuLegend(!isCpuLegend)}>
              <StockOutlined style={{ color: themeStore.$s.theme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '18px' }} />
              <span style={{ color: themeStore.$s.theme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '14px' }} >{t('resource.legend')}</span>
            </div>
          </div>
          <OverviewEcharts overOption={cpuOption} idProps="cpu_index" echartHeight={'210px'} />
        </div>
      </div>
      <div className="overview_flex_resource">
        <div className="overview_flex_resource_left">
          <div className="overview_flex_resource_left_title">
            <img src={overviewMemory} className="overview_flex_resource_left_memoryimg" alt="" />
            <p style={{ fontWeight: 'bold', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7', margin: '0 8px' }}>{t('resource.memory')}</p>
            <p style={{ fontWeight: 'bold', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.total')}</p>
            <p style={{ fontWeight: 'bold', color: '#b37feb', margin: '0 2px 0 8px' }}>{(resourceData.memoryTotal / 1024 / 1024 / 1024).toFixed(2)}</p>
            <p style={{ marginLeft: '2px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>GiB</p>
          </div>
          <div className={`overview_flex_resource_left_progress ${isMemoryDataChose === 'node_memory_utilisation' ? 'overview_card_selected_memory_color' : ''}`} onClick={() => changeMemoryUsed('node_memory_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.usageRate')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{(resourceData.memoryUtilisation * 100).toFixed(1)}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{(resourceData.memoryUsage / 1024 / 1024 / 1024).toFixed(2)}GiB<span>&nbsp;/&nbsp;</span></p><p>{(resourceData.memoryTotal / 1024 / 1024 / 1024).toFixed(2)}GiB</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={Number((resourceData.memoryUtilisation * 100).toFixed(1))} showInfo={false} strokeColor={'#b37feb'} trailColor={'#dcdcdc'} />
            </div>
          </div>

          <div className={`overview_flex_resource_left_progress ${isMemoryDataChose === 'node_memory_requests_utilisation' ? 'overview_card_selected_memory_color' : ''}`} onClick={() => changeMemoryUsed('node_memory_requests_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.requestRate')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{(resourceData.memoryRequestsUtilisation * 100).toFixed(1)}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{(resourceData.memoryRequestsUsage / 1024 / 1024 / 1024).toFixed(2)}GiB<span>&nbsp;/&nbsp;</span></p><p>{(resourceData.memoryRequestsTotal / 1024 / 1024 / 1024).toFixed(2)}GiB</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={Number((resourceData.memoryRequestsUtilisation * 100).toFixed(1))} showInfo={false} strokeColor={'#b37feb'} trailColor={'#dcdcdc'} />
            </div>
          </div>

          <div className={`overview_flex_resource_left_progress ${isMemoryDataChose === 'node_memory_limits_utilisation' ? 'overview_card_selected_memory_color' : ''}`} onClick={() => changeMemoryUsed('node_memory_limits_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.limitRate')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{(resourceData.memoryLimitsUtilisation * 100).toFixed(1)}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{(resourceData.memoryLimitsUsage / 1024 / 1024 / 1024).toFixed(2)}GiB<span>&nbsp;/&nbsp;</span></p><p>{(resourceData.memoryLimitsTotal / 1024 / 1024 / 1024).toFixed(2)}GiB</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={Number((resourceData.memoryLimitsUtilisation * 100).toFixed(1))} showInfo={false} strokeColor={'#b37feb'} trailColor={'#dcdcdc'} />
            </div>
          </div>
        </div>
        <div className="overview_flex_resource_right">
          <div className="overview_flex_resource_right_top">
            <div className="overview_flex_resource_right_legendList" style={{ display: isMemoryLegend ? 'block' : 'none' }}>
              {memoryData.map((item, index) => {
                return <div className='overview_flex_resource_right_legendList_single'>
                  <span className='legend_color' style={{ background: `${resourceColor[index]}` }}></span>
                  <span className='legend_text'>{item}</span>
                </div>;
              })}
            </div>
            <p className='overview_flex_resource_right_title' style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{`${memoryDataName}` + 'Top 5（%）'}</p>
            <div className='overview_flex_resource_right_legend' onClick={() => setIsMemoryLegend(!isMemoryLegend)}>
              <StockOutlined style={{ color: themeStore.$s.theme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '18px' }} />
              <span style={{ color: themeStore.$s.theme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '14px' }} >{t('resource.legend')}</span>
            </div>
          </div>
          <OverviewEcharts overOption={memoryOption} echartHeight={'210px'} idProps="memory_index" />
        </div>
      </div>
      <div className="overview_flex_resource">
        <div className="overview_flex_resource_left">
          <div className="overview_flex_resource_left_title">
            <img src={overviewDisk} className="overview_flex_resource_left_diskimg" alt="" />
            <p style={{ fontWeight: 'bold', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7', margin: '0 8px' }}>{t('resource.storage')}</p>
          </div>

          <div className={`overview_flex_resource_left_progress ${isStorageDataChose === 'node_disk_size_utilisation' ? 'overview_card_selected_storage_color' : ''}`} onClick={() => changeStorageUsed('node_disk_size_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.diskUsage')}</p><p style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{(resourceData.diskUtilisation * 100).toFixed(1)}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{(resourceData.diskUsage / 1024 / 1024 / 1024).toFixed(2)}GiB<span>&nbsp;/&nbsp;</span></p><p>{(resourceData.diskTotal / 1024 / 1024 / 1024).toFixed(2)}GiB</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={Number((resourceData.diskUtilisation * 100).toFixed(1))} showInfo={false} strokeColor={'#4b8bea'} trailColor={'#dcdcdc'} />
            </div>
          </div>

          <div className={`overview_flex_resource_left_progress ${isStorageDataChose === 'node_disk_inode_utilisation' ? 'overview_card_selected_storage_color' : ''}`} onClick={() => changeStorageUsed('node_disk_inode_utilisation')}>
            <div className="overview_flex_resource_left_progress_top">
              <div style={{ display: 'flex' }}><p style={{ marginRight: '16px', color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('resource.inodeUsage')}</p><p style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{(resourceData.diskInodeUtilisation * 100).toFixed(1)}%</p></div>
              <div style={{ display: 'flex' }} className="overview_card_resource_textcolor"><p>{resourceData.diskInodeUsage > 10000 ? `${(resourceData.diskInodeUsage / 10000).toFixed(0)}${t('resource.tenThousand')}` : `${resourceData.diskInodeUsage}`}<span>&nbsp;/&nbsp;</span></p><p>{resourceData.diskInodeTotal > 10000 ? `${(resourceData.diskInodeTotal / 10000).toFixed(0)}${t('resource.tenThousand')}` : `${resourceData.diskInodeTotal}`}</p></div>
            </div>
            <div className="overview_flex_resource_left_progress_bottom">
              <Progress percent={Number((resourceData.diskInodeUtilisation * 100).toFixed(1))} showInfo={false} strokeColor={'#4b8bea'} trailColor={'#dcdcdc'} />
            </div>
          </div>
        </div>
        <div className="overview_flex_resource_right">
          <div className="overview_flex_resource_right_top">
            <div className="overview_flex_resource_right_legendList" style={{ display: isDiskLegend ? 'block' : 'none' }}>
              {storageData.map((item, index) => {
                return <div className='overview_flex_resource_right_legendList_single'>
                  <span className='legend_color' style={{ background: `${resourceColor[index]}` }}></span>
                  <span className='legend_text'>{item}</span>
                </div>;
              })}
            </div>
            <p className='overview_flex_resource_right_title' style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{`${storageDataName}` + 'Top 5（%）'}</p>
            <div className='overview_flex_resource_right_legend' onClick={() => setIsDiskLegend(!isDiskLegend)}>
              <StockOutlined style={{ color: themeStore.$s.theme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '18px' }} />
              <span style={{ color: themeStore.$s.theme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '14px' }} >{t('resource.legend')}</span>
            </div>
          </div>
          <OverviewEcharts overOption={storageOption} echartHeight={'210px'} idProps="space_index" />
        </div>
      </div>
    </div>
  </div>;
}