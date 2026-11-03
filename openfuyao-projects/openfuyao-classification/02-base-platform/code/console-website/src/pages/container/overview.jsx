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
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant.js';
import { message } from 'antd';
import ManageNodeIcon from '@/assets/images/manageNodeIcon.png';
import WorkerNodeIcon from '@/assets/images/workerNodeIcon.png';
import '@/styles/pages/overview.less';
import { useHistory } from 'inula-router';
import { useCallback, useEffect, useRef, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import {
  solveNodeBaseInfo,
  solveNodeChartInfo,
  getWorkloadStatusJudge,
  getDaemonSetStatus,
  debounce,
} from '@/tools/utils';
import {
  listNodes,
  listNamespaces,
  getAuthorList,
  getHelmsData,
  listPods,
  listDeployments,
  listStatefulSets,
  listDaemonSets,
  getClusterRolesData,
  getOverviewUserList,
  getServiceTimeZoneInterface,
} from '@/api/containerApi';
import { getAlertOptions } from '@/api/containerAlertApi';
import { ResponseCode } from '@/common/constants';
import OverviewEcharts from '@/components/OverviewEcharts';
import NetworkLoad from '@/pages/container/overview/NetworkLoad';
import Resource from './overview/Resource';
import { getOverviewChartList } from '@/utils/common';
import DiffTimeInfo from '@/components/DiffTimeInfo';
import ToastMsg from '@/components/ToastMsg';

const DEFAULT_CIRCLE_ECHARTS_DATA = {
  radius: ['59%', '70%'],
  center: ['30%', '60%'],
  font: 16,
  legendRight: '10%',
  legendTop: '30%',
};

// 由 helm 应用/扩展列表构造环形图 option（应用与扩展两块图表共用）
const solveManageCircleOption = (items, circleConf, theme) => {
  let deployedCount = 0;
  let failedCount = 0;
  let pendingCount = 0;
  items.forEach(i => {
    if (i.info.status === 'deployed') {
      deployedCount++;
    } else if (i.info.status === 'failed') {
      failedCount++;
    } else {
      pendingCount++;
    }
  });
  const total = items.length || 1;
  const data = [
    { name: 'Deployed', value: deployedCount, percent: deployedCount / total },
    { name: 'Pending', value: pendingCount, percent: pendingCount / total },
    { name: 'Failed', value: failedCount, percent: failedCount / total },
  ];
  const labelColor = theme === 'light' ? 'black' : '#f7f7f7';
  const appSeries = [
    {
      name: '',
      type: 'pie',
      radius: circleConf.radius,
      center: circleConf.center,
      label: {
        show: true,
        useHTML: true,
        formatter: () => {
          return `  ${items.length}  \n \n Total`;
        },
        fontSize: circleConf.font,
        textAlign: 'center',
        position: 'center',
        textStyle: {
          color: labelColor,
        },
      },
      labelLine: {
        show: false,
      },
      data: data,
      itemStyle: {
        color: (params) => {
          let colors = ['#09aa71', '#4b8bea', '#e7434a'];
          return colors[params.dataIndex % colors.length];
        },
      },
    },
  ];
  return {
    tooltip: {
      trigger: 'item',
      formatter: () => {
        return data.map(item => `${item.name}:${item.value}`).join('<br/>');
      },
      borderColor: '#fff',
      borderWidth: 1,
    },
    legend: {
      right: circleConf.legendRight,
      top: circleConf.legendTop,
      orient: '',
      borderWidth: 0,
      itemGap: 30,
      itemWidth: 8,
      itemHeight: 8,
      icon: 'circle',
      formatter: (name) => {
        let _name = '';
        data.forEach(item => {
          if (item.name === name) {
            _name = `${item.name}` + ' ' + `${item.value}`;
          }
        });
        return _name;
      },
      textStyle: {
        color: labelColor,
      },
    },
    series: appSeries,
  };
};
export default function OverviewPage() {
  const history = useHistory();
  const { t } = useTranslation(['overview', 'common', 'rbac']);
  const [messageApi, contextHolder] = message.useMessage();
  const intervalStore = useStore('interval');
  const [nodeInfo, setNodeInfo] = useState({});
  const [namespaceCount, setNamespaceCount] = useState(0);
  const [userCount, setUserCount] = useState(0);
  const [authorCount, setAuthorCount] = useState(0);
  const [alertCount, setAlertCount] = useState({});
  const [nodeDataOption, setNodeDataOption] = useState();
  const [nodeTotal, setNodeTotal] = useState();
  const [podDataOption, setPodDataOption] = useState();
  const [podTotal, setPodTotal] = useState();
  const [workloadDataOption, setWorkloadDataOption] = useState();
  const [extendDataOption, setExtendDataOption] = useState();
  const [appDataOption, setAppDataOption] = useState();
  const [deploymentData, setDeploymentData] = useState([]);
  const [statefulSetData, setStatefulSetData] = useState([]);
  const [daemonSetData, setDaemonSetData] = useState([]);
  const [diffTime, setDiffTime] = useState(0);
  const circleEchartsDataRef = useRef(DEFAULT_CIRCLE_ECHARTS_DATA);
  const timerRef = useRef(0);
  const themeStore = useStore('theme');
  // 获取节点数据
  const getNodeManageInfo = async () => {
    try {
      const res = await listNodes();
      if (res.status === ResponseCode.OK) {
        // 处理数据
        setNodeInfo(solveNodeBaseInfo(res.data.items, true));
        let _arr = solveNodeChartInfo(res.data.items);
        let nodeSeries = [
          {
            name: 'Ready',
            type: 'bar',
            stack: 'total',
            data: [_arr.filter(item => item.item === 'Ready').length],
            itemStyle: { borderRadius: [5, 5, 5, 5], color: '#77aef7ff' },
            barWidth: 10,
          },
          {
            name: 'NotReady',
            type: 'bar',
            stack: 'total',
            data: [_arr.filter(item => item.item === 'NotReady').length],
            itemStyle: { borderRadius: [5, 5, 5, 5], color: '#f4840cff' },
            barWidth: 10,
          },
          {
            name: 'Unknown',
            type: 'bar',
            stack: 'total',
            data: [_arr.filter(item => item.item === 'Unknown').length],
            itemStyle: { borderRadius: [5, 5, 5, 5], color: '#ccccccff' },
            barWidth: 10,
          },
        ];
        let nodeOption = {
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'none',
            },
          },
          legend: {
            left: '2%',
            bottom: '35%',
            borderRadius: '50%',
            itemWidth: 8,
            itemHeight: 8,
            icon: 'circle',
            formatter: (name) => {
              let _name = '';
              nodeSeries.forEach(item => {
                if (item.name === name) {
                  _name = `${item.name}` + ' ' + `${item.data[0]}`;
                }
              });
              return _name;
            },
            textStyle: {
              color: themeStore.$s.theme === 'light' ? 'black' : '#fff',
            },
          },
          grid: {
            left: '0%',
            bottom: '10%',
            containLabel: true,
          },
          xAxis: {
            type: 'value',
            axisLabel: {
              show: false,
            },
            splitLine: {
              show: false,
            },
            axisTick: {
              show: false,
            },
            axisLine: {
              show: false,
            },
          },
          yAxis: {
            type: 'category',
            axisLine: {
              show: false,
            },
            axisTick: {
              show: false,
            },
            data: [''],
          },
          series: nodeSeries,
        };
        setNodeTotal(_arr.length);
        setNodeDataOption(nodeOption);
      }
    } catch (e) {
      messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message }));
    }
  };

  // 获取命名空间数量
  const getNameSpacesInfo = async () => {
    try {
      const res = await listNamespaces();
      setNamespaceCount(res.data.items.length);
    } catch {
      setNamespaceCount(0);
    }
  };

  // 获取用户数量
  const getUserInfo = async () => {
    try {
      const res = await getOverviewUserList();
      setUserCount(res.data.items.length);
    } catch {
      setUserCount(0);
    }
  };

  // 获取角色数量
  const getAuthorInfo = async () => {
    try {
      const res = await getAuthorList();
      const resCluster = await getClusterRolesData();
      setAuthorCount(res.data.items.length + resCluster.data.items.length);
    } catch {
      setAuthorCount(0);
    }
  };

  // 获取告警数量
  const getAlertInfo = async () => {
    try {
      const res = await getAlertOptions();
      let _obj = {};
      let warningCount = 0;
      let infoCount = 0;
      let criticalCount = 0;
      res.data.forEach(i => {
        if (i.labels.severity === 'warning') {
          warningCount++;
        } else if (i.labels.severity === 'info') {
          infoCount++;
        } else {
          criticalCount++;
        }
      });
      _obj = {
        warning: warningCount,
        info: infoCount,
        critical: criticalCount,
      };
      setAlertCount(_obj);
    } catch {
      setAlertCount({});
    }
  };

  // 获取应用数据
  const getApplicationManageInfo = async () => {
    try {
      const res = await getHelmsData(false);
      if (res.status === ResponseCode.OK) {
        setAppDataOption(
          solveManageCircleOption(res.data?.data?.items ?? [], circleEchartsDataRef.current, themeStore.$s.theme),
        );
      }
    } catch {
      setAppDataOption([]);
    }
  };

  // 获取扩展数据
  const getExtendManageInfo = async () => {
    try {
      const res = await getHelmsData(true);
      if (res.status === ResponseCode.OK) {
        setExtendDataOption(
          solveManageCircleOption(res.data?.data?.items ?? [], circleEchartsDataRef.current, themeStore.$s.theme),
        );
      }
    } catch {
      setExtendDataOption([]);
    }
  };

  // 生成podService
  const birthPodSeries = (chartArr) => {
    return [
      {
        name: 'Running',
        type: 'bar',
        stack: 'total',
        data: [chartArr.filter(item => item.item === 'Running').length],
        itemStyle: {
          borderRadius: [5, 5, 5, 5], color: '#09AA71',
        },
        barWidth: 10,
      },
      {
        name: 'Pending',
        type: 'bar',
        stack: 'total',
        data: [chartArr.filter(item => item.item === 'Pending').length],
        itemStyle: {
          borderRadius: [5, 5, 5, 5], color: '#fcd72e',
        },
        barWidth: 10,
      },
      {
        name: 'Succeeded',
        type: 'bar',
        stack: 'total',
        data: [chartArr.filter(item => item.item === 'Succeeded').length],
        itemStyle: {
          borderRadius: [5, 5, 5, 5], color: '#77aef7',
        },
        barWidth: 10,
      },
      {
        name: 'Failed',
        type: 'bar',
        stack: 'total',
        data: [chartArr.filter(item => item.item === 'Failed').length],
        itemStyle: {
          borderRadius: [5, 5, 5, 5], color: '#e7434a',
        },
        barWidth: 10,
      },
      {
        name: 'Unknown',
        type: 'bar',
        stack: 'total',
        data: [chartArr.filter(item => item.item === 'Unknown').length],
        itemStyle: {
          borderRadius: [5, 5, 5, 5], color: '#cccccc',
        },
        barWidth: 10,
      },
    ];
  };

  const birthPodOption = (podSeries) => {
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'none',
        },
      },
      legend: {
        left: '0.5%',
        bottom: '35%',
        borderRadius: '50%',
        itemWidth: 8,
        itemHeight: 8,
        icon: 'circle',
        formatter: (name) => {
          let _name = '';
          podSeries.forEach(item => {
            if (item.name === name) {
              _name = `${item.name}` + ' ' + `${item.data[0]}`;
            }
          });
          return _name;
        },
        textStyle: {
          color: themeStore.$s.theme === 'light' ? 'black' : '#f7f7f7',
        },
      },
      grid: {
        left: '0%',
        bottom: '10%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        axisLabel: {
          show: false,
        },
        splitLine: {
          show: false,
        },
        axisTick: {
          show: false,
        },
        axisLine: {
          show: false,
        },
      },
      yAxis: {
        type: 'category',
        axisLine: {
          show: false,
        },
        axisTick: {
          show: false,
        },
        data: [''],
      },
      series: podSeries,
    };
  };

  // 获取pod数据
  const getPodInfo = async () => {
    try {
      const res = await listPods();
      if (res.status === ResponseCode.OK) {
        const chartArr = getOverviewChartList(res.data.items);
        let podSeries = birthPodSeries(chartArr);
        let podOption = birthPodOption(podSeries);
        setPodTotal(chartArr.length);
        setPodDataOption(podOption);
      }
    } catch {
      setPodDataOption([]);
    }
  };

  // 获取工作负载 Deployment数据
  const getDeploymentInfo = async () => {
    try {
      const res = await listDeployments();
      if (res.status === ResponseCode.OK) {
        let arr = [...res.data.items];
        let runningCount = 0;
        let pendingCount = 0;
        let FailedCount = 0;
        let _arr = [];
        arr.forEach(i => {
          if (getWorkloadStatusJudge(i.status) === 'Failed') {
            FailedCount++;
            _arr.push({ name: 'Deployment', item: 'Failed', count: FailedCount });
          } else if (getWorkloadStatusJudge(i.status) === 'Active') {
            runningCount++;
            _arr.push({ name: 'Deployment', item: 'Active', count: runningCount });
          } else {
            pendingCount++;
            _arr.push({ name: 'Deployment', item: 'Updating', count: pendingCount });
          }
        });
        setDeploymentData(_arr);
      }
    } catch {
      setDeploymentData([]);
    }
  };

  // 获取工作负载 Statefulset数据
  const getStatefulSetInfo = async () => {
    try {
      const res = await listStatefulSets();
      if (res.status === ResponseCode.OK) {
        let arr = [...res.data.items];
        let runningCount = 0;
        let pendingCount = 0;
        let FailedCount = 0;
        let _arr = [];
        arr.forEach(i => {
          if (getWorkloadStatusJudge(i.status) === 'Failed') {
            FailedCount++;
            _arr.push({ name: 'StatefulSet', item: 'Failed', count: FailedCount });
          } else if (getWorkloadStatusJudge(i.status) === 'Active') {
            runningCount++;
            _arr.push({ name: 'StatefulSet', item: 'Active', count: runningCount });
          } else {
            pendingCount++;
            _arr.push({ name: 'StatefulSet', item: 'Updating', count: pendingCount });
          }
        });
        setStatefulSetData(_arr);
      }
    } catch {
      setStatefulSetData([]);
    }
  };

  // 获取工作负载 DaemonSet数据
  const getDaemonSetInfo = async () => {
    try {
      const res = await listDaemonSets();
      if (res.status === ResponseCode.OK) {
        let arr = [...res.data.items];
        let runningCount = 0;
        let pendingCount = 0;
        let FailedCount = 0;
        let _arr = [];
        arr.forEach(i => {
          if (getDaemonSetStatus(i.status) === 'Failed') {
            FailedCount++;
            _arr.push({ name: 'DaemonSet', item: 'Failed', count: FailedCount });
          } else if (getDaemonSetStatus(i.status) === 'Active') {
            runningCount++;
            _arr.push({ name: 'DaemonSet', item: 'Active', count: runningCount });
          } else {
            pendingCount++;
            _arr.push({ name: 'DaemonSet', item: 'Updating', count: pendingCount });
          }
        });
        setDaemonSetData(_arr);
      }
    } catch {
      setDaemonSetData([]);
    }
  };

  // 工作负载数据总和
  const getWorkLoadInfo = useCallback(() => {
    let workloadSeries = [
      {
        name: 'Active',
        type: 'bar',
        stack: 'total',
        data: [
          daemonSetData.filter(item => item.item === 'Active').length,
          statefulSetData.filter(item => item.item === 'Active').length,
          deploymentData.filter(item => item.item === 'Active').length,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#09AA71' },
        barWidth: 10,
      },
      {
        name: 'Updating',
        type: 'bar',
        stack: 'total',
        data: [
          daemonSetData.filter(item => item.item === 'Updating').length,
          statefulSetData.filter(item => item.item === 'Updating').length,
          deploymentData.filter(item => item.item === 'Updating').length,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#fcd72e' },
        barWidth: 10,
      },
      {
        name: 'Failed',
        type: 'bar',
        stack: 'total',
        data: [
          daemonSetData.filter(item => item.item === 'Failed').length,
          statefulSetData.filter(item => item.item === 'Failed').length,
          deploymentData.filter(item => item.item === 'Failed').length,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#e7434a' },
        barWidth: 10,
      },
    ];
    let workloadOption = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
      },
      legend: {
        right: '10%',
        top: '10%',
        borderRadius: '50%',
        borderWidth: 0,
        padding: 5,
        itemGap: 10,
        itemWidth: 8,
        itemHeight: 8,
        icon: 'circle',
        formatter: (name) => {
          let _name = '';
          workloadSeries.forEach(item => {
            if (item.name === name) {
              _name = item.name;
            }
          });
          return _name;
        },
        textStyle: {
          color: themeStore.$s.theme === 'light' ? 'black' : '#f7f7f7',
        },
      },
      grid: {
        left: '4%',
        bottom: '',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        axisLabel: {
          show: true,
        },
        splitLine: {
          show: true,
          lineStyle: {
            type: 'dashed',
            color: themeStore.$s.theme === 'light' ? '#f7f7f7' : '#444444',
          },
        },
        axisTick: {
          show: false,
        },
        axisLine: {
          show: false,
        },
      },
      yAxis: {
        type: 'category',
        axisLine: {
          show: false,
        },
        axisLabel: {
          textStyle: {
            color: themeStore.$s.theme === 'light' ? 'black' : '#89939b',
          },
        },
        axisTick: {
          show: false,
        },
        data: ['DaemonSet', 'StatefulSet', 'Deployment'],
      },
      series: workloadSeries,
    };
    setWorkloadDataOption(workloadOption);
  }, [deploymentData, statefulSetData, daemonSetData]);

  const goAlertPage = (type) => {
    history.push({
      pathname: `/${containerRouterPrefix}/alert/alertIndex`,
      state: { alertType: type },
    });
  };

  const timeSearch = async () => {
    clearTimeout(timerRef.current);
    await Promise.allSettled([
      getTimeService(),
      getNodeManageInfo(),
      getNameSpacesInfo(),
      getUserInfo(),
      getAuthorInfo(),
      getAlertInfo(),
      getApplicationManageInfo(),
      getExtendManageInfo(),
      getPodInfo(),
      getDeploymentInfo(),
      getStatefulSetInfo(),
      getDaemonSetInfo(),
    ]);
    timerRef.current = setTimeout(() => {
      timeSearch();
    }, 30000);
  };
  const reasizeEchartCircle = () => {
    const width = window.innerWidth;
    if (width >= 1280 && width <= 1440) {
      circleEchartsDataRef.current = {
        radius: ['35%', '45%'],
        center: ['24%', '58%'],
        font: 12,
        legendRight: '0%',
        legendTop: '30%',
      };
    } else if (width >= 1441 && width <= 1680) {
      circleEchartsDataRef.current = {
        radius: ['40%', '50%'],
        center: ['28%', '60%'],
        font: 14,
        legendRight: '4%',
        legendTop: '30%',
      };
    } else if (width >= 1681) {
      circleEchartsDataRef.current = {
        radius: ['59%', '70%'],
        center: ['30%', '60%'],
        font: 14,
        legendRight: '6%',
        legendTop: '30%',
      };
    } else {
      circleEchartsDataRef.current = {
        radius: ['35%', '45%'],
        center: ['24%', '58%'],
        font: 12,
        legendRight: '0%',
        legendTop: '30%',
      };
    }
  };

  useEffect(() => {
    clearTimeout(timerRef.current);
    reasizeEchartCircle();
    if (intervalStore.$s.loginStatus) {
      timeSearch();
    }
    return () => clearTimeout(timerRef.current);
  }, [themeStore.$s.theme, intervalStore.$s.loginStatus]);

  useEffect(() => {
    const handleResize = debounce(() => {
      reasizeEchartCircle();
      getApplicationManageInfo();
      getExtendManageInfo();
    }, 300);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [getApplicationManageInfo, getExtendManageInfo]);

  useEffect(() => {
    getWorkLoadInfo();
  }, [getWorkLoadInfo]);

  const getTimeService = useCallback(async () => {
    const res = await getServiceTimeZoneInterface();
    if (res.status === ResponseCode.OK) {
      if (res.data.data?.currentServerTime) {
        setDiffTime(Math.abs(Math.floor(new Date().getTime() / 1000) - (Number(res.data.data.currentServerTime))));
      }
    }
  }, []);

  return <div className="child_content overview">
    <ToastMsg contextHolder={contextHolder} />
    <div className="page_header">
      <BreadCrumbCom items={[{ title: t('module.title'), path: `/${containerRouterPrefix}/overview` }]} />
    </div>
    {diffTime >= 10 && <DiffTimeInfo time={diffTime} className={'monitor_diff_time'} />}
    <div className="overview_card" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#f7f7f7' : 'black' }}>
      <div className="overview_flex">
        <div className="overview_flex_node overview_flex_pod overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="overview_flex_node_progress overviewBorderRight" style={{ position: 'relative' }}>
            <p className="overview_card_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/nodeManage`); }}>{t('node.title')}</p>
            <p style={{ fontSize: '12px', color: themeStore.$s.theme === 'light' ? '#675e5e' : '#f7f7f7', position: 'absolute', top: '32%', left: 0 }}>{t('node.total')}<span style={{ fontSize: '13px', color: themeStore.$s.theme === 'light' ? '#675e5e' : '#f7f7f7', marginLeft: '4px' }}>{nodeTotal}</span></p>
            <div className="overview_flex_node_progress_chart">
              <OverviewEcharts idProps={'node'} overOption={nodeDataOption} echartHeight={'80px'} />
            </div>
          </div>
          <div className="overview_flex_node_count">
            <div className="overview_flex_node_manage overview_flex_node_manage1">
              <p className="overview_card_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/nodeManage`); }}>{t('node.manageNode')}</p>
              <div className="overview_flex_node_manage_content">
                <div>
                  <img src={ManageNodeIcon} style={{ marginBottom: '8px' }} />
                </div>
                <div className="overview_flex_node_manage_content_count">
                  <div className="overview_flex_node_manage_content_count_top">
                    <span className="first_span">{nodeInfo.manageUnnormalNode}</span>
                    <span className="nomarl_span" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>/</span>
                    <span className="nomarl_span" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{nodeInfo.manageNode}</span>
                  </div>
                  <div className="overview_flex_node_manage_content_count_bottom">
                    <span className="nomarl_span" >{t('node.abnormal')}</span>
                    <span className="nomarl_span">&nbsp;/&nbsp;</span>
                    <span className="nomarl_span">{t('node.total')}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="overview_flex_node_manage">
              <p className="overview_card_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/nodeManage`); }}>{t('node.workerNode')}</p>
              <div className="overview_flex_node_manage_content">
                <div>
                  <img src={WorkerNodeIcon} style={{ marginBottom: '8px' }} />
                </div>
                <div className="overview_flex_node_manage_content_count">
                  <div className="overview_flex_node_manage_content_count_top">
                    <span className="first_span">{nodeInfo.unnormalNode}</span>
                    <span className="nomarl_span" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>/</span>
                    <span className="nomarl_span" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{nodeInfo.workerNode}</span>
                  </div>
                  <div className="overview_flex_node_manage_content_count_bottom">
                    <span className="nomarl_span">{t('node.abnormal')}</span>
                    <span className="nomarl_span">&nbsp;/&nbsp;</span>
                    <span className="nomarl_span">{t('node.total')}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="overview_flex_namespace_user_box">
          <div className="overview_flex_namespace overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
            <div className="overview_flex_namespace_group">
              <p className="overview_card_title" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} >{t('namespace.title')}</p>
              <p className="overview_flex_namespace_group_count cursor" onClick={() => { history.push(`/${containerRouterPrefix}/namespace/namespaceManage`); }}>{namespaceCount}</p>
            </div>
          </div>
          <div className="overview_flex_user overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
            <div className="overview_flex_user_group">
              <p className="overview_card_title" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} >{t('serviceAccount.title')}</p>
              <p className="overview_flex_user_group_count cursor" onClick={() => { history.push(`/${containerRouterPrefix}/userManage/serviceAccount`); }}>{userCount}</p>
            </div>
            <div className="overview_flex_author_group">
              <p className="overview_card_title" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{t('role.title')}</p>
              <p className="overview_flex_author_group_count cursor" onClick={() => { history.push(`/${containerRouterPrefix}/userManage/role`); }}>{authorCount}</p>
            </div>
          </div>
        </div>
      </div>
      <div className="overview_flex ">
        <div className="overview_flex_node overview_flex_pod overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="overview_flex_pod_progress" style={{ position: 'relative' }}>
            <p className="overview_card_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/workload/pod`); }}>{t('pod.title')}</p>
            <p style={{ fontSize: '12px', color: themeStore.$s.theme === 'light' ? '#675e5e' : '#f7f7f7', position: 'absolute', top: '34%', left: 0 }}>{t('pod.total')}<span style={{ fontSize: '13px', color: themeStore.$s.theme === 'light' ? '#675e5e' : '#f7f7f7', marginLeft: '4px' }}>{podTotal}</span></p>
            <div className="overview_flex_pod_progress_chart">
              <OverviewEcharts idProps={'pod'} overOption={podDataOption} echartHeight={'80px'} />
            </div>
          </div>
        </div>
        <div className="overview_flex_permission overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="overview_flex_permission_group">
            <p className="overview_card_title" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{t('alert.title')}</p>
            <div className="overview_flex_permission_group_alert">
              <p className="overview_flex_permission_group_alert_heigh cursor" onClick={() => goAlertPage('critical')}>{alertCount.critical}</p>
              <span className="normal_span" style={{ color: themeStore.$s.theme !== 'light' && '#9ea9b3' }}>{t('alert.critical')}</span>
            </div>
          </div>
          <div className="overview_flex_permission_group">
            <p className="overview_card_title cursor"></p>
            <div className="overview_flex_permission_group_alert">
              <p className="overview_flex_permission_group_alert_middle cursor" onClick={() => goAlertPage('warning')}>{alertCount.warning}</p>
              <span className="normal_span" style={{ color: themeStore.$s.theme !== 'light' && '#9ea9b3' }}>{t('alert.warning')}</span>
            </div>
          </div>
          <div className="overview_flex_permission_group">
            <p className="overview_card_title cursor"></p>
            <div className="overview_flex_permission_group_alert">
              <p className="overview_flex_permission_group_alert_low cursor" onClick={() => goAlertPage('info')}>{alertCount.info}</p>
              <span className="normal_span" style={{ color: themeStore.$s.theme !== 'light' && '#9ea9b3' }}>{t('alert.info')}</span>
            </div>
          </div>
        </div>
      </div>
      <div className="overview_flex">
        <div className="overview_flex_application overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="overview_flex_application_donutchart" style={{ position: 'relative' }}>
            <p className="overview_card_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/applicationManageHelm`); }}>{t('application.title')}</p>
            <OverviewEcharts type='circle' idProps="application" overOption={appDataOption} echartHeight={'180px'} />
          </div>
        </div>
        <div className="overview_flex_application overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="overview_flex_application_donutchart" style={{ position: 'relative' }}>
            <p className="overview_card_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/extendManage`); }}>{t('extension.title')}</p>
            <OverviewEcharts type='circle' idProps="extend" overOption={extendDataOption} echartHeight={'180px'} />
          </div>
        </div>
        <div className="overview_flex_workload overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="overview_flex_workload_chart">
            <p className="overview_card_title overview_card_workload_title cursor" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }} onClick={() => { history.push(`/${containerRouterPrefix}/workload/deployment`); }}>{t('workload.title')}</p>
            <OverviewEcharts idProps={'workload'} overOption={workloadDataOption} echartHeight={'260px'} />
          </div>
        </div>
      </div>
      <Resource />
      <NetworkLoad />
    </div>
  </div >;
}