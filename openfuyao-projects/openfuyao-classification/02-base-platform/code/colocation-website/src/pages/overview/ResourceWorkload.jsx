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
import { containerRouterPrefix, ResponseCode } from '../../common/constants';
import '../../styles/overview.less';
import { useCallback, useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import { getNodesAndPodsCount, getResourcesTop5 } from '@/api/colocationApi';
import OverviewEcharts from './OverviewEcharts';
import useLocalStorage from '@/hooks/useLocalStorage';
import { filterEchartsOption } from '../../utils/common';
import EmptyData from '../../components/EmptyData';
import colocationNodeIcon from '../../assets/images/node/colocationNodeIcon.png';
import normalNodeIcon from '../../assets/images/node/normalNodeIcon.png';
import oversoldNodeIcon from '../../assets/images/node/oversoldNodeIcon.png';
let setTimeoutId = 0;

export default function ResourceWorkload() {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const [workloadDataOption, setWorkloadDataOption] = useState();
  const [podData, setPodData] = useState({}); // pod数据源
  const [deploymentData, setDeploymentData] = useState({}); // deployment数据源
  const [statefulSetData, setStatefulSetData] = useState({}); // statefulSet数据源
  const [cronJobData, setCronJobData] = useState({}); // cronJob数据源
  const [jobData, setJobData] = useState({}); // job数据源
  // 图表
  const [allNodesCount, setAllNodesCount] = useState(0);
  const [colocationNodesCount, setColocationNodesCount] = useState(0);
  const [nonColocationNodesCount, setNonColocationNodesCount] = useState(0);
  const [oversubcriptionCount, setOversubcriptionCount] = useState(0);

  // echarts配置项公共部分
  const getEchartsOption = useCallback(
    (series, leftName) => {
      return filterEchartsOption(series, leftName, currentTheme);
    },
    [currentTheme],
  );

  // 获取节点个数及工作负载图表数据
  const getColocationNodeManageInfo = useCallback(async () => {
    try {
      const res = await getNodesAndPodsCount();
      if (res.status === ResponseCode.OK) {
        // 节点个数赋值
        setAllNodesCount(res.data.data.totalNodeCount);
        setColocationNodesCount(res.data.data.colocationNodeCount);
        setNonColocationNodesCount(res.data.data.nonColocationNodeCount);
        setOversubcriptionCount(res.data.data.oversubcriptionNodeCount);
        //  处理echarts数据
        setPodData(res.data.data.podCount);
        setDeploymentData(res.data.data.deploymentCount);
        setStatefulSetData(res.data.data.statefulsetCount);
        setJobData(res.data.data.jobCount);
        setCronJobData(res.data.data.cronjobCount);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setCronJobData({});
      setJobData({});
      setStatefulSetData({});
      setDeploymentData({});
      setPodData({});
    }
  }, []);

  // 工作负载数据总和
  const getWorkLoadInfo = useCallback(() => {
    let workloadSeries = [
      {
        name: intl.formatMessage({ id: 'overview.normal' }),
        type: 'bar',
        stack: 'total',
        data: [
          cronJobData.normalCount,
          jobData.normalCount,
          statefulSetData.normalCount,
          deploymentData.normalCount,
          podData.normalCount,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#77AEF7' },
        barWidth: 10,
      },
      {
        name: intl.formatMessage({ id: 'workload.status.oversold' }),
        type: 'bar',
        stack: 'total',
        data: [
          cronJobData.oversubcriptionCount,
          jobData.oversubcriptionCount,
          statefulSetData.oversubcriptionCount,
          deploymentData.oversubcriptionCount,
          podData.oversubcriptionCount,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#CEC4FF' },
        barWidth: 10,
      },
      {
        name: intl.formatMessage({ id: 'workload.status.online' }),
        type: 'bar',
        stack: 'total',
        data: [
          cronJobData.colocationOnlineCount,
          jobData.colocationOnlineCount,
          statefulSetData.colocationOnlineCount,
          deploymentData.colocationOnlineCount,
          podData.colocationOnlineCount,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#6AD37A' },
        barWidth: 10,
      },
      {
        name: intl.formatMessage({ id: 'workload.status.offline' }),
        type: 'bar',
        stack: 'total',
        data: [
          cronJobData.colocationOfflineCount,
          jobData.colocationOfflineCount,
          statefulSetData.colocationOfflineCount,
          deploymentData.colocationOfflineCount,
          podData.colocationOfflineCount,
        ],
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#E2F8CC' },
        barWidth: 10,
      },
    ];
    let workloadName = ['CronJob', 'Job', 'StatefulSet', 'Deployment', 'Pod'];
    let workloadOption = getEchartsOption(workloadSeries, workloadName);
    setWorkloadDataOption(workloadOption);
  }, [podData, deploymentData, statefulSetData, cronJobData, jobData]);

  const timeSearch = () => {
    // 存入storage
    localStorage.setItem('colocationTime1Id', setTimeoutId);
    clearTimeout(setTimeoutId);
    getColocationNodeManageInfo();
    const id = setTimeout(() => {
      timeSearch();
    }, 60000);
    setTimeoutId = id;
  };

  useEffect(() => {
    if (setTimeoutId) {
      clearTimeout(setTimeoutId);
    }
    timeSearch();
    return () => {
      clearTimeout(setTimeoutId);
    };
  }, [currentTheme]);

  useEffect(() => {
    getWorkLoadInfo();
  }, [getWorkLoadInfo]);

  return (
    <>
      <div className="overview_flex">
        <div
          className="overview_flex_node overview_card_shadow"
          style={{ backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff' }}
        >
          <div className="overview_flex_node_chart">
            <p className="node_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
              {intl.formatMessage({ id: 'overview.nodes' })}
            </p>
            <div className="node_box">
              <div className="node_box_single">
                <p className="title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                  {intl.formatMessage({ id: 'overview.normalNode' })}
                </p>
                <div className="image">
                  <img src={normalNodeIcon} />
                </div>
                <p className="count">{nonColocationNodesCount ? nonColocationNodesCount : 0}</p>
              </div>
              <div className="node_box_single">
                <p className="title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                  {intl.formatMessage({ id: 'overview.colocationNode' })}
                </p>
                <div className="image">
                  <img src={colocationNodeIcon} />
                </div>
                <p className="count">{colocationNodesCount ? colocationNodesCount : 0}</p>
              </div>
              <div className="node_box_single">
                <p className="title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                  {intl.formatMessage({ id: 'overview.oversoldNode' })}
                </p>
                <div className="image">
                  <img src={oversoldNodeIcon} />
                </div>
                <p className="count">{oversubcriptionCount ? oversubcriptionCount : 0}</p>
              </div>
            </div>
          </div>
        </div>
        <div
          className="overview_flex_workload overview_card_shadow"
          style={{ backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff' }}
        >
          <div className="overview_flex_workload_chart">
            <p className="workload_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
              {intl.formatMessage({ id: 'overview.workloadDistribution' })}
            </p>
            {Object.keys(podData).length !== 0 ? (
              <OverviewEcharts idProps={'workload'} overOption={workloadDataOption} echartHeight={'260px'} />
            ) : (
              <EmptyData />
            )}
          </div>
        </div>
      </div>
    </>
  );
}
