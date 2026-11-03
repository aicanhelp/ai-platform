/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import useLocalStorage from '@/hooks/useLocalStorage';
import { message, Popover } from 'antd';
import { StockOutlined } from '@ant-design/icons';
import { useIntl } from 'inula-intl';
import DountChart from '@/components/DountChart';
import LineChart from '@/components/LineChart';
import { solveEchartsClean } from '@/tools/utils';
import { mockData } from '@/mock/mock';
import { ResponseCode } from '@/common/constants';
import { useEffect, useState, useCallback } from 'openinula';
import { getColocationNodeCount, getColocationCluster } from '@/api/colocationApi';

let setTimeoutIdCluster = 0;
export default function ClusterMonitor({ propsType, propsColor }) {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const [statisticsData, setStatisticsData] = useState();

  const getStaticData = useCallback(async () => {
    let finallyData = {};
    const nodeRes = await getColocationNodeCount();
    if (nodeRes.status === ResponseCode.OK) {
      finallyData.nodeInfo = [
        {
          name: intl.formatMessage({ id: 'monitor.nodesWithColocationEnabled' }),
          value: nodeRes.data.data.colocationNodeCount,
        },
        {
          name: intl.formatMessage({ id: 'monitor.nodesWithoutColocationEnabled' }),
          value: nodeRes.data.data.nonColocationNodeCount,
        },
      ];
      finallyData.nodeTotal = nodeRes.data.data.totalNodeCount;
      finallyData.deploymentTotal =
        nodeRes.data.data.podCount.normalCount +
        nodeRes.data.data.podCount.hlsCount +
        nodeRes.data.data.podCount.lsCount +
        nodeRes.data.data.podCount.beCount;
      finallyData.deploymentInfo = [
        {
          name: intl.formatMessage({ id: 'monitor.nonColocationPods' }),
          value: nodeRes.data.data.podCount.normalCount,
        },
        {
          name: intl.formatMessage({ id: 'monitor.hlsPods' }),
          value: nodeRes.data.data.podCount.hlsCount,
        },
        {
          name: intl.formatMessage({ id: 'monitor.lsPods' }),
          value: nodeRes.data.data.podCount.lsCount,
        },
        {
          name: intl.formatMessage({ id: 'monitor.bePods' }),
          value: nodeRes.data.data.podCount.beCount,
        },
      ];
    }

    const clusterRes = await getColocationCluster();
    if (clusterRes.status === ResponseCode.OK) {
      const { cpuUse, memoryUse, cpuApply, memoryApply, normalCpu, normalMemory } = solveEchartsClean(
        clusterRes.data.data,
      );
      finallyData = { ...finallyData, cpuUse, memoryUse, cpuApply, memoryApply, normalCpu, normalMemory };
    }
    setStatisticsData(finallyData);
  }, []);

  const timeSearch = () => {
    localStorage.setItem('colocationTime1Id', setTimeoutIdCluster);
    clearTimeout(setTimeoutIdCluster);
    getStaticData();
    const clusterId = setTimeout(() => {
      timeSearch();
    }, 60000);
    setTimeoutIdCluster = clusterId;
  };

  useEffect(() => {
    if (setTimeoutIdCluster) {
      clearTimeout(setTimeoutIdCluster);
    }
    timeSearch();
    return () => {
      clearTimeout(setTimeoutIdCluster);
    };
  }, [currentTheme]);

  return (
    <div className="monitor_namespace">
      <div
        style={{
          background: currentTheme === 'dark' ? '#2a2d34ff' : '#fff',
          color: currentTheme === 'dark' ? '#fff' : '#333',
        }}
      >
        {contextHolder}
      </div>
      <div className="monitor_header">
        <div className="monitor_dount">
          <p className="title">{intl.formatMessage({ id: 'monitor.colocationNodeInfo' })}</p>
          <DountChart
            title={intl.formatMessage({ id: 'monitor.colocationNodeInfo' })}
            id="node"
            data={statisticsData?.nodeInfo}
            total={{ text: intl.formatMessage({ id: 'monitor.totalNodes' }), value: statisticsData?.nodeTotal }}
            colorList={propsColor}
          />
        </div>
        <div className="monitor_dount">
          <p className="title">{intl.formatMessage({ id: 'monitor.colocationLoadInfo' })}</p>
          <DountChart
            title={intl.formatMessage({ id: 'monitor.colocationLoadInfo' })}
            id="deployment"
            data={statisticsData?.deploymentInfo}
            total={{ text: intl.formatMessage({ id: 'monitor.totalPods' }), value: statisticsData?.deploymentTotal }}
            colorList={propsColor}
          />
        </div>
      </div>
      <div className="monitor_line_container">
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.clusterCpuUsage' })}</p>
          <LineChart id="cpuUse" data={statisticsData?.cpuUse} colorList={propsColor} isShowLegend={true} />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.clusterMemoryUsage' })}</p>
          <LineChart id="memoryUse" data={statisticsData?.memoryUse} colorList={propsColor} isShowLegend={true} />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.clusterOversoldCpuRequest' })}</p>
          <LineChart id="cpuApply" data={statisticsData?.cpuApply} colorList={propsColor} isShowLegend={true} />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.clusterOversoldMemoryRequest' })}</p>
          <LineChart id="memoryApply" data={statisticsData?.memoryApply} colorList={propsColor} isShowLegend={true} />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.clusterNormalCpuRequest' })}</p>
          <LineChart id="normalCpu" data={statisticsData?.normalCpu} colorList={propsColor} isShowLegend={true} />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.clusterNormalMemoryRequest' })}</p>
          <LineChart id="normalMemory" data={statisticsData?.normalMemory} colorList={propsColor} isShowLegend={true} />
        </div>
      </div>
    </div>
  );
}
