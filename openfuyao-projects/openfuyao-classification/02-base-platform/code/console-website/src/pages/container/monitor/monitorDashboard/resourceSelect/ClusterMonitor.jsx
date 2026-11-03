/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import MonitorPod from '@/assets/images/monitor/monitorPod.png';
import MonitorNormal from '@/assets/images/monitor/monitorNormal.png';
import MonitorNodeUnNormal from '@/assets/images/monitor/monitorNodeUnnormal.png';
import DataAreaChart from '@/components/echarts/DataAreaChart';
import DataDonutChart from '@/components/echarts/DataDonutChart';
import { Table } from 'antd';
import { solveBytesToMiB, solveUnitMonitor } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';

export default function ClusterMonitor({
  data,
  timelyData,
  loaded,
  propsColor,
  refreshFn,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('monitoring');

  const columns = useMemo(() => [
    {
      title: t('table.namespace'),
      key: 'namespace',
      render: (_, record) => record.namespace,
    },
    {
      title: t('table.pods'),
      key: 'pods',
      render: (_, record) => record.pods !== '--' && !isNaN(record.pods) ? record.pods : '--',
    },
    {
      title: t('table.workloads'),
      key: 'workloads',
      render: (_, record) => record.workloads !== '--' && !isNaN(record.workloads) ? record.workloads : '--',
    },
    {
      title: t('table.memoryUsage'),
      key: 'memory',
      render: (_, record) => record.usage !== '--' && !isNaN(record.usage) ? `${record.usage} MiB` : '--',
    },
    {
      title: t('table.memoryRequests'),
      key: 'requests',
      render: (_, record) => record.requests !== '--' && !isNaN(record.requests) ? `${record.requests}%` : '--',
    },
    {
      title: t('table.memoryLimits'),
      key: 'limitsValue',
      render: (_, record) => record.memoryLimits !== '--' && !isNaN(record.memoryLimits) ? `${record.memoryLimits}MiB` : '--',
    },
    {
      title: t('table.memoryLimitsRate'),
      key: 'limitsRate',
      render: (_, record) => record.memoryLimitsRate !== '--' && !isNaN(record.memoryLimitsRate) ? `${record.memoryLimitsRate}%` : '--',
    },
  ], [t]);

  return <>
    <div className='monitor_chart_container' >
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.overview')}</p>
      <div className="monitor_overview">
        <div className="info_box" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div className="info_item">
            <div className="info_title" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.nodesNotReady')}</div>
            <div className="info_content">
              <img src={MonitorNodeUnNormal} />
              <div className="number_count" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{timelyData?.nodesNotReadyCount || 0}</div>
            </div>
          </div>
          <div className="info_item">
            <div className="info_title" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.nodesCount')}</div>
            <div className="info_content">
              <img src={MonitorNormal} />
              <div className="number_count" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{timelyData?.nodesCount || 0}</div>
            </div>
          </div>
          <div className="info_item">
            <div className="info_title" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.podsCount')}</div>
            <div className="info_content">
              <img src={MonitorPod} />
              <div className="number_count" style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{timelyData?.podsCount || 0}</div>
            </div>
          </div>
        </div>
        <div className='monitor_chart_container donut_chart'>
          {loaded
            ? <DataDonutChart
              className='empty_data_area'
              data={timelyData?.clusterCpuUtilisationRate}
              id="cluster_cpu_utilisation"
              type={'cluster'}
              title={t('resourceChart.cluster.cpuUtilization')}
              refreshFn={refreshFn}
            />
            : <SkeletonArea height='188px' />
          }
        </div>
      </div>
      <div className='monitor_chart_container'>
        <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.cpu')}</p>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.cluster.cpuUsage')}
            data={solveBytesToMiB(data?.cluster_cpu_usage_by_namespace, false)}
            type="line"
            rangeColorProps={propsColor}
            id="cluster_cpu_usage_by_namespace"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
      <div className='monitor_chart_container'>
        <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.memory')}</p>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.cluster.memoryUsage', { unit: solveUnitMonitor(data?.cluster_memory_usage_without_cache_by_namespace) })}
            data={solveBytesToMiB(data?.cluster_memory_usage_without_cache_by_namespace)}
            type="line"
            rangeColorProps={propsColor}
            id="cluster_memory_usage_by_namespace"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
      <div className='monitor_chart_container'>
        <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.memoryRequest')}</p>
        {loaded ? <Table
          className='chart_table_monitor ClusterMonitor'
          style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}
          columns={columns}
          dataSource={timelyData?.requests}
          pagination={{
            style: { paddingBottom: 0 },
            showTotal: total => t('table.total', { total }),
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showQuickJumper: true,
            className: 'page ClusterMonitor',
          }} />
          : <SkeletonArea />
        }
      </div>
      <div className='monitor_chart_container'>
        <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.packetRate')}</p>
        <div className='monitor_half'>
          {loaded
            ? <DataAreaChart
              title={t('resourceChart.cluster.receivedPacketRate')}
              data={solveBytesToMiB(data?.cluster_rate_of_received_packets_by_namespace, false)}
              type="line"
              styleProps={{ gridRight: '40%' }}
              id="cluster_rate_of_received"
              rangeColorProps={propsColor}
              refreshFn={refreshFn}
            />
            : <SkeletonArea />
          }
          {loaded
            ? <DataAreaChart
              title={t('resourceChart.cluster.transmittedPacketRate')}
              data={solveBytesToMiB(data?.cluster_rate_of_transmitted_packets_by_namespace, false)}
              type="line"
              styleProps={{ gridRight: '40%' }}
              id="cluster_rate_of_transmitted"
              rangeColorProps={propsColor}
              refreshFn={refreshFn}
            />
            : <SkeletonArea />
          }
        </div>
      </div>
      <div className='monitor_chart_container'>
        <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.cluster.droppedPacketRate')}</p>
        <div className='monitor_half'>
          {loaded
            ? <DataAreaChart
              title={t('resourceChart.cluster.receivedDroppedRate')}
              data={data?.cluster_rate_of_received_packets_dropped_by_namespace}
              type="line"
              styleProps={{ gridRight: '40%' }}
              id="rate_received_dropped_namespace"
              rangeColorProps={propsColor}
              refreshFn={refreshFn}
            />
            : <SkeletonArea />
          }
          {loaded
            ? <DataAreaChart
              title={t('resourceChart.cluster.transmittedDroppedRate')}
              data={data?.cluster_rate_of_transmitted_packets_dropped_by_namespace}
              type="line"
              styleProps={{ gridRight: '40%' }}
              id="rate_transmitted_dropped_namespace"
              rangeColorProps={propsColor}
              refreshFn={refreshFn}
            />
            : <SkeletonArea />
          }
        </div>
      </div>
    </div>
  </>;
}
