/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import DataAreaChart from '@/components/echarts/DataAreaChart';
import { Table, Skeleton } from 'antd';
import { solveBytesToMiB, solveUnitMonitor } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';

export default function PodMonitor({
  data,
  timelyData,
  loaded,
  propsColor,
  refreshFn,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('monitoring');

  const columns = [
    {
      title: t('resourceChart.pod.podColumn'),
      key: 'Pod',
      render: (_, record) => record.pod,
    },
    {
      title: t('resourceChart.pod.iopsReads'),
      key: 'iopsRead',
      render: (_, record) => {
        const value = Number(record.iopsRead);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}io/second`;
      },
    },
    {
      title: t('resourceChart.pod.iopsWrites'),
      key: 'iopsWrite',
      render: (_, record) => {
        const value = Number(record.iopsWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}io/second`;
      },
    },
    {
      title: t('resourceChart.pod.iopsReadWrite'),
      key: 'iopsReadWrite',
      render: (_, record) => {
        const value = Number(record.iopsReadWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}io/second`;
      },
    },
    {
      title: t('resourceChart.pod.throughputReads'),
      key: 'throughputRead',
      render: (_, record) => {
        const value = Number(record.throughputRead);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}bytes/second`;
      },
    },
    {
      title: t('resourceChart.pod.throughputWrites'),
      key: 'throughputWrite',
      render: (_, record) => {
        const value = Number(record.throughputWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}bytes/second`;
      },
    },
    {
      title: t('resourceChart.pod.throughputReadWrite'),
      key: 'throughputReadWrite',
      render: (_, record) => {
        const value = Number(record.throughputReadWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}bytes/second`;
      },
    },
  ];

  return <>
    <div className='monitor_chart_container PodMonitor'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.pod.cpu')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.pod.cpuUsage')}
          data={solveBytesToMiB(data.pod_cpu_usage, false)}
          type="line"
          id="pod_cpu_usage"
          className='PodMonitor'
          rangeColorProps={propsColor}
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container PodMonitor'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.pod.memory')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.pod.memoryUsage', { unit: solveUnitMonitor(data.pod_memory_usage) })}
          data={solveBytesToMiB(data.pod_memory_usage)}
          rangeColorProps={propsColor}
          type="line"
          id="pod_memory_usage"
          refreshFn={refreshFn}
        />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.pod.packetRate')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.pod.receivedRate')}
            data={solveBytesToMiB(data.pod_network_packets_received, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="pod_network_packets_received"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.pod.transmittedRate')}
            data={solveBytesToMiB(data.pod_network_packets_transmitted, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="pod_network_packets_transmitted"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container PodMonitor'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.pod.droppedPacketRate')}</p>
      <div className='monitor_half PodMonitor'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.pod.receivedDroppedRate')}
            data={solveBytesToMiB(data.pod_network_packets_received_dropped, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            className='PodMonitor'
            rangeColorProps={propsColor}
            id="pod_network_packets_received_dropped"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            className='PodMonitor'
            title={t('resourceChart.pod.transmittedDroppedRate')}
            data={solveBytesToMiB(data.pod_network_packets_transmitted_dropped, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="pod_network_packets_transmitted_dropped"
            refreshFn={refreshFn}
            rangeColorProps={propsColor}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.pod.storageIO')}</p>
    <div className='monitor_chart_container' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      {loaded ?
        <Table className='chart_table_monitor PodMonitor'
          style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}
          columns={columns}
          dataSource={timelyData?.storage}
          pagination={{
            style: { paddingBottom: 0 },
            className: 'page PodMonitor',
            pageSizeOptions: [10, 20, 50],
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: total => t('table.total', { total }),
          }} />
        : <Skeleton active />
      }
    </div>
  </>;
}
