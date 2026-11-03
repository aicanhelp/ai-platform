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
import { Table } from 'antd';
import { solveBytesToMiB, solveUnitMonitor } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';

export default function ContainerMonitor({
  data,
  timelyData,
  propsColor,
  refreshFn,
  loaded,
}) {
  const themeStore = useStore('theme');
  const { t } = useTranslation('monitoring');

  const columns = [
    {
      title: t('resourceChart.container.containerColumn'),
      key: 'container',
      render: (_, record) => record.container ?? record.pod ?? '--',
    },
    {
      title: t('resourceChart.container.iopsReads'),
      key: 'iopsRead',
      render: (_, record) => {
        const value = Number(record.iopsRead);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}io/second`;
      },
    },
    {
      title: t('resourceChart.container.iopsReadWrite'),
      key: 'iopsReadWrite',
      id: 'iopsReadWrite',
      render: (_, record) => {
        const value = Number(record.iopsReadWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}io/second`;
      },
    },
    {
      title: t('resourceChart.container.throughputReads'),
      key: 'throughputRead',
      id: 'throughputRead',
      render: (_, record) => {
        const value = Number(record.throughputRead);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}bytes/second`;
      },
    },
    {
      title: t('resourceChart.container.throughputWrites'),
      key: 'throughputWrite',
      id: 'throughputWrite',
      render: (_, record) => {
        const value = Number(record.throughputWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}bytes/second`;
      },
    },
    {
      title: t('resourceChart.container.throughputReadWrite'),
      key: 'throughputReadWrite',
      id: 'throughputReadWrite',
      render: (_, record) => {
        const value = Number(record.throughputReadWrite);
        return `${Number.isFinite(value) && value !== -1 ? value : '--'}bytes/second`;
      },
    },
  ];

  return <>
    <div className='monitor_chart_container ContainerMonitor'>
      <p className='chart_title ContainerMonitor' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.container.cpu')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.container.cpuUsage')}
          data={solveBytesToMiB(data.pod_cpu_usage, false)}
          type="line"
          rangeColorProps={propsColor}
          id="container_cpu_usage"
          className='ContainerMonitor'
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container ContainerMonitor'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.container.memory')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.container.memoryUsage', { unit: solveUnitMonitor(data.pod_memory_usage) })}
          data={solveBytesToMiB(data.pod_memory_usage)}
          type="line"
          id="container_memory_usage"
          className='ContainerMonitor'
          rangeColorProps={propsColor}
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container ContainerMonitor'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.container.packetRate')}</p>
      <div className='monitor_half ContainerMonitor'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.container.receivedRate')}
            data={solveBytesToMiB(data.pod_network_packets_received, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="container_network_packets_received"
            className='ContainerMonitor'
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            data={solveBytesToMiB(data.pod_network_packets_transmitted, false)}
            title={t('resourceChart.container.transmittedRate')}
            rangeColorProps={propsColor}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="container_network_packets_transmitted"
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.container.droppedPacketRate')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.container.receivedDroppedRate')}
            data={solveBytesToMiB(data.pod_network_packets_received_dropped, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="container_network_packets_received_dropped"
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.container.transmittedDroppedRate')}
            data={solveBytesToMiB(data.pod_network_packets_transmitted_dropped, false)}
            type="line"
            id="container_network_packets_transmitted_dropped"
            rangeColorProps={propsColor}
            styleProps={{ gridRight: '40%' }}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.container.storageIO')}</p>
    <div className='monitor_chart_container' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      {loaded
        ? <Table className='chart_table_monitor ContainerMonitor'
          style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}
          columns={columns}
          dataSource={timelyData?.storage}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: total => t('table.total', { total }),
            style: { paddingBottom: 0 },
            pageSizeOptions: [10, 20, 50],
            className: 'page ContainerMonitor',
          }} />
        : <SkeletonArea />
      }
    </div>
  </>;
}
