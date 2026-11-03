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
import DataDonutChart from '@/components/echarts/DataDonutChart';
import { Table, Progress, Skeleton } from 'antd';
import { solveBytesToMiB, solveUnitMonitor, solveEachBytesToMB } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useStore } from 'openinula';
import { useTranslation } from 'react-i18next';

const colorsRange = {
  '0%': '#09aa71',
  '80%': '#f4840c',
};

const solveColor = (rate) => {
  if ((rate * 100).toFixed(1) > 80) {
    return colorsRange;
  } else if ((rate * 100) < 50) {
    return '#77AEF7';
  } else {
    return '#09aa71';
  }
};

export default function NodeMonitor({
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
      title: t('resourceChart.node.mounted'),
      width: 500,
      ellipsis: true,
      key: 'mounted',
      render: (_, record) => record.mounted,
    },
    {
      title: t('resourceChart.node.size'),
      key: 'size',
      render: (_, record) => `${solveEachBytesToMB(record.size)}MB`,
    },
    {
      title: t('resourceChart.node.available'),
      key: 'available',
      render: (_, record) => `${solveEachBytesToMB(record.available)}MB`,
    },
    {
      title: t('resourceChart.node.used'),
      key: 'used',
      render: (_, record) => `${solveEachBytesToMB(record.used)}MB`,
    },
    {
      title: t('resourceChart.node.usedPercent'),
      key: 'usedRate',
      render: (_, record) => {
        const rate = Number(record.usedRate);
        const isValid = Number.isFinite(rate);
        return <Progress percent={isValid ? (rate * 100).toFixed(1) : 0}
          strokeColor={() => solveColor(isValid ? rate : 0)} />;
      },
    },
  ];

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.node.cpu')}</p>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.node.cpuUsage')}
          data={solveBytesToMiB(data.node_cpu_usage, false)}
          type="line"
          id="node_cpu_usage"
          rangeColorProps={propsColor}
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      {loaded
        ? <DataAreaChart
          title={t('resourceChart.node.loadAverage')}
          data={data.nodeLoadUsage}
          type="line"
          id="node_load_average"
          rangeColorProps={propsColor}
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.node.memory')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('resourceChart.node.memoryUsage', { unit: solveUnitMonitor(data.node_memory_usage) })}
            data={solveBytesToMiB(data.node_memory_usage)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="node_memory_usage"
            rangeColorProps={propsColor}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        <div className='donut_chart'>
          {loaded ? <DataDonutChart
            className='empty_node_data'
            styleProps={{ height: 300 }}
            data={timelyData?.nodeMemoryUtilisationRate}
            type={'node'}
            id="node_memory_utilisation"
            title={t('resourceChart.node.memoryUtilization')}
            refreshFn={refreshFn}
          />
            : <SkeletonArea />
          }
        </div>
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.node.disk')}</p>
      {loaded ? <DataAreaChart
        title={t('resourceChart.node.diskIO')}
        data={solveBytesToMiB(data.disk, false)}
        type="line"
        id="node_disk_io"
        rangeColorProps={propsColor}
        refreshFn={refreshFn}
      />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container monitor_bk' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <p className='chart_title' style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('resourceChart.node.diskSpace')}</p>
      {loaded
        ? <Table className='chart_table_monitor NodeMonitor' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }} columns={columns} dataSource={timelyData?.diskUsage}
          pagination={{
            style: { paddingBottom: 0 },
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: [10, 20, 50],
            className: 'page NodeMonitor',
            showTotal: total => t('table.total', { total }),
          }} />
        : <Skeleton active />
      }
    </div>
  </>;
}
