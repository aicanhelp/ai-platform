/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { DotChartOutlined } from '@ant-design/icons';
import { Skeleton } from 'antd';
import '@/styles/components/skeletonArea.less';
import { useStore } from 'openinula';

export default function SkeletonArea({ height = '300px' }) {
  const themeStore = useStore('theme');
  return <div
    className={'display_skeleton'}
    style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34' : '#fff' }}
  >
    <Skeleton.Input active />
    <Skeleton.Node active className='dataPic' style={{ height }}>
      <DotChartOutlined
        style={{
          fontSize: 40,
          color: '#bfbfbf',
        }}
      />
    </Skeleton.Node>
  </div>;
}