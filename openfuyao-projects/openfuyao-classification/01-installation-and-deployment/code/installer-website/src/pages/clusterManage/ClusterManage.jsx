/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useState, Fragment } from 'openinula';
import ClusterManageTable from '@/pages/clusterManage/Index';

export default function ClusterManagePage() {
  const [hiddenPage, setHiddenPage] = useState(false);

  const handlePassMessage = (event) => {
    if (event.data.type === 'clusterPageDestory') {
      if (event.data.isDestory) {
        setHiddenPage(true);
        // 清理定时器
        const timeId = localStorage.getItem('clusterTimeManageId') || '';
        clearTimeout(timeId);
        localStorage.removeItem('clusterTimeManageId');
      } else {
        setHiddenPage(false);
      }
    }
  };

  useEffect(() => {
    window.addEventListener('message', handlePassMessage);
    return () => window.removeEventListener('message', handlePassMessage);
  }, []);

  return <Fragment>
    {!hiddenPage && <ClusterManageTable />}
  </Fragment>
} 