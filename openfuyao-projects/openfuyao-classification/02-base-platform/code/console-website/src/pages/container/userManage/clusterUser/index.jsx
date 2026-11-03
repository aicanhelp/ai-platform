/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Table, message } from 'antd';
import '@/styles/userManage/user.less';
import { useEffect, useState, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import { sorterFirstAlphabet } from '@/tools/utils';
import { getClusterRoleList } from '@/api/clusterApi';
import ToastMsg from '@/components/ToastMsg';

export default function ClusterUser() {
  const { t } = useTranslation('rbac');
  const [messageApi, contextHolder] = message.useMessage();
  const [clusterUserPage, setClusterUserPage] = useState(DEFAULT_CURRENT_PAGE);
  const [clusterUserListData, setClusterUserListData] = useState();

  const clusterUserColumns = useMemo(() => [
    {
      title: t('clusterRole.table.name'),
      key: 'member_name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => record.name || '--',
    },
    {
      title: t('clusterRole.table.description'),
      key: 'member_user',
      render: (_, record) => record.description || '--',
    },
  ], [t]);

  const getRoleList = async () => {
    try {
      const res = await getClusterRoleList();
      let dataList = [];
      if (res.status === ResponseCode.OK) {
        (res.data?.Data || []).map(item => {
          dataList.push({
            name: item.metadata.name,
            description: item.metadata.annotations?.description,
          });
        });
      }
      setClusterUserListData(dataList);
    } catch (e) {
      setClusterUserListData([]);
      messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e.response?.data?.message }));
    }
  };

  useEffect(() => {
    getRoleList();
  }, []);

  return <div className='child_content withBread_content'>
    <ToastMsg contextHolder={contextHolder} />
    <BreadCrumbCom className='create_bread' items={[
      { title: t('clusterRole.title'), path: '/', disabled: true },
    ]} />
    <div className="container_margin_box" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className='tab_table_flex cluster_container_height' style={{ paddingTop: '32px', height: 'calc(100vh - 200px)' }}>
        <Table
          className='table_padding'
          columns={clusterUserColumns}
          dataSource={clusterUserListData}
          pagination={{
            className: 'page',
            current: clusterUserPage,
            showTotal: (total) => t('pagination.total', { ns: 'common', total }),
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: [10, 20, 50],
            onChange: page => setClusterUserPage(page),
          }}
          scroll={{ x: 1280 }}
        />
      </div>
    </div>
  </div>;
}