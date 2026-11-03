/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useTranslation } from 'react-i18next';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { ResponseCode } from '@/common/constants';
import { usermanageRouterPrefix } from '@/constant.js';
import { message, Table } from 'antd';
import { useState, useEffect, useStore, useMemo } from 'openinula';
import { InfoCircleFilled } from '@ant-design/icons';
import '@/styles/userManage/user.less';
import '@/styles/userManage/dark.less';
import { getRoleList } from '@/api/clusterApi';
import { sorterFirstAlphabet } from '@/tools/utils';

export default function RoleManage() {
  const { t } = useTranslation('setting');
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const [roleDetailTableList, setRoleDetailTableList] = useState();

  const roleDetailcolumns = useMemo(() => [
    {
      title: t('table.roleName'),
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (record.name),
    },
    {
      title: t('table.description'),
      dataIndex: 'description',
      render: (_, record) => (record.description),
    },
  ], [t]);

  const getPlateRoleList = async () => {
    try {
      const res = await getRoleList();
      let datalist = [];
      if (res.status === ResponseCode.OK) {
        (res.data.Data || []).map(item => {
          datalist.push({
            name: item.metadata.name,
            description: item.metadata.annotations.description,
          });
        });
      }
      setRoleDetailTableList(datalist);
    } catch {
      messageApi.error(t('messages.roleListError'));
    }
  };

  useEffect(() => {
    getPlateRoleList();
  }, []);

  return <div className="child_content user_detail" style={{ height: 'calc(100vh - 48px)' }}>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <BreadCrumbCom items={[{ title: t('module.roleManage'), path: `/${usermanageRouterPrefix}/role` }]} />
    <div className='user_detail_tip'>
      <InfoCircleFilled className='tip-icon' />
      <p className='user_detail_tip_text'>{t('tip.platformRoleOnly')}</p>
    </div>
    <Table
      className='table_padding role_detail_info_table'
      columns={roleDetailcolumns}
      dataSource={roleDetailTableList}
      pagination={false}
    />
  </div>;
}