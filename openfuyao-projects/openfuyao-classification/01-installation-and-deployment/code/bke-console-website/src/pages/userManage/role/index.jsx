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
import { ResponseCode } from '@/common/constants';
import { usermanageRouterPrefix } from '@/constant.js';
import zhCN from 'antd/es/locale/zh_CN';
import { message, Table, ConfigProvider } from 'antd';
import { useState, useEffect, useStore } from 'openinula';
import { useIntl } from 'inula-intl';
import { InfoCircleFilled } from '@ant-design/icons';
import '@/styles/userManage/user.less';
import '@/styles/userManage/dark.less';
import { getRoleList } from '@/api/clusterApi';
import { sorterFirstAlphabet } from '@/tools/utils';
export default function RoleManage() {
  const intl = useIntl();
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const [roleDetailTableList, setRoleDetailTableList] = useState();
  const roleDetailcolumns = [
    {
      title: intl.formatMessage({ id: 'role.roleName' }),
      dataIndex: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => record.name,
    },
    {
      title: intl.formatMessage({ id: 'role.description' }),
      dataIndex: 'role',
      render: (_, record) => record.description,
    },
  ];
  const getPlateRoleList = async () => {
    try {
      const res = await getRoleList();
      let datalist = [];
      if (res.status === ResponseCode.OK) {
        res.data.Data.map((item) => {
          datalist.push({
            name: item.metadata.name,
            description: item.metadata.annotations.description,
          });
        });
      }
      setRoleDetailTableList(datalist);
    } catch (e) {
      messageApi.error(intl.formatMessage({ id: 'role.roleListFetchError' }));
    }
  };
  useEffect(() => {
    getPlateRoleList();
  }, []);
  return (
    <div className="child_content user_detail" style={{ height: 'calc(100vh - 48px)' }}>
      <div
        style={{
          background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff',
          color: themeStore.$s.theme === 'dark' ? '#fff' : '#333',
        }}
      >
        {contextHolder}
      </div>
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'userManage.roleManage' }), path: `/${usermanageRouterPrefix}/role` },
        ]}
      />
      <div className="user_detail_tip">
        <InfoCircleFilled className="tip-icon" />
        <p className="user_detail_tip_text">{intl.formatMessage({ id: 'role.platformRoleTip' })}</p>
      </div>
      <ConfigProvider locale={zhCN}>
        <Table
          className="table_padding role_detail_info_table"
          columns={roleDetailcolumns}
          dataSource={roleDetailTableList}
          pagination={false}
        />
      </ConfigProvider>
    </div>
  );
}
