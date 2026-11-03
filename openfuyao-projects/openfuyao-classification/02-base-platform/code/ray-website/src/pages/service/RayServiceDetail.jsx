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
import { containerRouterPrefix, ResponseCode } from '@/common/constants';
import { useHistory, Link, useLocation } from 'inula-router';
import { useCallback, useEffect, useState } from 'openinula';
import { Button, Dropdown, Tabs, message } from 'antd';
import { DownOutlined, UpOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceDetail.less';
import Detail from '@/pages/service/detailComponent/Detail';
import Yaml from '@/pages/service/detailComponent/Yaml';
import Log from '@/pages/service/detailComponent/Log';
import { deleteServiceList, stopServiceList, restartServiceList, getServiceDetail } from '@/api/rayApi.js';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { useIntl } from 'inula-intl';
export default function RayServiceDetail() {
  const intl = useIntl();
  const location = useLocation();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [isShowServiceDetailOperation, setIsShowServiceDetailOperation] = useState(false);
  const serviceName = location.state.name;
  const serviceNameSpace = location.state.namespace;
  const [serviceStatus, setServiceStatus] = useState('');
  const [serviceDetailDelModalOpen, setServiceDetailDelModalOpen] = useState(false);
  const [isServiceDetailDelCheck, setIsServiceDetailDelCheck] = useState(false); // 是否选中
  const items = [
    {
      key: '1',
      label: (
        <Button
          type="link"
          disabled={serviceStatus !== '失败' && serviceStatus !== '未运行'}
          onClick={() => handlRestartServiceDetail()}
        >
          {intl.formatMessage({ id: 'common.start' })}
        </Button>
      ),
    },
    {
      key: '2',
      label: (
        <Button type="link" disabled={serviceStatus === '未运行'} onClick={() => handleStopServiceDetail()}>
          {intl.formatMessage({ id: 'common.stop' })}
        </Button>
      ),
    },
    {
      key: '3',
      label: (
        <Button type="link" disabled={serviceStatus === '运行中'} onClick={() => handleDeleteServiceDetail()}>
          {intl.formatMessage({ id: 'common.delete' })}
        </Button>
      ),
    },
  ];

  const serviceDetailTabItems = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'detail.tab.detail' }),
      children: <Detail className="defaultRayServiceDetail" />,
    },
    {
      key: '2',
      label: 'YAML',
      children: <Yaml className="defaultRayServiceDetail" />,
    },
    {
      key: '3',
      label: intl.formatMessage({ id: 'detail.tab.log' }),
      children: <Log className="defaultRayServiceDetail" />,
    },
  ];
  const handleServiceDetailOperation = (open) => {
    // 展示创建下拉菜单
    setIsShowServiceDetailOperation(open);
  };

  const handlRestartServiceDetail = async () => {
    try {
      const res = await restartServiceList(serviceNameSpace, serviceName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.service.startSuccess' }, { name: serviceName }));
        getServiceStatus();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.startFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.startFail' })},${msg}！`);
      }
    }
  };
  const handleStopServiceDetail = async () => {
    try {
      const res = await stopServiceList(serviceNameSpace, serviceName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.service.stopSuccess' }, { name: serviceName }));
        getServiceStatus();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.stopFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.stopFail' })},${msg}！`);
      }
    }
  };

  // 打开删除弹窗
  const handleDeleteServiceDetail = () => {
    setServiceDetailDelModalOpen(true);
  };
  // 取消删除
  const handleDelServiceDetailCancel = () => {
    setServiceDetailDelModalOpen(false);
    setIsServiceDetailDelCheck(false);
  };
  // 确认删除按钮删除
  const handleServiceDetailCheckFn = (e) => {
    setIsServiceDetailDelCheck(e.target.checked);
  };
  const deleteServiceDetail = async () => {
    try {
      const res = await deleteServiceList(serviceNameSpace, serviceName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.service.deleteSuccess' }, { name: serviceName }));
        setTimeout(() => {
          history.push('/RayService');
          setServiceDetailDelModalOpen(false);
        }, 500);
        setIsServiceDetailDelCheck(false);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.deleteFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.service.deleteFail' })},${msg}！`);
      }
    }
  };
  const getServiceStatus = async () => {
    const res = await getServiceDetail(serviceNameSpace, serviceName);
    try {
      if (res.data.code === ResponseCode.OK) {
        setServiceStatus(res.data.data.cluster_status_zhcn);
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(msg);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(intl.formatMessage({ id: 'ray.service.detailError' }, { name: serviceName }) + ',' + msg);
      }
    }
  };
  useEffect(() => {
    getServiceStatus();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content service_detail">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayService', path: `/RayService` },
          { title: intl.formatMessage({ id: 'detail.title' }), path: `/detail` },
        ]}
      />
      {contextHolder}
      <div>
        <div className="service_detail_top defaultRayServiceDetail">
          <div className="service_detail_top_title_btn">
            <p className="service_detail_top_title defaultRayServiceDetail">{location.state.name}</p>
            <div>
              <Dropdown
                menu={{ items }}
                open={isShowServiceDetailOperation}
                className="defaultRayServiceDetail"
                onOpenChange={handleServiceDetailOperation}
                placement="bottom"
              >
                <Button className="primary_btn">
                  {intl.formatMessage({ id: 'detail.operation' })}
                  {isShowServiceDetailOperation ? <UpOutlined /> : <DownOutlined />}{' '}
                </Button>
              </Dropdown>
            </div>
          </div>
        </div>
        <Tabs defaultActiveKey="1" items={serviceDetailTabItems} />
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'ray.service.deleteTitle' })}
        open={serviceDetailDelModalOpen}
        cancelFn={handleDelServiceDetailCancel}
        content={[
          intl.formatMessage({ id: 'ray.service.deleteWarning' }),
          intl.formatMessage({ id: 'ray.service.deleteConfirm' }, { name: serviceName, extra: '' }),
        ]}
        showCheck={true}
        isCheck={isServiceDetailDelCheck}
        checkFn={handleServiceDetailCheckFn}
        confirmFn={deleteServiceDetail}
      />
    </div>
  );
}
