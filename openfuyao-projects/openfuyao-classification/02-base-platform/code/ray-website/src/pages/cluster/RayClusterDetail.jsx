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
import Detail from '@/pages/cluster/detailComponent/Detail';
import Yaml from '@/pages/cluster/detailComponent/Yaml';
import Log from '@/pages/cluster/detailComponent/Log';
import Monitor from '@/pages/cluster/detailComponent/Monitor';
import { deleteClusterList, stopClusterList, restartClusterList, getClusterDetail } from '@/api/rayApi.js';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { useIntl } from 'inula-intl';
export default function RayClusterDetail() {
  const intl = useIntl();
  const location = useLocation();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [isShowClusterDetailOperation, setIsShowClusterDetailOperation] = useState(false);
  const clusterName = location.state.name;
  const clusterNameSpace = location.state.namespace;
  const [clusterStatus, setClusterStatus] = useState('');
  const [clusterDetailDelModalOpen, setClusterDetailDelModalOpen] = useState(false);
  const [isClusterDetailDelCheck, setIsClusterDetailDelCheck] = useState(false); // 是否选中

  const items = [
    {
      key: '1',
      label: (
        <Button
          type="link"
          disabled={clusterStatus !== '失败' && clusterStatus !== '未运行'}
          onClick={() => handleRestartClusterDetail()}
        >
          {intl.formatMessage({ id: 'common.start' })}
        </Button>
      ),
    },
    {
      key: '2',
      label: (
        <Button type="link" disabled={clusterStatus === '未运行'} onClick={() => handleStopClusterDetail()}>
          {intl.formatMessage({ id: 'common.stop' })}
        </Button>
      ),
    },
    {
      key: '3',
      label: (
        <Button type="link" disabled={clusterStatus === '运行中'} onClick={() => handleDeleteClusterDetail()}>
          {intl.formatMessage({ id: 'common.delete' })}
        </Button>
      ),
    },
  ];
  const clusterDetailTabItems = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'detail.tab.detail' }),
      children: <Detail className="defaultRayClusterDetail" />,
    },
    {
      key: '2',
      label: 'YAML',
      children: <Yaml className="defaultRayClusterDetail" />,
    },
    {
      key: '3',
      label: intl.formatMessage({ id: 'detail.tab.log' }),
      children: <Log className="defaultRayClusterDetail" />,
    },
    {
      key: '4',
      label: intl.formatMessage({ id: 'detail.clusterHealth' }),
      children: <Monitor clusterName={clusterName} />,
    },
  ];
  const handleClusterDetailOperation = (open) => {
    // 展示创建下拉菜单
    setIsShowClusterDetailOperation(open);
  };
  const handleRestartClusterDetail = async () => {
    try {
      const res = await restartClusterList(clusterNameSpace, clusterName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.cluster.startSuccess' }, { name: clusterName }));
        getClusterStatus();
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.startFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.startFail' })},${msg}`);
      }
    }
  };
  const handleStopClusterDetail = async () => {
    try {
      const res = await stopClusterList(clusterNameSpace, clusterName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.cluster.stopSuccess' }, { name: clusterName }));
        getClusterStatus();
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.stopFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.stopFail' })},${msg}`);
      }
    }
  };

  // 打开删除弹窗
  const handleDeleteClusterDetail = () => {
    setClusterDetailDelModalOpen(true);
  };
  // 取消删除
  const handleDelClusterDetailCancel = () => {
    setClusterDetailDelModalOpen(false);
    setIsClusterDetailDelCheck(false);
  };
  // 确认删除按钮删除
  const handleClusterDetailCheckFn = (e) => {
    setIsClusterDetailDelCheck(e.target.checked);
  };
  const deleteClusterDetail = async () => {
    try {
      const res = await deleteClusterList(clusterNameSpace, clusterName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.cluster.deleteSuccess' }, { name: clusterName }));
        setTimeout(() => {
          history.push('/RayCluster');
          setClusterDetailDelModalOpen(false);
        }, 500);
        setIsClusterDetailDelCheck(false);
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.deleteFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.cluster.deleteFail' })},${msg}`);
      }
    }
  };
  const getClusterStatus = async () => {
    const res = await getClusterDetail(clusterNameSpace, clusterName);
    try {
      if (res.data.code === ResponseCode.OK) {
        setClusterStatus(res.data.data.cluster_status_zhcn);
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
        messageApi.error(intl.formatMessage({ id: 'ray.cluster.detailError' }, { name: clusterName }) + ',' + msg);
      }
    }
  };
  useEffect(() => {
    getClusterStatus();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content service_detail">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayCluster', path: `/RayCluster` },
          { title: intl.formatMessage({ id: 'detail.title' }), path: `/detail` },
        ]}
      />
      {contextHolder}
      <div>
        <div className="service_detail_top defaultRayClusterDetail">
          <div className="service_detail_top_title_btn">
            <p className="service_detail_top_title defaultRayClusterDetail">{location.state.name}</p>
            <div>
              <Dropdown
                menu={{ items }}
                open={isShowClusterDetailOperation}
                className="defaultRayClusterDetail"
                onOpenChange={handleClusterDetailOperation}
                placement="bottom"
              >
                <Button className="primary_btn">
                  {intl.formatMessage({ id: 'detail.operation' })}
                  {isShowClusterDetailOperation ? <UpOutlined /> : <DownOutlined />}{' '}
                </Button>
              </Dropdown>
            </div>
          </div>
        </div>
        <Tabs defaultActiveKey="1" items={clusterDetailTabItems} />
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'ray.cluster.deleteTitle' })}
        open={clusterDetailDelModalOpen}
        cancelFn={handleDelClusterDetailCancel}
        content={[
          intl.formatMessage({ id: 'ray.cluster.deleteWarning' }),
          intl.formatMessage({ id: 'ray.cluster.deleteConfirm' }, { name: clusterName, extra: '' }),
        ]}
        showCheck={true}
        isCheck={isClusterDetailDelCheck}
        checkFn={handleClusterDetailCheckFn}
        confirmFn={deleteClusterDetail}
      />
    </div>
  );
}
