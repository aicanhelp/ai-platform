/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/applicationMarket/index.less';
import { ResponseCode } from '@/common/constants';
import HelmIcon from '@/assets/images/helmIcon.png';
import {
  Table,
  Button,
  Space,
  Popover,
  message,
} from 'antd';
import { useEffect, useState, useStore } from 'openinula';
import { MoreOutlined } from '@ant-design/icons';
import {
  getAllVersionInfo,
  deleteHelmChartVersion,
  getAppointChartInfo,
} from '@/api/applicationMarketApi';
import { useLocation, useParams } from 'inula-router';
import Dayjs from 'dayjs';
import CreateVersion from '@/pages/applicationMarket/component/CreateVersion';
import { containerRouterPrefix } from '@/constant';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { sorterFirstAlphabet, showSuccessThen } from '@/tools/utils';
import { useTranslation } from 'react-i18next';
import ToastMsg from '@/components/ToastMsg';

export default function PackageManagementDetail() {
  const { t } = useTranslation('stash');
  const { repo, chart } = useParams();
  const { state } = useLocation();
  const { pkgDetail } = state || {};
  const [total, setTotal] = useState(0);
  const [popOpen, setPopOpen] = useState(false);
  const [dataSource, setDataSource] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [openAdd, setOpenAdd] = useState(false);
  const [detailsInfo, setDetailsInfo] = useState(null);
  const [isRepoDelCheck, setIsRepoDelCheck] = useState(false);
  const [repoDelModalOpen, setRepoDelModalOpen] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  const [deleteVersion, setDeleteVersion] = useState('');
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();

  const handleDelete = (record) => {
    setPopOpen('');
    setDeleteName(record.name);
    setDeleteVersion(record.version);
    setRepoDelModalOpen(true);
  };

  function handleDelRepoCancel() {
    setDeleteName('');
    setRepoDelModalOpen(false);
    setIsRepoDelCheck(false);
  }

  const handleRepoCheckFn = (e) => {
    setIsRepoDelCheck(e.target.checked);
  };

  const deleteReopFn = async () => {
    try {
      await deleteHelmChartVersion(deleteName, deleteVersion);
      showSuccessThen(messageApi, t('message.deleteVersionSuccess'), () => {
        getAllVersionInfoData();
        getAppointChartInfoDetails();
        setDeleteName('');
        setRepoDelModalOpen(false);
        setIsRepoDelCheck(false);
      });
    } catch (error) {
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('message.noPermission'));
      } else {
        messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
      }
    }
  };
  const columns = [
    {
      title: t('packageDetail.version'),
      key: 'version',
      sorter: (a, b) => sorterFirstAlphabet(a.version, b.version),
      render: (_, record) => record.version,
    },
    {
      title: t('packageDetail.addTime'),
      key: 'created',
      sorter: (a, b) => Dayjs(a.created) - Dayjs(b.created),
      render: (_, record) =>
        Dayjs(record.created).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: t('common.operation'),
      key: 'handle',
      width: 100,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Popover
            placement='bottom'
            className='packageManagementDetail'
            content={
              <div className='pop_modal'>
                <Button type='link' onClick={() => handleDelete(record)}>
                  {t('common.delete')}
                </Button>
              </div>
            }
            trigger='click'
            open={popOpen === `${record.name}${record.created}`}
            onOpenChange={(newOpen) =>
              newOpen
                ? setPopOpen(`${record.name}${record.created}`)
                : setPopOpen('')
            }
          >
            <MoreOutlined className='common_antd_icon primary_color packageManagementDetail' />
          </Popover>
        </Space>
      ),
    },
  ];

  const handleCreateVersion = () => {
    setOpenAdd((open) => !open);
    getAllVersionInfoData();
    getAppointChartInfoDetails();
  };

  const handleCancelVersion = () => {
    setOpenAdd((open) => !open);
  };

  const handleRefresh = () => {
    getAllVersionInfoData();
    getAppointChartInfoDetails();
    setOpenAdd((open) => !open);
  };

  const getAppointChartInfoDetails = async () => {
    try {
      const res = await getAppointChartInfo(chart);
      if (res.data?.data?.items?.length) {
        setDetailsInfo((res.data.data.items)[0].metadata);
      }
    } catch (error) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
    }
  };

  const handlePageChange = (page, pageSize) => {
    setCurrentPage(page); // 更新当前页码
    if (pageSize) {
      setLimit(pageSize);
    }
  };

  const getAllVersionInfoData = async () => {
    try {
      const res = await getAllVersionInfo(repo, chart, currentPage, limit);
      const infoData = (res.data?.data ?? []).map((el) => el.metadata);
      infoData.sort((a, b) => sorterFirstAlphabet(a.version, b.version));
      setDataSource(infoData);
      setTotal(res.data?.data?.length ?? 0);
    } catch (error) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
    }
  };

  useEffect(() => {
    getAllVersionInfoData();
  }, [repo, chart, currentPage, limit]);

  useEffect(() => {
    getAppointChartInfoDetails();
  }, [chart, repo]);

  return (
    <div className='child_content withBread_content'>
      <ToastMsg contextHolder={contextHolder} />
      <BreadCrumbCom
        items={[
          { title: t('module.wareHouseConfig') },
          { title: t('module.wareHouse'), path: `/${containerRouterPrefix}/appMarket/stash` },
          { title: t('module.packageDetail') },
        ]}
      />
      <div className='package_details_header' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
        <div>
          <img src={detailsInfo?.icon || HelmIcon} style={{ height: '48px', width: '48px' }} />
        </div>
        <div>
          <div className='application_description' style={{ fontWeight: 'bold', fontSize: '20px', minWidth: '250px', color: themeStore.$s.theme !== 'light' && '#fff' }}>{chart}</div>
          <div style={{ minWidth: '1000px', color: themeStore.$s.theme === 'light' ? '#89939b' : '#fff', fontSize: '14px', marginTop: '12px' }}>{detailsInfo?.description || pkgDetail?.description}</div>
        </div>
      </div>
      <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', margin: '0 32px' }}>
        <div className='version_management' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: themeStore.$s.theme !== 'light' && '#fff' }}>{t('packageDetail.versionManagement')}</div>
          <Button
            className='primary_btn'
            onClick={handleCreateVersion}
          >
            {t('packageDetail.addVersion')}
          </Button>
        </div>
        <div style={{ margin: '0 32px' }}>
          <Table dataSource={dataSource}
            columns={columns}
            pagination={{
              current: currentPage,
              pageSize: limit,
              pageSizeOptions: [10, 20, 50],
              showSizeChanger: true,
              showQuickJumper: true,
              onChange: handlePageChange,
              hideOnSinglePage: dataSource.length <= 10,
              showTotal: () => t('pagination.total', { ns: 'common', total: total }),
            }} />
        </div>
      </div>
      <CreateVersion
        openAdd={openAdd}
        onCancelFn={handleCancelVersion}
        onRefresh={handleRefresh}
      />
      <DeleteInfoModal
        title={t('common.deleteTitle')}
        cancelFn={handleDelRepoCancel}
        open={repoDelModalOpen}
        content={[
          t('modal.deleteWarning'),
          t('packageDetail.confirmDeleteVersion', { name: deleteName, version: deleteVersion }),
        ]}
        showCheck={true}
        isCheck={isRepoDelCheck}
        checkFn={handleRepoCheckFn}
        confirmFn={deleteReopFn} />
    </div>
  );
}
